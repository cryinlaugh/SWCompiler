/*
 * IRGraph.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-12-04
 */
#include "IRGraph.h"

#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "op/dlOp/dlOp.h"

#include "common.h"
#include <cassert>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace swc {

IRNode *IRGraph::getNodeByName(std::string name) const {
    for (auto &node : _tensors)
        if (node->name() == name)
            return node;

    for (auto &node : _ops)
        if (node->name() == name)
            return node;
    return nullptr;
}

bool IRGraph::buildSubGraphs(TensorNode *in, TensorNode *out,
                             ParallelStrategy strategy, int axis, int num) {
    assert(strategy == ParallelStrategy::SLICE && "only support SLICE ");
    assert(axis == 0 && "only herizonnal SLICE ");

    auto inDims = in->getDims();
    size_t dimPerSub = inDims[/*axis*/ 0] / num;
    assert((inDims[0] % dimPerSub) == 0 && "");

    OpNode *subGNode = extractSubGraph(in, out);
    if (!subGNode)
        return false;

    auto subInDims = inDims;
    subInDims[axis] = dimPerSub;
    std::vector<size_t> *shape = new std::vector<size_t>();
    for (auto dim : subInDims)
        shape->push_back(dim);

    IRGraph *subG = ((SubGraphOp *)subGNode->getOp())->getGraph();
    auto *inNodeOfSubG = (TensorNode *)subG->getNodeByName(in->name() + "_sub");
    if (!inNodeOfSubG)
        return false;
    inNodeOfSubG->setTensor(new Tensor(new TensorShape(shape)));
    subG->initTensorNodes();

    for (int i = 1; i < num; i++) {
        // TensorNode reference to the same Tensor
        // OpNode reference to the same Op
        auto *subG_cp = subG->clone();
        inNodeOfSubG =
            (TensorNode *)subG_cp->getNodeByName(in->name() + "_sub");
        if (!inNodeOfSubG)
            return false;

        inNodeOfSubG->setTensor(new Tensor(new TensorShape(shape)));
        subG_cp->initTensorNodes();

        auto *subG_Op = new SubGraphOp();
        subG_Op->setGraph(subG_cp);
        auto *subGNode_cp = new OpNode("subG", subG_Op);

        for (auto &p : subGNode->getParentNodes())
            subGNode_cp->exlinkUpperNode(p);
        for (auto &c : subGNode->getChildNodes())
            c->exlinkUpperNode(subGNode_cp);

        this->pushOpNode(subGNode_cp);
    }
    return true;
}

OpNode *IRGraph::extractSubGraph(TensorNode *in, TensorNode *out) {

    std::cout << "extract SubGraph from " << in->name() << " to " << out->name()
              << "\n";

    std::unordered_set<IRNode *> found;
    std::queue<IRNode *> toVisit;
    found.insert(out);
    toVisit.push(out);

    while (!toVisit.empty()) {
        auto *cur = toVisit.front();
        toVisit.pop();

        if (cur == in)
            continue;

        for (auto child : cur->getParentNodes()) {
            if (!found.count(child)) {
                toVisit.push(child);
                found.insert(child);
            }
        }
    }

    if (!found.count(in)) {
        return nullptr;
    }

    IRGraph *subG = new IRGraph();
    SubGraphOp *subG_Op = new SubGraphOp();
    subG_Op->setGraph(subG);
    OpNode *subGNode = new OpNode("subG", subG_Op);

    for (auto irNode : found) {
        std::cout << "process node " << irNode->name() << "\n";
        if (irNode->nodeType() == OP_NODE) {
            auto *node = (OpNode *)irNode;
            subG->pushOpNode(node);
            this->delOpNode(node);
        } else if (irNode->nodeType() == TENSOR_NODE) {
            auto *node = (TensorNode *)irNode;
            if (node == in) {

                TensorNode *node_mirror = node->clone();
                node_mirror->setExternal(true);

                OpNode *scatter = new OpNode("scatter", new ScatterOp());
                scatter->exlinkUpperNode(node_mirror);

                TensorNode *node_sub = new TensorNode(
                    node->name() + "_sub",
                    new Tensor(node->getTensor()->getTensorShape()), scatter);
                node->replaceUseKeepOrder(node_sub);

                subG->pushTensorNode(node_mirror, node_sub);
                subG->pushOpNode(scatter);

                continue;
            }
            if (node == out) {
                // suppose TensorNode only have one ParentNode
                assert(node->parentNum() == 1 && "");

                TensorNode *node_sub = new TensorNode(
                    node->name() + "_sub",
                    new Tensor(node->getTensor()->getTensorShape()));
                node_sub->exlinkUpperNode(node->getParentNode(0));

                TensorNode *node_mirror = node->clone();
                node_mirror->setExternal(true);
                OpNode *gather = new OpNode("gather", new GatherOp());
                gather->exlinkUpperNode(node_sub);
                node_mirror->exlinkUpperNode(gather);

                subG->pushTensorNode(node_mirror, node_sub);
                subG->pushOpNode(gather);
                continue;
            }
            if (node->parentNum() == 0) {
                // parameter of Op. e.g. weight and bias of FC;
                // filter and bias of Conv

                TensorNode *node_mirror = node->clone();
                node_mirror->setExternal(true);
                OpNode *scatter = new OpNode("scatter", new ScatterOp());
                scatter->setRunOnce();
                scatter->exlinkUpperNode(node_mirror);

                TensorNode *node_sub = new TensorNode(
                    node->name() + "_sub",
                    new Tensor(node->getTensor()->getTensorShape()), scatter);
                node->replaceUseKeepOrder(node_sub);

                subG->pushTensorNode(node_mirror, node_sub);
                subG->pushOpNode(scatter);
                subGNode->exlinkUpperNode(node);

                continue;
            }

            subG->pushTensorNode(node);
            this->delTensorNode(node);
        } // TENSOR_NODE
    }     // for irNode : found

    for (auto c : in->getChildNodes()) {
        if (found.count(c))
            c->destroyUpperNode(in);
    }

    for (auto p : out->getParentNodes()) {
        std::cout << "destroy " << out->name() << "->" << p->name() << "\n";
        if (found.count(p))
            out->destroyUpperNode(p);
    }
    subGNode->exlinkUpperNode(in);
    out->exlinkUpperNode(subGNode);

    this->pushOpNode(subGNode);
    std::cout << "extract subGraph successfully\n";

    return subGNode;
}

//---------------------------------------------------------
void IRGraph::initTensorNodes() {
    updateTopology();

    for (int i = 0; i < topologyNum(); i++) {
        for (int j = 0; j < getNumInTopoLevel(i); j++) {
            auto *irNode = getNodeInTopo(i, j);
            if (irNode->nodeType() == OP_NODE) {
                auto *node = (OpNode *)irNode;
                auto *op = node->getOp();
                if (dynamic_cast<MatrixMatrixFCOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto *weight = (TensorNode *)node->getParentNode(1);
                    auto wdims = weight->getDims();
                    // TODO ref count for tensor to decide whether release
                    // tensor
                    weight->setTensor(new Tensor({idims[1], wdims[1]}));

                    auto *out = (TensorNode *)node->getChildNode(0);
                    out->setTensor(new Tensor({idims[0], wdims[1]}));
                }
                if (dynamic_cast<MatrixTanhOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto *out = (TensorNode *)node->getChildNode(0);
                    out->setTensor(new Tensor({idims[0], idims[1]}));
                }
                if (dynamic_cast<MatrixSoftmaxOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto *out = (TensorNode *)node->getChildNode(0);
                    out->setTensor(new Tensor({idims[0], idims[1]}));
                }
                if (dynamic_cast<ScatterOp *>(op)) {
                    // child reinit
                    auto *out = (TensorNode *)node->getChildNode(0);
                    // auto odims = out->getDims();
                    auto *shape = out->getTensor()->getTensorShape();
                    out->setTensor(new Tensor(shape));
                }
            }
        }
    }
}

void IRGraph::findInOut() {
    _inNodes.clear();
    _outNodes.clear();
    typename std::vector<TensorNode *>::iterator tnIter;

    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
        if ((*tnIter)->parentNum() == 0)
            _inNodes.push_back(*tnIter);
    }
    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
        if ((*tnIter)->childNum() == 0)
            _outNodes.push_back(*tnIter);
    }
}

template <typename T> void IRGraph::updateTopology(T node) {
    int currentTopoId = node->topologyId();
    /*std::cout << "Current Node: " << node->name()
      << " TopologyID: " << node->topologyId()
      << std::endl;*/
    for (int i = 0; i < node->childNum(); i++) {
        if (node->getChildNode(i)->topologyId() < currentTopoId + 1) {
            node->getChildNode(i)->setTopologyId(currentTopoId + 1);
            /*std::cout << "Update " << node->getChildNode(i)->name()
              << " TopologyID " << node->getChildNode(i)->topologyId()
              << " To " << currentTopoId + 1
              << std::endl;*/
            updateTopology(node->getChildNode(i));
        }
    }
}

void IRGraph::updateTopology() {
    findInOut();

    typename std::vector<TensorNode *>::iterator tnIter;
    typename std::vector<OpNode *>::iterator opIter;

    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++)
        (*tnIter)->setTopologyId(0);
    for (opIter = _ops.begin(); opIter != _ops.end(); opIter++)
        (*opIter)->setTopologyId(0);

    for (tnIter = _inNodes.begin(); tnIter != _inNodes.end(); tnIter++)
        updateTopology(*tnIter);

    updateTopoNodeList();
}

void IRGraph::updateTopoNodeList() {
    typename std::vector<TensorNode *>::iterator tnIter;
    typename std::vector<OpNode *>::iterator opIter;
    std::vector<std::vector<IRNode *>>::iterator ndIter;
    int topoId;
    std::vector<IRNode *> vtemp;

    for (ndIter = _nodesByTopology.begin(); ndIter != _nodesByTopology.end();
         ndIter++) {
        ndIter->clear();
    }
    _nodesByTopology.clear();

    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
        topoId = (*tnIter)->topologyId();
        while (topoId >= (int)_nodesByTopology.size()) {
            _nodesByTopology.push_back(vtemp);
        }
        _nodesByTopology[topoId].push_back(*tnIter);
    }
    for (opIter = _ops.begin(); opIter != _ops.end(); opIter++) {
        topoId = (*opIter)->topologyId();
        while (topoId >= (int)_nodesByTopology.size()) {
            _nodesByTopology.push_back(vtemp);
        }
        _nodesByTopology[topoId].push_back(*opIter);
    }
}

IRGraph *IRGraph::clone() const {
    // TODO: add topo check before clone
    IRGraph *graph = new IRGraph();
    /*
    for(int i=0; i<this->topologyNum(); i++){
        for(int j=0; j<this->getNumInTopoLevel(i); i++){
            auto node = this->getNodeInTopo(i, j);
            if(node->nodeType() == NodeType::OP_NODE){
                graph->pushOpNode(static_cast<OpNode*>(node)->clone());
            }else if(node->nodeType() == NodeType::TENSOR_NODE){
                graph->pushTensorNode(static_cast<TensorNode*>(node)->clone());
            }
        }
    }
    */
    std::unordered_map<TensorNode *, TensorNode *> tensors_map;
    std::unordered_map<OpNode *, OpNode *> ops_map;
    for (auto &N : _tensors) {
        TensorNode *tn = (N->isExternal()) ? N->clone() : N->deepClone();
        tensors_map[N] = tn;
        graph->pushTensorNode(tn);
    }
    for (auto &N : _ops) {
        OpNode *opn = N->clone();
        ops_map[N] = opn;
        graph->pushOpNode(opn);
    }

    // create links
    /*
    for(auto &N : _tensors){
        auto tn = tensors_map[N];
        for(int i=0; i<N->parentNum(); i++){
            auto it = ops_map.find((OpNode*)N->getParentNode(i));
            if(it != ops_map.end())
                tn->exlinkUpperNode(it->second);
        }
    }
    */

    // it worked, but remind that
    // static_cast may cause offset
    for (auto &N : _tensors) {
        auto tn = tensors_map[N];
        for (int i = 0; i < N->parentNum(); i++) {
            auto parent = ops_map[(OpNode *)N->getParentNode(i)];
            tn->exlinkUpperNode(parent);
        }
    }
    for (auto &N : _ops) {
        auto opn = ops_map[N];
        for (int i = 0; i < N->parentNum(); i++) {
            auto parent = tensors_map[(TensorNode *)N->getParentNode(i)];
            opn->exlinkUpperNode(parent);
        }
    }

    return graph;
}

void IRGraph::setDeviceLabel(Device dev) {
    _dev = dev;
    for (auto tnode : _tensors) {
        // suppose Device Graph, all TensorNodes(in degree=0)
        // should be mirror of cpu TensorNodes
        if (tnode->isExternal())
            std::cout << tnode->name() << " External skip\n";
        if (!tnode->isExternal())
            tnode->getLabel()->setDeviceLabel(dev.type, dev.id);
    }
    for (auto opnode : _ops) {
        opnode->getLabel()->setDeviceLabel(dev.type, dev.id);
    }
}
} // namespace swc
