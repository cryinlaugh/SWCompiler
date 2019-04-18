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
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cassert>

namespace swc {

IRNode* IRGraph::getNodeByName(std::string name) const {
    for(auto &node : _tensors)
        if(node->name() == name)
            return node;

    for(auto &node : _ops)
        if(node->name() == name)
            return node;
    return nullptr;
}
    bool buildSubGraph(TensorNode *in, TensorNode *out,
                        ParallelStrategy strategy, 
                        int axis=0,
                        int num=2);
bool IRGraph::buildSubGraph(TensorNode *in, TensorNode *out,
                            ParallelStrategy strategy, 
                            int axis,
                            int num) {

    std::cout << "begin build SubGraph\n";

    std::vector<IRNode *> subGraphNodes;
    std::unordered_set<IRNode *> found; 
    std::queue<IRNode *> toVisit; 

    found.insert(out);
    toVisit.push(out);

    while(!toVisit.empty()){
        auto *cur = toVisit.front();
        toVisit.pop();

        if(cur == in)
            continue;

        for(auto child : cur->getParentNodes()){
           if(!found.count(child)){
                toVisit.push(child);
                found.insert(child);
           }
        }
    }

    std::cout << "BFS ok\n";
    if(found.count(in)){
        assert(strategy == ParallelStrategy::SLICE && "only support SLICE ");
        assert(axis == 0 && "only herizonnal SLICE ");

        auto inDims = in->getDims();
        size_t dimPerSub = inDims[/*axis*/0] / num;
        assert((inDims[0]%dimPerSub)==0 && "");
        //size_t lastSubDim = (inDims[0]%dimPerSub) ? (inDims[0]%dimPerSub) : dimPerSub;

        IRGraph *subG = new IRGraph();
        SubGraphOp *subG_Op = new SubGraphOp();
        subG_Op->setGraph(subG);
        OpNode *subGNode = new OpNode("subG", subG_Op); 

        for(auto irNode : found){
            std::cout << "process node " << irNode->name() << "\n";
            if(irNode->nodeType() == OP_NODE){
                auto *node = (OpNode*)irNode;
                subG->pushOpNode(node); 
                this->delOpNode(node); 
            }else if(irNode->nodeType() == TENSOR_NODE){
                auto *node = (TensorNode *)irNode;
                if(node == in) {

                    TensorNode *node_mirror = node->clone();  
                    node_mirror->setExternal(true);

                    OpNode *scatter = new OpNode("scatter", new ScatterOp()); 
                    scatter->exlinkUpperNode(node_mirror);

                    TensorNode *node_sub = new TensorNode(node->name()+"_sub", new Tensor(node->getTensor()->getTensorShape()), scatter);
                    node->replaceUseKeepOrder(node_sub);

                    subG->pushTensorNode(node_mirror, node_sub); 
                    subG->pushOpNode(scatter); 

                    continue;
                }
                if(node == out) {
                    // suppose TensorNode only have one ParentNode
                    assert(node->parentNum()==1 && ""); 

                    TensorNode *node_sub = new TensorNode(node->name()+"_sub", new Tensor(node->getTensor()->getTensorShape()));
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
                if(node->parentNum() == 0){
                    // parameter of Op. e.g. weight and bias of FC; 
                    // filter and bias of Conv

                    TensorNode *node_mirror = node->clone();  
                    node_mirror->setExternal(true);
                    OpNode *scatter = new OpNode("scatter", new ScatterOp()); 
                    scatter->exlinkUpperNode(node_mirror);

                    TensorNode *node_sub = new TensorNode(node->name()+"_sub", new Tensor(node->getTensor()->getTensorShape()), scatter);
                    node->replaceUseKeepOrder(node_sub);

                    subG->pushTensorNode(node_mirror, node_sub); 
                    subG->pushOpNode(scatter); 
                    subGNode->exlinkUpperNode(node);

                    continue;
                }

                subG->pushTensorNode(node); 
                this->delTensorNode(node); 
            }
        }


        for(auto c : in->getChildNodes()){
            c->destroyUpperNode(in);
        }
        
        for(auto p : out->getParentNodes()){
            std::cout << "destroy " << out->name()
                    << "->" << p->name() << "\n";
            out->destroyUpperNode(p);
            std::cout << p->name() << " has childs " << p->childNum() << "\n";
            std::cout << p->getChildNode(0)->name() << "\n";
        }
        subGNode->exlinkUpperNode(in);
        out->exlinkUpperNode(subGNode);

        this->pushOpNode(subGNode); 
        std::cout << "build subGraph successfully\n";
        return true;
    }
    return false;
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
        TensorNode *tn = N->clone();
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
        if(!tnode->isExternal())
            tnode->getLabel()->setDeviceLabel(dev.type, dev.id);
    }
    for (auto opnode : _ops) {
        opnode->getLabel()->setDeviceLabel(dev.type, dev.id);
    }
}
} // namespace swc
