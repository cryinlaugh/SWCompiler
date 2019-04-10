/*
 * IRGraph.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-12-04
 */
#include "IRGraph.h"

#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"

#include "common.h"
#include <unordered_map>

namespace swc {

void IRGraph::findInOut() {
    _inNodes.clear();
    _outNodes.clear();
    typename std::vector<TensorNode* >::iterator tnIter; 

    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
        if ((*tnIter)->parentNum() == 0)
            _inNodes.push_back(*tnIter);
    }
    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
        if((*tnIter)->childNum() == 0)
            _outNodes.push_back(*tnIter);
    }
}

template<typename T>
void IRGraph::updateTopology(T node) {
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
    typename std::vector<TensorNode* >::iterator tnIter;
    typename std::vector<OpNode* >::iterator opIter; 

    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++)
        (*tnIter)->setTopologyId(0);
    for (opIter = _ops.begin(); opIter != _ops.end(); opIter++)
        (*opIter)->setTopologyId(0);

    for (tnIter = _inNodes.begin(); tnIter != _inNodes.end(); tnIter++)
        updateTopology(*tnIter);
}


void IRGraph::updateTopoNodeList() {
    typename std::vector<TensorNode* >::iterator tnIter; 
    typename std::vector<OpNode* >::iterator opIter; 
    std::vector<std::vector<IRNode*> >::iterator ndIter; 
    int topoId;
    std::vector<IRNode*> vtemp;

    for (ndIter = _nodesByTopology.begin();
            ndIter != _nodesByTopology.end(); 
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


IRGraph* IRGraph::clone() const{
    //TODO: add topo check before clone
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
    std::unordered_map<TensorNode*, TensorNode*> tensors_map;
    std::unordered_map<OpNode*, OpNode*> ops_map;   
    for(auto &N : _tensors){
        TensorNode *tn =  N->clone(); 
        tensors_map[N] =  tn; 
        graph->pushTensorNode(tn);
    }
    for(auto &N : _ops){
        OpNode *opn =  N->clone(); 
        ops_map[N] =  opn; 
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
    for(auto &N : _tensors){
        auto tn = tensors_map[N];
        for(int i=0; i<N->parentNum(); i++){
            auto parent = ops_map[(OpNode*)N->getParentNode(i)];
            tn->exlinkUpperNode(parent);
        }
    }
    for(auto &N : _ops){
        auto opn = ops_map[N];
        for(int i=0; i<N->parentNum(); i++){
            auto parent = tensors_map[(TensorNode*)N->getParentNode(i)];
            opn->exlinkUpperNode(parent);
        }
    }
    
    return graph;
}


void IRGraph::setDeviceLabel(Device dev){
    _dev = dev;
    for(auto tnode : _tensors){
        tnode->getLabel()->setDeviceLabel(dev.type, dev.id);
    } 
    for(auto opnode : _ops){
        opnode->getLabel()->setDeviceLabel(dev.type, dev.id);
    } 
}
} //namespace swc
