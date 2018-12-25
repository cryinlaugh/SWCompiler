/*
 * IRGraph.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-12-04
 */
#include "IRGraph.h"

namespace swc {

template<typename Dtype>
void IRGraph<Dtype>::findInOut() {
  _inNodes.clear();
  _outNodes.clear();
  typename std::vector<TensorNode<Dtype>* >::iterator tnIter; 

  for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
    if ((*tnIter)->parentNum() == 0)
      _inNodes.push_back(*tnIter);
  }
  for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
    if((*tnIter)->childNum() == 0)
      _outNodes.push_back(*tnIter);
  }
}

template<typename Dtype>
template<typename T>
void IRGraph<Dtype>::updateTopology(T node) {
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

template<typename Dtype>
void IRGraph<Dtype>::updateTopology() {
  typename std::vector<TensorNode<Dtype>* >::iterator tnIter;
  typename std::vector<OpNode<Dtype>* >::iterator opIter; 

  for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++)
    (*tnIter)->setTopologyId(0);
  for (opIter = _ops.begin(); opIter != _ops.end(); opIter++)
    (*opIter)->setTopologyId(0);

  for (tnIter = _inNodes.begin(); tnIter != _inNodes.end(); tnIter++)
    updateTopology(*tnIter);
}

template<typename Dtype>
void IRGraph<Dtype>::updateTopoNodeList() {
  typename std::vector<TensorNode<Dtype>* >::iterator tnIter; 
  typename std::vector<OpNode<Dtype>* >::iterator opIter; 
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

INSTANTIATE_CLASS(IRGraph);

} //namespace swc
