/*
 * IRGraph.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRGRAPH_H_
#define IRGRAPH_H_

#include <vector>
#include <iostream>
#include "TensorNode.h"
#include "OpNode.h"

namespace swc {

/**
 * @brief IR Node Graph class 
 */
template<typename Dtype>
class IRGraph {
 public:
  IRGraph(){};
  ~IRGraph(){};

  TensorNode<Dtype>* getTensorNode(int i) const { return _tensors[i]; }
  OpNode<Dtype>* getOpNode(int i) const { return _ops[i]; }

  TensorNode<Dtype>* getInNode(int i) const { return _inNodes[i]; }
  TensorNode<Dtype>* getOutNode(int i) const { return _outNodes[i]; }
  

  int getNumInTopoLevel(int i) const { 
    return _nodesByTopology[i].size(); 
  }
  std::vector<IRNode>& getNodeInTopoLevel(int i) const { 
    return _nodesByTopology[i]; 
  }
  IRNode* getNodeInTopo(int i, int j) const { 
    return _nodesByTopology[i][j]; 
  }
  
  //GraphStructure Construct Interface
  void pushTensorNode() {};
  template<typename T, typename... Types>
  void pushTensorNode(const T& firstArg, const Types&... args) {
    _tensors.push_back(firstArg);
    pushTensorNode(args...);
  }
 
  void delTensorNode() {};
  template<typename T, typename... Types>
  void delTensorNode(const T& firstArg, const Types&... args) {
    if (!delVecMember(_tensors, firstArg)) {
      std::cout << "Del Tensor Failed" << firstArg->name() << std::endl;
    }
    delTensorNode(args...);
  }

  void pushOpNode() {};
  template<typename T, typename... Types>
  void pushOpNode(const T& firstArg, const Types&... args) {
    _ops.push_back(firstArg);
    pushOpNode(args...);
  }

  void delOpNode() {};
  template<typename T, typename... Types>
  void delOpNode(const T& firstArg, const Types&... args) {
    if (!delVecMember(_ops, firstArg)) {
      std::cout << "Del Op Failed" << firstArg->name() << std::endl;
    }
    delOpNode(args...);
  }

  void pushInNode() {};
  template<typename T, typename... Types>
  void pushInNode(const T& firstArg, const Types&... args) {
    _inNodes.push_back(firstArg);
    pushInNode(args...);
  }

  void pushOutNode() {};
  template<typename T, typename... Types>
  void pushOutNode(const T& firstArg, const Types&... args) {
    _outNodes.push_back(firstArg);
    pushOutNode(args...);
  }


  inline const int ternsorNodeNum() const { return _tensors.size(); }
  inline const int opNodeNum() const { return _ops.size(); }
  inline const int inNodeNum() const { return _inNodes.size(); }
  inline const int outNodeNum() const { return _outNodes.size(); }
  inline const int topologyNum() const { return _nodesByTopology.size(); }

  void findInOut() {
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

  template<typename T>
  void updateTopology(T node) {
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

  void updateTopology() {
    typename std::vector<TensorNode<Dtype>* >::iterator tnIter;
    typename std::vector<OpNode<Dtype>* >::iterator opIter; 
    
    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++)
      (*tnIter)->setTopologyId(0);
    for (opIter = _ops.begin(); opIter != _ops.end(); opIter++)
      (*opIter)->setTopologyId(0);
    
    for (tnIter = _inNodes.begin(); tnIter != _inNodes.end(); tnIter++)
      updateTopology(*tnIter);
  }

  void updateTopoNodeList() {
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

 private:
  std::vector<TensorNode<Dtype>* > _tensors;
  std::vector<OpNode<Dtype>* > _ops;

  std::vector<TensorNode<Dtype>* > _inNodes;
  std::vector<TensorNode<Dtype>* > _outNodes;

  std::vector<std::vector<IRNode*> > _nodesByTopology;

};


} //namespace swc

#endif /* !IRGRAPH_H_ */
