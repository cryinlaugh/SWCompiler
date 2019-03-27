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

#include "common.h"

namespace swc {

//Forward declarations
template<typename Dtype> class TensorNode;
template<typename Dtype> class OpNode;
class IRNode;


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
  std::vector<IRNode*> getNodeInTopoLevel(int i) const { 
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


  inline const int tensorNodeNum() const { return _tensors.size(); }
  inline const int opNodeNum() const { return _ops.size(); }
  inline const int inNodeNum() const { return _inNodes.size(); }
  inline const int outNodeNum() const { return _outNodes.size(); }
  inline const int topologyNum() const { return _nodesByTopology.size(); }

  void findInOut();
  
  template<typename T> 
  void updateTopology(T node);
  
  void updateTopology();
  void updateTopoNodeList();

  IRGraph* clone() const;
  void setDeviceLabel(Device dev);
  Device getDeviceLabel() {return _dev; }

 private:
  std::vector<TensorNode<Dtype>* > _tensors;
  std::vector<OpNode<Dtype>* > _ops;

  std::vector<TensorNode<Dtype>* > _inNodes;
  std::vector<TensorNode<Dtype>* > _outNodes;

  std::vector<std::vector<IRNode*> > _nodesByTopology;

  Device _dev;

};


} //namespace swc

#endif /* !IRGRAPH_H_ */
