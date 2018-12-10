/*
 * TensorNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef TENSORNODE_H
#define TENSORNODE_H

#include "IRNode.h"
#include "../tensor/tensor.h"

namespace swc {

template <typename Dtype>
class TensorNode : public IRNode
{
  
  public:
    TensorNode() : _tensor(NULL) {};
    TensorNode(const char name[]) : IRNode(TENSOR_NODE, name) {};
    ~TensorNode(){};

    void setTensor(Tensor<Dtype>* tensor) {
      _tensor = tensor; 
    }

    Tensor<Dtype>* getTensor() {
      return _tensor;
    }

    std::string dotGen();

  private:
    Tensor<Dtype>* _tensor; 
};

template <typename Dtype>
std::string TensorNode<Dtype>::dotGen() {

  std::string tensorInfo = " [shape = record, ";

  std::string tensorName = name();
  int NDim = getTensor()->getNDim();  // get NDim through "getTensor()->getNDim()"

  // generate the tensorInfo
  tensorInfo = tensorInfo + "label = \"{Name: " + tensorName + " |" ;
  tensorInfo = tensorInfo + "NDim: " + std::to_string(NDim) + " |"; 
  
  for (int i = 0; i < NDim; ++i) {
    if (i < NDim-1) 
      tensorInfo = tensorInfo + "Dim[" + std::to_string(i) + "]:" + std::to_string(getTensor()->getDim(i)) + " |";
    else         
      tensorInfo = tensorInfo + "Dim[" + std::to_string(i) + "]:" + std::to_string(getTensor()->getDim(i)) + " }\"];";
  }

  return IRNode::dotGen(tensorInfo, ";\n");
}

} //namespace swc


#endif /* !TENSORNODE_H */
