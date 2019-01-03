/*
 * TensorNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef TENSORNODE_H_
#define TENSORNODE_H_

#include "IRNode.h"
#include "tensor/tensor.h"

namespace swc {

template <typename Dtype>
class TensorNode : public IRNode
{
  
  public:
    TensorNode() : _tensor(NULL) {};
    explicit TensorNode(const char name[]) : IRNode(TENSOR_NODE, name) {};
    ~TensorNode(){};

    void destroy(){
        printf("free TensorNode:%s\n", name().c_str());
    };

    void setTensor(Tensor<Dtype>* tensor) {
      _tensor = tensor; 
    }

    Tensor<Dtype>* getTensor() {
      return _tensor;
    }

  private:
    Tensor<Dtype>* _tensor; 
};

} //namespace swc


#endif /* !TENSORNODE_H_ */
