/*
 * TensorNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef TENSORNODE_H
#define TENSORNODE_H


#include "../tensor/tensor.h"

namespace swc {

class TensorNode : public IRNode
{
  
  public:
    TensorNode();
    ~TensorNode();

    void setTensor(Tensor<Dtype>* tensor) {
      _tensor = tensor; 
    }

    TensorDtype* tensor() {
      return _tensor;
    }

  private:
    Tensor<Dtype>* _tensor; 
};

#endif /* !TENSORNODE_H */
