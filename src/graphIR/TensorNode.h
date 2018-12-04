/*
 * TensorNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef TENSORNODE_H
#define TENSORNODE_H


#include "tensor.h"

namespace swc {

template <typename Dtype>
class TensorNode : public IRNode<Dtype> 
{
  
  public:
    TensorNode();
    ~TensorNode();

    void setTensor(Tensor<Dtype> tensor) {
      _tensor = tensor; 
    }

    TensorDtype& Operation() {
      return _tensor;
    }

  private:
    Tensor<Dtype> _tensor; 
}

#endif /* !TENSORNODE_H */
