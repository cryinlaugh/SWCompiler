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
//#include "SWDSL.h"
#include <sstream>

namespace swc {

class TensorNode : public IRNode
{
  public:
    TensorNode() : tensor_(NULL) {};
    explicit TensorNode(const char name[], IRNode *parent = nullptr) : IRNode(TENSOR_NODE, name, parent) {};
    explicit TensorNode(const char name[], Tensor *tensor, IRNode *parent = nullptr) : IRNode(TENSOR_NODE, name, parent), tensor_(tensor) {};
    explicit TensorNode(const char name[], const std::initializer_list<int> &shape, IRNode *parent = nullptr) : IRNode(TENSOR_NODE, name, parent){    
        tensor_ = new Tensor(shape);
    }

    ~TensorNode(){};

    void destroy(){
        printf("free TensorNode:%s\n", name().c_str());
    };

    void setTensor(Tensor* tensor) {
      tensor_ = tensor; 
    }

    Tensor* getTensor() {
      return tensor_;
    }

    DataType getDataType() { return tensor_->getDataType(); }
    std::vector<unsigned long> getDims() { return tensor_->getDims(); }
    TensorNode *clone() const;
    std::string toString() const;

  private:
    Tensor* tensor_; 
};

} //namespace swc
#endif /* !TENSORNODE_H_ */
