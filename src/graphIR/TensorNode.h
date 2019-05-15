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

class TensorNode : public IRNode {
  public:
    TensorNode() : tensor_(NULL){};
    explicit TensorNode(std::string name, IRNode *parent = nullptr)
        : IRNode(TENSOR_NODE, name, parent){};
    explicit TensorNode(std::string name, Tensor *tensor,
                        IRNode *parent = nullptr)
        : IRNode(TENSOR_NODE, name, parent), tensor_(tensor){};
    explicit TensorNode(std::string name,
                        const std::initializer_list<size_t> &shape,
                        IRNode *parent = nullptr)
        : IRNode(TENSOR_NODE, name, parent) {
        tensor_ = new Tensor(shape);
    }

    ~TensorNode(){};

    void destroy() { printf("free TensorNode:%s\n", name().c_str()); };

    void setTensor(Tensor *tensor) { tensor_ = tensor; }
    Tensor *getTensor() { return tensor_; }

    void setTraining(int train) { tensor_->setTraining(train); }
    int getTraining() const { return tensor_->getTraining(); }

    DataType getDataType() { return tensor_->getDataType(); }
    std::vector<unsigned long> getDims() { return tensor_->getDims(); }
    TensorNode *clone() const;
    TensorNode *deepClone() const;
    std::string toString() const;

  private:
    Tensor *tensor_{nullptr};
};

} // namespace swc
#endif /* !TENSORNODE_H_ */
