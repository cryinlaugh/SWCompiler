/*
 * TensorNode.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-11-23
 */

#include "TensorNode.h"

namespace swc {
/// share tensor, that tensor_ point to
TensorNode *TensorNode::clone() const {
    TensorNode *tn = new TensorNode(name());
    tn->setTensor(tensor_);
    tn->setLabel(getLabel()); // mainly for training flag
    tn->setExternal(isExternal());
    return tn;
}

// TensorShape can be globally shared
// so we create new Tensor()
// but point to the same tenorshape
TensorNode *TensorNode::deepClone() const {
    TensorNode *tn = new TensorNode(name());
    tn->setTensor(new Tensor(tensor_->getTensorShape()));
    tn->setExternal(isExternal());
    return tn;
}

std::string TensorNode::toString() const {
    std::stringstream os;
    os << "TensorNode: " << name() << "\n"
       << "  tensorDim: " << tensor_->getNDim() << "\n  ";
    for (int i = 0; i < tensor_->getNDim(); i++)
        os << tensor_->getDim(i) << " ";

    return os.str();
}
} // namespace swc
