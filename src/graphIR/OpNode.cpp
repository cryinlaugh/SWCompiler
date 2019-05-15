/*
 * OpNode.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-11-23
 */

#include "OpNode.h"

namespace swc {
/// must clone op_ because destructed in ctor
OpNode *OpNode::clone() const {
    OpNode *opNode = new OpNode(name());
    opNode->setOp(op_);
    return opNode;
}

// we may want to clone scatter node
// but reset ScatterOp::Offset
// ConvOp::kernels...
// so we have to implement clone or copy ctor
// for every dlOp?
// TODO solve this problem
OpNode *OpNode::deepClone() const {
    OpNode *opNode = new OpNode(name());
    opNode->setOp(op_);
    return opNode;
}

std::string OpNode::toString() const {
    std::stringstream os;
    os << "OpNode " << name() << "\n"
       << "  op: " << op_->getOpName() << "\n"
       << "    nInput : " << op_->getnInput() << "\n"
       << "    nOutput: " << op_->getnOutput();
    return os.str();
}

} // namespace swc
