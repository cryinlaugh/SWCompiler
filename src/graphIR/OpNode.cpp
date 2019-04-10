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
OpNode* OpNode::clone() const{
    OpNode* opNode = new OpNode((name()+"_cp").c_str());
    opNode->setOp(op_->clone());
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

} //namespace swc

