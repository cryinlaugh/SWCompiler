/*
 * OpNode.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-11-23
 */

#include "OpNode.h"

#include "graphIR/TensorNode.h"

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


void OpNode::checkValid()
{
    unsigned int i;
    SWLOG_DEBUG(4) << "Checking connect validation for " 
        << this->name() << std::endl;
    for (i = 0; i < this->getParentNodes().size(); i++) {
        TensorNode* parentIter = (TensorNode*)(this->getParentNode(i));
        
        if (parentIter->getTensor()->getNDim() 
                != this->getOp()->getInputDims(i)) {
            std::cout << "Warnning: The "
                << i << "th input tensor "
                << parentIter->name()
                << " with dim:"
                << parentIter->getTensor()->getNDim()
                << " while the current op "
                << this->name()
                << " with " << i << "th"
                << " dim:"
                << this->getOp()->getInputDims(i)
                << std::endl;
            //abort();
        }
    }

}


} // namespace swc
