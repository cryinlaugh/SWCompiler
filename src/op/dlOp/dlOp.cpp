/*
 * dlOp.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-07-16
 */

#include "dlOp.h"

#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"


void ReluOp::checkValid(OpNode *node) 
{
    
    SWLOG_DEBUG(4) << "Checking connect validation for " 
        << node->name() << std::endl;
    assert(node->parentNum() == 1 &&
            "Relu input should be 1: data");
    
    TensorNode* parent = (TensorNode*)(node->getParentNode(0));

    if (parent->getTensor()->getNDim() != 4) {

        SWLOG_DEBUG(5) << "Customized the relu Op"
            << " with input and output dimensions:"
            << parent->getTensor()->getNDim() << std::endl;
        this->_inputNDims[0] = parent->getTensor()->getNDim();
        this->_outputNDims[0] = this->_inputNDims[0];
    }
}

void Conv2dOp::outTensorShapeGen(OpNode* node,
                                 size_t index, 
                                 TensorShape* tShape)
{
    std::vector<size_t> idims = 
        ((TensorNode *)node->getParentNode(0))->getDims();
    std::vector<size_t> wdims = 
        ((TensorNode *)node->getParentNode(1))->getDims();
    std::vector<size_t> kernels = ((Conv2dOp*)node->getOp())->getKernels();
    std::vector<size_t> strides = ((Conv2dOp*)node->getOp())->getStrides();
    std::vector<size_t> pads = ((Conv2dOp*)node->getOp())->getPads();
    
    assert(kernels.size() == 2);
    assert(strides.size() == 2);
    assert(pads.size() == 4);

    size_t oh = ((idims[1] + pads[0] + pads[2] - kernels[0]) / strides[0] + 1);
    size_t ow = ((idims[2] + pads[1] + pads[3] - kernels[1]) / strides[1] + 1);

    std::vector<size_t> shape;
    shape.push_back(idims[0]);
    shape.push_back(oh);
    shape.push_back(ow);
    shape.push_back(wdims[0]);

    tShape->setShape(shape);

}
