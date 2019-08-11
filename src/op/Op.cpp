/*************************************************************************
	> File Name: Op.cpp
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: 二 12/ 4 15:57:16 2018
 ************************************************************************/

#include "Op.h"

#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"
#include "tensor/tensor.h"
#include <sstream>

using namespace swc::op;

std::string Op::getOpInfo() {
    std::ostringstream stream;
    stream << "Operation: "+ _opClassName + "\\n"
        << "nInput : " << _nInput << "\\n"
        << "nOutput: " << _nOutput << "\\n";

    return stream.str();
}

bool Op::check() {
    if (_nInputTensor != _nInput)
        return false;
    if (_nOutputTensor != _nOutput)
        return false;
    for (int i = 0; i < _nInput; i++) {
        if (_inputTensors[i]->getNDim() != _inputNDims[i])
            return false;
    }
    for (int i = 0; i < _nOutput; i++) {
        if (_outputTensors[i]->getNDim() != _inputNDims[i])
            return false;
    }
    return true;
}

void Op::checkValid(OpNode *node) {

    SWLOG_DEBUG(4) << "Checking connect validation for "
        << node->name() << "by general" << std::endl;

    unsigned int i;

    for (i = 0; i < node->getParentNodes().size(); i++) {
        TensorNode* parentIter = (TensorNode*)(node->getParentNode(i));

        if (parentIter->getTensor()->getNDim()
                != node->getOp()->getInputDims(i)) {
            std::cout << "FATAL ERROR: The "
                << i << "th input tensor "
                << parentIter->name()
                << " with dim:"
                << parentIter->getTensor()->getNDim()
                << " while the current op "
                << node->name()
                << " with " << i << "th"
                << " dim:"
                << node->getOp()->getInputDims(i)
                << std::endl;
            abort();
        }
    }

}

void Op::outTensorShapeGen(OpNode* node,
                           size_t index,
                           TensorShape* tShape)
{
    TensorNode *inNode = (TensorNode*)node->getParentNode(0);
    std::vector<size_t> shape;

    TensorShape* inShape = inNode->getTensor()->getTensorShape();
    for (int i = 0; i < inShape->getNDim(); i++) {
        shape.push_back(inShape->getDim(i));
    }
    tShape->setShape(shape);
}
