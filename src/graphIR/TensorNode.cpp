/*
 * TensorNode.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-11-23
 */

#include "TensorNode.h"

#include "graphIR/OpNode.h"
#include "graphIR/IRGraph.h"

#include "op/dlOp/dlOp.h"
#include "pass/AutodiffPass.h"

using namespace swc::op;
using namespace swc::pass;

namespace swc {
/// share tensor, that tensor_ point to
TensorNode *TensorNode::clone() const {
    TensorNode *tn = new TensorNode(name()+"_cp");
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
    // tn->setTensor(new Tensor(tensor_->getTensorShape()));
    Tensor *tensor = tensor_->clone();
    tn->setTensor(tensor);
    tn->setLabel(getLabel()); // mainly for training flag
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

void TensorNode::autoDiff(IRGraph* graph, 
        std::unordered_map<IRNode*, IRNode*> &gradNodeMap,
        void* methodParams, 
        pass::METHOD_TYPE methodType)
{
    SWLOG_DEBUG(4) << "TensorNode: " << name() <<
        " begin to autodiff" << std::endl;

    if (gradNodeMap.count(this)) {
        auto *N = gradNodeMap[this];
        
        // label::needTraining not set yet
        // if (!label->needTraining())
        if (!getTraining()) {
            SWLOG_DEBUG(4) << "Tensor need not to train. Passed..." << std::endl;
            return;
        }

        if (methodType == SGD_METHOD) {
            SWLOG_DEBUG(4) << "SGD generate..." << std::endl;
            auto *node_mirror = clone();

            auto *mom_t = new Tensor(this->getTensor()->getTensorShape());
            mom_t->setTensorInit(TensorInitType::CONSTANT, 0);
            auto *momentum = new TensorNode("momentum", mom_t);

            SGD_PARAMETERS* profile = (SGD_PARAMETERS*)methodParams;

            auto *sgdOp = new SGDOp(profile->lr, profile->decay,
                    profile->momentum, profile->batch);
            auto *SGDNode = new OpNode(this->name() + "_sgd", sgdOp);

            SGDNode->exlinkUpperNode(this, N, momentum);
            node_mirror->exlinkUpperNode(SGDNode);
            graph->pushOpNode(SGDNode);
            graph->pushTensorNode(node_mirror, momentum);

        }
        else if (methodType == ADAM_METHOD) {
            SWLOG_DEBUG(4) << "No adam method now. Passed..." << std::endl;
        }
        else {
            SWLOG_DEBUG(4) << "Illegal method type" << std::endl;
            abort();
        }
        return;

    }

    //End point tensor
    if (childNum() == 0) {
        SWLOG_DEBUG(4) << "End point tensor" << std::endl;

        //generate new tensors
        auto *tensor = getTensor();
        auto *N = new TensorNode(name() + "_grad",
                new Tensor(tensor->getTensorShape()));
        tensor->setTensorInit(TensorInitType::CONSTANT, 0);

        //update information
        gradNodeMap[this] = N;
        //graph->pushTensorNode(N);
        return;
    }

}


void TensorNode::checkValid() {

    unsigned int i;
    SWLOG_DEBUG(4) << "Checking connect validation for " 
        << this->name() << std::endl;
        
    OpNode* parentIter = (OpNode*)(this->getParentNode(0));
    for (i = 0; i < parentIter->getChildNodes().size(); i++) {
        if(this == parentIter->getChildNode(i)) {
            break;
        }
    }
    if (parentIter->getOp()->getOutputDims(i) 
            != this->getTensor()->getNDim()) {
        std::cout << " Warnning: The upper op"
            << parentIter->name() 
            << " with " << i << "th output dim:"
            << parentIter->getOp()->getOutputDims(i)
            << " while the current tensor "
            << this->name()
            << " with " << i << "th"
            << " dim:"
            << this->getTensor()->getNDim()
            << std::endl;
        //abort();
    }

}


} // namespace swc
