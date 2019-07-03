/*
 * TensorNode.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-11-23
 */

#include "TensorNode.h"

#include "op/dlOp/dlOp.h"
#include "pass/AutodiffPass.h"
#include "graphIR/OpNode.h"

using namespace swc::op;
using namespace swc::pass;

namespace swc {
/// share tensor, that tensor_ point to
TensorNode *TensorNode::clone() const {
    TensorNode *tn = new TensorNode(name()+"_clone");
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
    SWLOG_INFO << "TensorNode: " << name() <<
        " begin to autodiff" << std::endl;

    if (gradNodeMap.count(this)) {
        auto *N = gradNodeMap[this];
        
        // label::needTraining not set yet
        // if (!label->needTraining())
        if (!getTraining()) {
            SWLOG_INFO << "Tensor need not to train. Passed..." << std::endl;
            return;
        }

        if (methodType == SGD_METHOD) {
            SWLOG_INFO << "SGD generate..." << std::endl;
            auto *node_mirror = clone();

            auto *mom_t = new Tensor(this->getTensor()->getTensorShape());
            mom_t->setTensorInit(TensorInitType::CONSTANT, 0);
            auto *momentum = new TensorNode("momentum", mom_t);

            SGD_PARAMETERS* profile = (SGD_PARAMETERS*)methodParams;

            auto *sgdOp = new SGDOp(profile->lr, profile->decay,
                    profile->momentum, 128);
            auto *SGDNode = new OpNode(this->name() + "_sgd", sgdOp);

            SGDNode->exlinkUpperNode(this, N, momentum);
            node_mirror->exlinkUpperNode(SGDNode);
            graph->pushOpNode(SGDNode);
            graph->pushTensorNode(node_mirror, momentum);

        }
        else if (methodType == ADAM_METHOD) {
            SWLOG_INFO << "No adam method now. Passed..." << std::endl;
        }
        else {
            SWLOG_INFO << "Illegal method type" << std::endl;
            abort();
        }
        return;

    }

    //End point tensor
    if (childNum() == 0) {
        SWLOG_INFO << "End point tensor" << std::endl;

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

} // namespace swc
