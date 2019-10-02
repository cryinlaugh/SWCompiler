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

#include "parallel/TilingLabel.h"

using namespace swc::op;
using namespace swc::pass;

namespace swc {

void TensorNode::destroy() {
    //printf("free TensorNode:%s\n", name().c_str());
    getLabel()->destroy();
    getTensor()->destroy();
    _tilingLabel = nullptr;
    SWLOG_DEBUG(4) << "Destroy TensorNode: " << name() << "\n";
};

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
    tn->setTilingLabel(_tilingLabel);
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
            // 19.10.2 let SGD not output (because N is actually inout, 
            // add mirror_node will cause difficulty for parallel strategy selection)
            // auto *node_mirror = clone();

            // graph->addLogicalOutNodes(node_mirror);

            auto *mom_t = new Tensor(this->getTensor()->getTensorShape());
            mom_t->setTensorInit(TensorInitType::CONSTANT, 0);
            auto *momentum = new TensorNode("momentum", mom_t);

            SGD_PARAMETERS* profile = (SGD_PARAMETERS*)methodParams;

            auto *sgdOp = new SGDOp(profile->lr, profile->decay,
                    profile->momentum, profile->batch);
            auto *SGDNode = new OpNode(this->name() + "_sgd", sgdOp);

            SGDNode->exlinkUpperNode(this, N, momentum);
            // node_mirror->exlinkUpperNode(SGDNode);
            graph->pushOpNode(SGDNode);
            graph->pushTensorNode(momentum);
            // graph->pushTensorNode(node_mirror, momentum);
            graph->addLogicalOutNodes(SGDNode);

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

    SWLOG_DEBUG(4) << "Checking connect validation for " 
        << this->name() << std::endl;

    size_t i;
    OpNode* parent = (OpNode*)(this->getParentNode(0));
    for (i = 0; i < parent->getChildNodes().size(); i++) {
        if(this == parent->getChildNode(i)) {
            break;
        }
    }
    // uninitialized tensors
    if (this->getTensor()->getNDim() == 1 &&
            this->getTensor()->getDim(0) == 0) {

        SWLOG_DEBUG(4) << "Uninitialized tensor, "
            << "acquire the tensor shape from upper op:" 
            << parent->name() << std::endl;
        
        parent->outTensorShapeGen(i, this->getTensor()->getTensorShape());
    
        TensorShape* tshapeGen = this->getTensor()->getTensorShape();
        std::stringstream ss;
        for (int i = 0; i < tshapeGen->getNDim(); i++) {
            ss << " " << tshapeGen->getDim(i) << " ";
        }
        SWLOG_DEBUG(4) << "Infer tensor shape by:" << ss.str() << std::endl;
    }

    if (parent->getOp()->getOutputDims(i) 
            != this->getTensor()->getNDim()) {
        std::cout << "FATAL ERROR: The upper op "
            << parent->name() 
            << " with " << i << "th output dim:"
            << parent->getOp()->getOutputDims(i)
            << " while the current tensor "
            << this->name()
            << " with " << i << "th"
            << " dim:"
            << this->getTensor()->getNDim()
            << std::endl;
        abort();
    }

}


} // namespace swc
