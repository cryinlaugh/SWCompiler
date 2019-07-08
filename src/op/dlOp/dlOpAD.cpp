/*************************************************************************
	> File Name: dlOp.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:35 2018
 ************************************************************************/

#include "dlOp.h"

#include <cassert>

#include "SWDSL.h"
#include "graphIR/IRGraph.h"
#include "graphIR/IRNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"

using namespace swc::op;

/*--------------------------------Auto Diff ------------------------
 * 
 * 
 * 
 * ----------------------------------------------------------------*/
void MatrixMatrixFCBiasOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName   << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *weight = opNode->getParentNode(1);
    auto *bias = opNode->getParentNode(2);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) &&
            "grad of FC output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad",
            new MatrixMatrixFCGradOp());
    N->exlinkUpperNode(input, weight, bias, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
    
    for (int i = 0; i < opNode->parentNum(); i++) {

        auto *tnode = (TensorNode *)(opNode->getParentNode(i));
        auto *tensor = tnode->getTensor();
        auto *N = new TensorNode(tnode->name() + "_grad",
                new Tensor(tensor->getTensorShape()),
                gradNodeMap[opNode]);

        SWLOG_DEBUG(4) << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}
void MatrixMatrixFCOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName   << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *weight = opNode->getParentNode(1);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) &&
            "grad of FC output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad",
            new MatrixMatrixFCBiasGradOp());
    N->exlinkUpperNode(input, weight, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
    
    for (int i = 0; i < opNode->parentNum(); i++) {

        auto *tnode = (TensorNode *)(opNode->getParentNode(i));
        auto *tensor = tnode->getTensor();
        auto *N = new TensorNode(tnode->name() + "_grad",
                new Tensor(tensor->getTensorShape()),
                gradNodeMap[opNode]);

        SWLOG_DEBUG(4) << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}


void ReluOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName   << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) &&
            "grad of Relu output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N =
        new OpNode(opNode->name() + "_grad", new ReluGradOp());
    N->exlinkUpperNode(input, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
    
    for (int i = 0; i < opNode->parentNum(); i++) {

        auto *tnode = (TensorNode *)(opNode->getParentNode(i));
        auto *tensor = tnode->getTensor();
        auto *N = new TensorNode(tnode->name() + "_grad",
                new Tensor(tensor->getTensorShape()),
                gradNodeMap[opNode]);

        SWLOG_DEBUG(4) << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}

void MatrixTanhOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName   << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) &&
            "grad of Tanh output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N =
        new OpNode(opNode->name() + "_grad", new MatrixTanhGradOp());
    N->exlinkUpperNode(input, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
    
    for (int i = 0; i < opNode->parentNum(); i++) {

        auto *tnode = (TensorNode *)(opNode->getParentNode(i));
        auto *tensor = tnode->getTensor();
        auto *N = new TensorNode(tnode->name() + "_grad",
                new Tensor(tensor->getTensorShape()),
                gradNodeMap[opNode]);

        SWLOG_DEBUG(4) << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}


void MaxPoolOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName   << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) &&
            "grad of MaxPool output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N =
        new OpNode(opNode->name() + "_grad", new MaxPoolGradOp());
    N->exlinkUpperNode(input, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
    
    for (int i = 0; i < opNode->parentNum(); i++) {

        auto *tnode = (TensorNode *)(opNode->getParentNode(i));
        auto *tensor = tnode->getTensor();
        auto *N = new TensorNode(tnode->name() + "_grad",
                new Tensor(tensor->getTensorShape()),
                gradNodeMap[opNode]);

        SWLOG_DEBUG(4) << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}


void MatrixSoftmaxOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName   << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *label = opNode->getParentNode(1);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) &&
            "grad of Softmax output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad",
            new MatrixSoftmaxGradOp());
    N->exlinkUpperNode(input, label, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
    
    
    for (int i = 0; i < opNode->parentNum(); i++) {

        auto *tnode = (TensorNode *)(opNode->getParentNode(i));
        auto *tensor = tnode->getTensor();
        auto *N = new TensorNode(tnode->name() + "_grad",
                new Tensor(tensor->getTensorShape()),
                gradNodeMap[opNode]);

        SWLOG_DEBUG(4) << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}



void Conv2dOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName   << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *weight = opNode->getParentNode(1);
    auto *bias = opNode->getParentNode(2);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) &&
            "grad of Conv2d output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad",
            new Conv2dGradOp());
    N->exlinkUpperNode(input, weight, bias, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
    
    for (int i = 0; i < opNode->parentNum(); i++) {

        auto *tnode = (TensorNode *)(opNode->getParentNode(i));
        auto *tensor = tnode->getTensor();
        auto *N = new TensorNode(tnode->name() + "_grad",
                new Tensor(tensor->getTensorShape()),
                gradNodeMap[opNode]);

        SWLOG_DEBUG(4) << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}


