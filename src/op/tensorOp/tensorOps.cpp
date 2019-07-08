/*
 * tensorOps.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-07-08
 */


#include "tensorOps.h"

#include <cassert>

#include "SWDSL.h"
#include "graphIR/IRGraph.h"
#include "graphIR/IRNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"



using namespace swc::op;

void TensorDescendOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName   << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) &&
            "grad of Descend output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *newOp = new OpNode(opNode->name() + "_grad",
            new TensorAscendOp(_nDim, _start, _end));

    newOp->exlinkUpperNode(outputGrad);

    gradNodeMap[opNode] = newOp;
    graph->pushOpNode(newOp);

    auto *tensor = ((TensorNode*)input)->getTensor();
    auto *N = new TensorNode(input->name() + "_grad",
            new Tensor(tensor->getTensorShape()),
            gradNodeMap[opNode]);

    SWLOG_DEBUG(4) << "get Gradient node for " << opNode->name()
        << " input " << input->name() << "\n";

    gradNodeMap[input] = N;
    graph->pushTensorNode(N);
}


void TensorAscendOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName   << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) &&
            "grad of Ascend output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *newOp = new OpNode(opNode->name() + "_grad",
            new TensorDescendOp(_nDim, _start, _end));

    newOp->exlinkUpperNode(outputGrad);

    gradNodeMap[opNode] = newOp;
    graph->pushOpNode(newOp);

    auto *tensor = ((TensorNode*)input)->getTensor();
    auto *N = new TensorNode(input->name() + "_grad",
            new Tensor(tensor->getTensorShape()),
            gradNodeMap[opNode]);

    SWLOG_DEBUG(4) << "get Gradient node for " << opNode->name()
        << " input " << input->name() << "\n";

    gradNodeMap[input] = N;
    graph->pushTensorNode(N);
}
