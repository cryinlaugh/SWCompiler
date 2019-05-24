/*************************************************************************
    > File Name: AutoDiff.cpp
    > Author: wayne
    > Mail:
    > Created Time: 三  4/10 17:26:14 2019
 ************************************************************************/

#include "common.h"

#include "graphIR/IRGraph.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"

#include "op/dlOp/dlOp.h"
#include "tensor/tensor.h"

#include "AutoDiff.h"

#include <cassert>
#include <unordered_map>
#include <vector>

using namespace swc::op;

namespace swc {

IRGraph *getTrainNet(IRGraph *graph, TrainingConfig &config) {
    SWLOG_DEBUG(4) << "start get trainingNet\n";
    // , TrainingConfig config){
    IRGraph *net = graph->clone();

    net->updateTopology();
    std::unordered_map<IRNode *, IRNode *> gradNodeMap;

    std::vector<IRNode *> topo_nodes;
    for (int i = 0; i < net->topologyNum(); i++)
        for (int j = 0; j < net->getNumInTopoLevel(i); j++) {
            auto node = net->getNodeInTopo(i, j);
            topo_nodes.push_back(node);
        }

    for (auto it = topo_nodes.rbegin(), e = topo_nodes.rend(); it != e; it++) {
        IRNode *irnode = *it;
        if (irnode->nodeType() == TENSOR_NODE) {
            auto *node = (TensorNode *)irnode;
            Label *label = node->getLabel();

            SWLOG_DEBUG(2) << "reverse order tenosr " << node->name()
                           << " label training flag " << label->needTraining()
                           << "\n";

            if (gradNodeMap.count(node)) {
                auto *N = gradNodeMap[node];

                // label::needTraining not set yet
                // if (!label->needTraining())
                if (!node->getTraining())
                    continue;

                auto *node_mirror = node->clone();

                auto *mom_t = new Tensor(node->getTensor()->getTensorShape());
                mom_t->setTensorInit(TensorInitType::CONSTANT, 0);
                auto *momentum = new TensorNode("momentum", mom_t);

                auto *sgdOp = new SGDOp(config.lr, config.decay,
                                        config.momentum, config.batch);
                auto *SGDNode = new OpNode(node->name() + "_sgd", sgdOp);

                SGDNode->exlinkUpperNode(node, N, momentum);
                node_mirror->exlinkUpperNode(SGDNode);

                net->pushOpNode(SGDNode);
                net->pushTensorNode(node_mirror, momentum);

                continue;
            }

            if (node->childNum() == 0) {
                SWLOG_DEBUG(1) << node->name() << " childNum = 0\n";

                /*
                auto *opNode = new OpNode(node->name()+"_negloss", new
                MatrixLogNegLossOp()); opNode->exlinkUpperNode(node); SWLOG_INFO
                << node->name() << " childNum = " << node->childNum() << "\n";
                net->pushOpNode(opNode);
                */

                auto *tensor = ((TensorNode *)node)->getTensor();
                auto *N = new TensorNode(node->name() + "_grad",
                                         new Tensor(tensor->getTensorShape()));
                // tensor->setTensorInit(TensorInitType::CONSTANT, 0);

                gradNodeMap[node] = N;
                net->pushTensorNode(N);
                continue;
            }

        } else if (irnode->nodeType() == OP_NODE) {
            auto *node = (OpNode *)irnode;
            std::cout << "reverse order op   " << node->name() << std::endl;
            if (auto op = dynamic_cast<MatrixMatrixFCOp *>(node->getOp())) {
                auto *input = node->getParentNode(0);
                auto *weight = node->getParentNode(1);
                auto *bias = node->getParentNode(2);
                auto *output = node->getChildNode(0);
                assert(gradNodeMap.count(output) &&
                       "grad of FC output unfound\n");
                auto *outputGrad = gradNodeMap[output];

                auto *N = new OpNode(node->name() + "_grad",
                                     new MatrixMatrixFCGradOp());
                N->exlinkUpperNode(input, weight, bias, output, outputGrad);

                gradNodeMap[node] = N;
                net->pushOpNode(N);
            } else if (auto op = dynamic_cast<MatrixTanhOp *>(node->getOp())) {
                SWLOG_DEBUG(2)
                    << "get Gradient node for op " << node->name() << "\n";
                auto *input = node->getParentNode(0);
                auto *output = node->getChildNode(0);
                assert(gradNodeMap.count(output) &&
                       "grad of Tanh output unfound\n");
                auto *outputGrad = gradNodeMap[output];

                auto *N =
                    new OpNode(node->name() + "_grad", new MatrixTanhGradOp());
                N->exlinkUpperNode(input, output, outputGrad);

                gradNodeMap[node] = N;
                net->pushOpNode(N);
            } else if (auto op =
                           dynamic_cast<MatrixSoftmaxOp *>(node->getOp())) {
                SWLOG_DEBUG(2)
                    << "get Gradient node for op " << node->name() << "\n";
                auto *input = node->getParentNode(0);
                auto *label = node->getParentNode(1);
                auto *output = node->getChildNode(0);
                assert(gradNodeMap.count(output) &&
                       "grad of Softmax output unfound\n");
                auto *outputGrad = gradNodeMap[output];

                auto *N = new OpNode(node->name() + "_grad",
                                     new MatrixSoftmaxGradOp());
                N->exlinkUpperNode(input, label, output, outputGrad);

                gradNodeMap[node] = N;
                net->pushOpNode(N);
            }
            for (int i = 0; i < node->parentNum(); i++) {

                auto *tnode = (TensorNode *)(node->getParentNode(i));
                auto *tensor = tnode->getTensor();
                auto *N = new TensorNode(tnode->name() + "_grad",
                                         new Tensor(tensor->getTensorShape()),
                                         gradNodeMap[node]);

                SWLOG_DEBUG(2) << "get Gradient node for " << node->name()
                               << " input " << tnode->name() << "\n";

                gradNodeMap[tnode] = N;
                net->pushTensorNode(N);
            }
        }
    }

    return net;
}

} // namespace swc
