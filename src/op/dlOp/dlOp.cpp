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

void MatrixMatrixFCOp::lowering(IRGraph *graph, IRNode *node) {
    SWLOG_DEBUG(4) << "Lowering MatrixMatrixFCOp ..." << std::endl;

    // define lowered subgraph
    // substitute MatrixMatrixFCOp with MatrixMatrixMulOp

    // define MatrixMatrixMulOp
    // OP(O1, MatrixMatrixMulOp);
    assert(node->parentNum() == 3 &&
           "FC input should be 3: data, weight, bias");
    assert(node->childNum() == 1 && "FC input should be 1");

    Device dev = node->getLabel()->getDeviceLabel();

    auto input = (TensorNode *)node->getParentNode(0);
    auto weight = (TensorNode *)node->getParentNode(1);
    auto idims = input->getDims();
    auto wdims = weight->getDims();

    std::string mm_name = node->name() + "_mm";
    auto *mm_op = new MatrixMatrixMulOp();
    auto *O1 = new OpNode(mm_name, mm_op);
    std::string mm_out_name = mm_name + "_out";
    auto *O1_out = new TensorNode(mm_out_name, {idims[0], wdims[1]}, O1);

    O1->getLabel()->setDeviceLabel(dev.type, dev.id);
    O1_out->getLabel()->setDeviceLabel(dev.type, dev.id);

    std::string madd_name = node->name() + "_add";
    auto *madd_op = new BatchedAddOp();
    auto *O2 = new OpNode(madd_name, madd_op);

    O2->getLabel()->setDeviceLabel(dev.type, dev.id);

    // link parent nodes
    LINKUPPER(O1, node->getParentNode(0), node->getParentNode(1));
    // O1_T already linked to O1
    LINKUPPER(O2, O1_out, node->getParentNode(2));

    // link children nodes
    LINKUPPER(node->getChildNode(0), O2);

    graph->pushOpNode(O1, O2);
    graph->pushTensorNode(O1_out);

    // break parent links
    DESTROYUPPER(node, node->getParentNode(0), node->getParentNode(1),
                 node->getParentNode(2));
    // break child links
    DESTROYUPPER(node->getChildNode(0), node);

    // remove node from graph
    graph->delOpNode(node);

    // delete node
    // TODO
    node->destroy();

    // Update graph info
    graph->updateTopology();
}

void MatrixMatrixFCBiasGradOp::lowering(IRGraph *graph, IRNode *node) {
    SWLOG_DEBUG(4) << "Lowering MatrixMatrixFCGradOp ..." << std::endl;
    for (int i = 0; i < node->parentNum(); i++) {
        std::cout << node->getParentNode(i)->name() << std::endl;
    }

    for (int i = 0; i < node->childNum(); i++) {
        std::cout << node->getChildNode(i)->name() << std::endl;
    }
    auto *input = (TensorNode *)node->getParentNode(0);
    auto *weight = (TensorNode *)node->getParentNode(1);
    auto *bias = (TensorNode *)node->getParentNode(2);
    auto *output = (TensorNode *)node->getParentNode(3);
    auto *outputG = (TensorNode *)node->getParentNode(4);

    auto *inputG = (TensorNode *)node->getChildNode(0);
    auto *weightG = (TensorNode *)node->getChildNode(1);
    auto *biasG = (TensorNode *)node->getChildNode(2);

    auto idims = input->getDims();
    auto wdims = weight->getDims();

    // Y = XW + B e.g. 8*10 8*512 512*10 10
    // dx = dy*WT
    // may be we tanspose W again, we can optimize tanspose-transpose
    auto op_w_t =
        new OpNode("op_" + weight->name() + "_T", new TransposeOp({1, 0}));
    op_w_t->exlinkUpperNode(weight);
    Tensor *wt =
        new Tensor(weight->getTensor()->getShuffledTensorShape({1, 0}));
    auto w_trans = new TensorNode(weight->name() + "_T", wt, op_w_t);

    auto dx = new OpNode(node->name() + "_dx_mm", new MatrixMatrixMulOp());
    dx->exlinkUpperNode(outputG, w_trans);
    std::cout << "FCGrad inputG " << inputG->name() << " link to " << dx->name()
              << std::endl;
    inputG->exlinkUpperNode(dx);

    // dw = XT*dy
    auto op_x_t =
        new OpNode("op_" + input->name() + "_T", new TransposeOp({1, 0}));
    op_x_t->exlinkUpperNode(input);
    Tensor *xt = new Tensor(input->getTensor()->getShuffledTensorShape({1, 0}));
    auto x_trans = new TensorNode(input->name() + "_T", xt, op_x_t);

    auto dw = new OpNode(node->name() + "_dw_mm", new MatrixMatrixMulOp());
    dw->exlinkUpperNode(x_trans, outputG);
    weightG->exlinkUpperNode(dw);

    // dB = reduceadd dy
    auto db = new OpNode(node->name() + "_db_bra", new BatchedReduceAddOp());
    db->exlinkUpperNode(outputG);
    biasG->exlinkUpperNode(db);

    node->destroyUpperNode(input, weight, bias, output, outputG);
    inputG->destroyUpperNode(node);
    weightG->destroyUpperNode(node);
    biasG->destroyUpperNode(node);

    graph->delOpNode(node);
    graph->pushOpNode(op_w_t, op_x_t, dx, dw, db);
    graph->pushTensorNode(w_trans, x_trans);

    graph->updateTopology();
}


void MatrixMatrixFCGradOp::lowering(IRGraph *graph, IRNode *node) 
{

}
/*-----------------------------------Auto Diff ----------------------------
 * 
 * 
 * 
 * -------------------------------------------------------------------------*/
void MatrixMatrixFCBiasOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_INFO << "autoDiff: " << _opClassName   << std::endl;
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

        SWLOG_INFO << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}
void MatrixMatrixFCOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_INFO << "autoDiff: " << _opClassName   << std::endl;
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

        SWLOG_INFO << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}


void ReluOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_INFO << "autoDiff: " << _opClassName   << std::endl;
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

        SWLOG_INFO << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}

void MatrixTanhOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_INFO << "autoDiff: " << _opClassName   << std::endl;
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

        SWLOG_INFO << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}


void MaxPoolOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_INFO << "autoDiff: " << _opClassName   << std::endl;
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

        SWLOG_INFO << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}


void MatrixSoftmaxOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_INFO << "autoDiff: " << _opClassName   << std::endl;
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

        SWLOG_INFO << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}



void Conv2dOp::autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap)
{
    SWLOG_INFO << "autoDiff: " << _opClassName   << std::endl;
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

        SWLOG_INFO << "get Gradient node for " << opNode->name()
            << " input " << tnode->name() << "\n";

        gradNodeMap[tnode] = N;
        graph->pushTensorNode(N);
    }
}


