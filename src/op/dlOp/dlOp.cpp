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

namespace swc {

void MatrixMatrixFCOp::lowering(IRGraph *graph, IRNode *node) {
    SWLOG_INFO << "Lowering MatrixMatrixFCOp ..." << std::endl;

    // define lowered subgraph
    // substitute MatrixMatrixFCOp with MatrixMatrixMulOp

    // define MatrixMatrixMulOp
    // OP(O1, MatrixMatrixMulOp);
    assert(node->parentNum() == 3 &&
           "FC input should be 3: data, weight, bias");
    assert(node->childNum() == 1 &&
           "FC input should be 1");

    Device dev = node->getLabel()->getDeviceLabel();

    auto input = (TensorNode *)node->getParentNode(0);
    auto weight = (TensorNode *)node->getParentNode(1);
    auto idims = input->getDims();
    auto wdims = weight->getDims();

    std::string mm_name = node->name() + "_mm";
    auto *mm_op = new MatrixMatrixMulOp();
    auto *O1 = new OpNode(mm_name, mm_op);
    std::string mm_out_name = mm_name + "_out";
    auto *O1_out =
        new TensorNode(mm_out_name, {(int)idims[0], (int)wdims[1]}, O1);

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
    SWLOG_INFO << "Finish lowering MatrixMatrixFCOp." << std::endl;
}
void MatrixMatrixFCGradOp::lowering(IRGraph *graph, IRNode *node) {
    SWLOG_INFO << "Lowering MatrixMatrixFCGradOp ..." << std::endl;
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
    SWLOG_INFO << "Finish lowering MatrixMatrixFCGradOp." << std::endl;
}

} // namespace swc
