/*
 * dlOpEinsum.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-07-08
 */


#include "dlOp.h"

#include <cassert>

#include "SWDSL.h"
#include "graphIR/IRGraph.h"
#include "graphIR/IRNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "op/tensorOp/tensorOps.h"

#define COP(token, name, method, parent...)             \
    OpNode* token = new OpNode(name, new method());     \
    LINKUPPER(token, parent)

#define CTENSOR(token, name, shape, parent)     \
    TensorNode* token = new TensorNode(name, new Tensor(shape), parent)


namespace swc {
namespace op {

/************************************************************
 * This method LOWER is to lower dlop into basic op
 * *********************************************************/

void MatrixMatrixFCOp::einsumLowering(IRGraph *graph, IRNode *node) 
{
    SWLOG_DEBUG(4) << "einsumLowering MatrixMatrixFCOp ..." << std::endl;

    // Op check;
    assert(node->parentNum() == 2 &&
           "FC input should be 2: data, weight");
    assert(node->childNum() == 1 && "FC output should be 1");

    // Op info fetch
    Device dev = node->getLabel()->getDeviceLabel();
    
    auto input = (TensorNode *)node->getParentNode(0);
    auto weight = (TensorNode *)node->getParentNode(1);
    auto output = (TensorNode *)node->getChildNode(0);

    // define lowered sugraph
    // define MatrixMatrixMulOp Node  
    //auto *O1 = new OpNode(node->name() + "_mm", new MatrixMatrixMulOp());

    COP(O1, node->name() + "_mm", MatrixMatrixMulOp,
            input, weight);
    
    LINKUPPER(output, O1);

    O1->getLabel()->setDeviceLabel(dev.type, dev.id);

    // break parent links
    DESTROYUPPER(node, input, weight);
    // break child links
    DESTROYUPPER(output, node);

    // remove node from graph
    GdO(graph, node);
    // add node to graph
    GpO(graph, O1);

    // delete node
    // TODO
    node->destroy();

    // Update graph info
    graph->updateTopology();
}

void MatrixMatrixFCGradOp::einsumLowering(IRGraph *graph, IRNode *node) 
{

    SWLOG_DEBUG(4) << "einsumLowering MatrixMatrixFCGradOp ..." << std::endl;
    
    // Op check;
    assert(node->parentNum() == 4 &&
           "FCGrad input should be 4: data, weight, output, outputGrad");
    assert(node->childNum() == 2 && "FCGrad output should be 2: dataGrad, weightGrad");

    for (int i = 0; i < node->parentNum(); i++) {
        std::cout << node->getParentNode(i)->name() << std::endl;
    }

    for (int i = 0; i < node->childNum(); i++) {
        std::cout << node->getChildNode(i)->name() << std::endl;
    }
    
    // Op info fetch
    auto *input = (TensorNode *)node->getParentNode(0);
    auto *weight = (TensorNode *)node->getParentNode(1);
    auto *output = (TensorNode *)node->getParentNode(2);
    auto *outputG = (TensorNode *)node->getParentNode(3);

    auto *inputG = (TensorNode *)node->getChildNode(0);
    auto *weightG = (TensorNode *)node->getChildNode(1);

    // define lowered sugraph  
    // Y = XW + B e.g. 8*10 8*512 512*10 10
    // Calculate dx first
    // dx = dy*WT
    COP(op_w_t, "op_" + weight->name() + "_T", MatrixTransposeOp, weight);
    
    CTENSOR(w_trans, weight->name() + "_T",
            weight->getTensor()->getShuffledTensorShape({1, 0}), 
            op_w_t);

    COP(dx, node->name() + "_dx_mm", MatrixMatrixMulOp,
            outputG, w_trans);
    
    LINKUPPER(inputG, dx);
    std::cout << "FCGrad inputG " << inputG->name() << " link to " << dx->name()
              << std::endl;

    // dw = XT*dy
    COP(op_x_t, "op_" + input->name() + "_T", MatrixTransposeOp, input);
   
    CTENSOR(x_trans, input->name() + "_T",
            input->getTensor()->getShuffledTensorShape({1, 0}),
            op_x_t);

    COP(dw, node->name() + "_dw_mm", MatrixMatrixMulOp, 
            x_trans, outputG);
   
    LINKUPPER(weightG, dw);

    DESTROYUPPER(node, input, weight, output, outputG);
    DESTROYUPPER(inputG, node);
    DESTROYUPPER(weightG, node);

    GdO(graph, node);
    GpO(graph, op_w_t, op_x_t, dx, dw);
    GpT(graph, w_trans, x_trans);

    node->destroy();

    graph->updateTopology();
}

void MatrixMatrixFCBiasOp::einsumLowering(IRGraph *graph, IRNode *node)
{
    SWLOG_DEBUG(4) << "einsumLowering MatrixMatrixFCBiasOp ..." << std::endl;
    
    // Op check;
    assert(node->parentNum() == 3 &&
           "FC input should be 3: data, weight, bias");
    assert(node->childNum() == 1 && "FC input should be 1");

    // Op info fetch
    Device dev = node->getLabel()->getDeviceLabel();
    
    auto input = (TensorNode *)node->getParentNode(0);
    auto weight = (TensorNode *)node->getParentNode(1);
    auto bias = (TensorNode *)node->getParentNode(2);
    auto idims = input->getDims();
    auto wdims = weight->getDims();

    // define lowered sugraph
    // define MatrixMatrixMulOp Node  
    std::string mm_name = node->name() + "_mm";
    auto *mm_op = new MatrixMatrixMulOp();
    auto *O1 = new OpNode(mm_name, mm_op);
    
    // define intermediate result tensor
    std::string mm_out_name = mm_name + "_out";
    auto *O1_out = new TensorNode(mm_out_name, {idims[0], wdims[1]}, O1);

    O1->getLabel()->setDeviceLabel(dev.type, dev.id);
    O1_out->getLabel()->setDeviceLabel(dev.type, dev.id);

    // define Add Op 
    std::string madd_name = node->name() + "_add";
    auto *madd_op = new MatrixVectorAddOp();
    auto *O2 = new OpNode(madd_name, madd_op);

    O2->getLabel()->setDeviceLabel(dev.type, dev.id);

    // link parent nodes
    LINKUPPER(O1, input, weight);
    LINKUPPER(O2, O1_out, bias);
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

void MatrixMatrixFCBiasGradOp::einsumLowering(IRGraph *graph, IRNode *node)
{
    SWLOG_DEBUG(4) << "einsumLowering MatrixMatrixFCBiasGradOp ..." << std::endl;
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
        new OpNode("op_" + weight->name() + "_T", new MatrixTransposeOp());
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
        new OpNode("op_" + input->name() + "_T", new MatrixTransposeOp());
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

} // namespace op
} // namespace swc
