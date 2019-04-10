/*************************************************************************
	> File Name: dlOp.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:35 2018
 ************************************************************************/

#include "dlOp.h"

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

    auto input = (TensorNode *)node->getParentNode(0);
    auto weight = (TensorNode *)node->getParentNode(1);
    auto idims = input->getDims();
    auto wdims = weight->getDims();

    std::string mm_name = node->name() + "_mm";
    auto *mm_op = new MatrixMatrixMulOp();
    auto *O1 = new OpNode(mm_name.c_str(), mm_op);
    std::string mm_out_name = mm_name + "_out";
    auto *O1_out =
        new TensorNode(mm_out_name.c_str(), {(int)idims[0], (int)wdims[1]}, O1);

    std::string madd_name = node->name() + "_add";
    auto *madd_op = new BatchedAddOp();
    auto *O2 = new OpNode(madd_name.c_str(), madd_op);

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
    graph->findInOut();
    graph->updateTopology();
    graph->updateTopoNodeList();
    SWLOG_INFO << "Finish lowering MatrixMatrixFCOp." << std::endl;
}

} // namespace swc
