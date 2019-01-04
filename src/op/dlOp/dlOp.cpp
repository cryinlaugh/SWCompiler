/*************************************************************************
	> File Name: dlOp.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:35 2018
 ************************************************************************/

#include "dlOp.h"

#include "SWDSL.h"
#include "graphIR/IRNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/IRGraph.h"

namespace swc {

template <typename Dtype>
void MatrixMatrixFCOp<Dtype>::lowering(IRGraph<Dtype>* graph, IRNode* node){
    SWLOG_INFO<< "Lowering MatrixMatrixFCOp ..." << std::endl;

    //define lowered subgraph
    //substitute MatrixMatrixFCOp with MatrixMatrixMulOp

    //define MatrixMatrixMulOp 
    OP(O1, MatrixMatrixMulOp);

    //link parent nodes
    LINKUPPER(O1, node->getParentNode(0), node->getParentNode(1));

    //link children nodes
    LINKUPPER(node->getChildNode(0), O1);

    graph->pushOpNode(O1);

    //break parent links
    DESTROYUPPER(node, node->getParentNode(0), node->getParentNode(1));

    //break child links
    DESTROYUPPER(node->getChildNode(0), node);

    //remove node from graph
    graph->delOpNode(node);

    //delete node
    //TODO
    node->destroy();

    //Update graph info
    graph->findInOut();
    graph->updateTopology();
    graph->updateTopoNodeList();

    SWLOG_INFO<< "Finish lowering MatrixMatrixFCOp." << std::endl;
}



INSTANTIATE_CLASS(MatrixMatrixFCOp);
INSTANTIATE_CLASS(MatrixTanhOp);
INSTANTIATE_CLASS(MatrixSoftmaxOp);
INSTANTIATE_CLASS(MatrixLogNegLossOp);


INSTANTIATE_CLASS(VectorTanhOp);
INSTANTIATE_CLASS(VectorSoftmaxOp);
INSTANTIATE_CLASS(VectorLogNegLossOp);


INSTANTIATE_CLASS(ScalarTanhOp);

} //namespace swc
