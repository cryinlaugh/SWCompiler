/*************************************************************************
    > File Name: dlOpLowering.cpp
    > Author: wayne
    > Mail:
    > Created Time: 三  7/10 19:12:05 2019
 ************************************************************************/
 #include "dlOp.h"

 #include <cassert>

 #include "SWDSL.h"
 #include "graphIR/IRGraph.h"
 #include "graphIR/IRNode.h"
 #include "graphIR/OpNode.h"
 #include "graphIR/TensorNode.h"

 using namespace swc::op;

void MatrixMatrixFCBiasOp::lowering(IRGraph *graph, IRNode *node) {
	SWLOG_DEBUG(4) << "Lowering MatrixMatrixFCBiasOp ..." << std::endl;
	this->einsumLowering(graph, node);
}

void MatrixMatrixFCBiasGradOp::lowering(IRGraph *graph, IRNode *node) {
	SWLOG_DEBUG(4) << "Lowering MatrixMatrixFCBiasGradOp ..." << std::endl;
	this->einsumLowering(graph, node);
}

void MatrixMatrixFCOp::lowering(IRGraph *graph, IRNode *node) {
	SWLOG_DEBUG(4) << "Lowering MatrixMatrixFCOp ..." << std::endl;
	this->einsumLowering(graph, node);
}

void MatrixMatrixFCGradOp::lowering(IRGraph *graph, IRNode *node) {
	SWLOG_DEBUG(4) << "Lowering MatrixMatrixFCGradOp ..." << std::endl;
	this->einsumLowering(graph, node);
}
