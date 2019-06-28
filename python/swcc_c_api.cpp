#include "swcc_c_api.h"
#include <string.h>
#include <fstream>

swc::IRGraph* IRGraph() {
    return new swc::IRGraph();
}

// swc::TensorNode* TensorNode(const char* name) {
swc::TensorNode* TensorNode(const char* name, int ndim, size_t *dims) {
    std::vector<size_t> *vec = new std::vector<size_t>(ndim);
    vec->assign(dims, dims+ndim);
    return new swc::TensorNode(name, new swc::Tensor(new swc::TensorShape(vec)));
}

swc::OpNode *OpNode(const char* name, const char* op) {
    if (strcmp(op, "Add") == 0) {
        return new swc::OpNode(name, new swc::op::MatrixAddOp());
    } else if (strcmp(op, "FC") == 0) {
        return new swc::OpNode(name, new swc::op::MatrixMatrixFCOp());
    } else if (strcmp(op, "Tanh") == 0) {
        return new swc::OpNode(name, new swc::op::MatrixTanhOp());
    } else if (strcmp(op, "Softmax") == 0) {
        return new swc::OpNode(name, new swc::op::MatrixSoftmaxOp());
    }
    return new swc::OpNode(name, new swc::op::MatrixAddOp());
}

void IRGraph_pushTensorNode(swc::IRGraph *graph, swc::TensorNode *tnode) {
    graph->pushTensorNode(tnode);
}

void IRGraph_pushOpNode(swc::IRGraph *graph, swc::OpNode *onode) {
    graph->pushOpNode(onode);
}

void OpNode_toString(swc::OpNode *node, char *str) {
    swc::op::Op *op_ = node->getOp();
    // std::string tmp = (op->name()+": "+op_->getOpName()
    //     +std::to_string(op->parentNum()));
    // strcpy(str, tmp.c_str());
    std::stringstream oss;
    oss << "OpNode " << node->name() << "\n"
       << "  op: " << op_->getOpName() << "\n"
       << "  nInput : " << op_->getnInput() << "\n"
       << "  nOutput: " << op_->getnOutput();
    strcpy(str, oss.str().c_str());
}

void TensorNode_toString(swc::TensorNode *node, char *str) {
    // std::string tmp = n->name()+": "
    //     +std::to_string(n->parentNum());
    // strcpy(str, tmp.c_str());
    std::stringstream oss;
    swc::Tensor *tensor_ = node->getTensor();
    oss << "TensorNode: " << node->name() << "\n"
       << "  tensorDim: " << tensor_->getNDim() << "\n  ";
    for (int i = 0; i < tensor_->getNDim(); i++)
        oss << tensor_->getDim(i) << " ";
    strcpy(str, oss.str().c_str());
}

void OpNode_link(swc::OpNode* a, swc::TensorNode* b) {
    a->exlinkUpperNode(b);
}
void TensorNode_link(swc::TensorNode* a, swc::OpNode* b) {
    a->exlinkUpperNode(b);
}

void IRGraph_addOpNode(swc::IRGraph *graph, swc::OpNode *o) {
    graph->pushOpNode(o);
}
void IRGraph_addTensorNode(swc::IRGraph *graph, swc::TensorNode *t) {
    graph->pushTensorNode(t);
}

const char *IRGraph_summary(swc::IRGraph* graph) {
    std::string str = "ops="+std::to_string(graph->opNodeNum()) \
        + " tensors="+std::to_string(graph->tensorNodeNum());
    return str.c_str();
}

void IRGraph_dotGen(swc::IRGraph* graph, const char* path) {
    graph->updateTopology();
    swc::pass::Optimizer *opt = new swc::pass::Optimizer(graph);
    opt->runOptimizer();
    swc::dotGen(graph, path);
}