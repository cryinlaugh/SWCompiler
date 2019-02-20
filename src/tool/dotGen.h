#ifndef DOTGEN_H_
#define DOTGEN_H_

#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/IRGraph.h"

#include "../op/op.h"
#include "../op/basicOp/basicOps.h"
#include "../op/dlOp/dlOp.h"
#include "../op/tensorOp/tensorOps.h"

namespace swc {

template<typename Dtype> class IRGraph;

template<typename Dtype>
void dotGen(IRGraph<Dtype>* graph); 

// convert string to vector<string>
std::vector<std::string> str_split(std::string str, std::string pattern);

// drop " " in input str
void drop_mark(std::string &str, const std::string &mark);

template<typename Dtype>
TensorNode<Dtype>* create_TensorNode( std::vector<std::string>&   tNodeInfo, 
                                      std::vector<unsigned long>* t_dim      ) {

    TensorNode<Dtype>* t_node      = new TensorNode<Dtype>(tNodeInfo[1].c_str());
    TensorShape*       tensorshape = new TensorShape(t_dim);
    Tensor<Dtype>*     tensor      = new Tensor<Dtype>(tensorshape);
    t_node->setTensor(tensor);
    
    return t_node;
}

// There is a mem BUG in "create_OpNode()"
template<typename Dtype>
OpNode<Dtype>* create_OpNode(std::vector<std::string>& opNodeInfo) {

    // define op nodes 
    // std::string op_name = result[1];
    OpNode<Dtype>* op_node = new OpNode<Dtype>(opNodeInfo[1].c_str());
    // MatrixVectorMulOp<Dtype>* op = new MatrixVectorMulOp<Dtype>();
    // op_node->setOp(op);

    // Basic Ops
    if (opNodeInfo[2] == "MatrixMatrixFCOp") {
        MatrixMatrixFCOp<Dtype>* op = new MatrixMatrixFCOp<Dtype>(); 
        op_node->setOp(op);
    }
    if (opNodeInfo[2] == "VectorMatrixMulOp") {
        VectorMatrixMulOp<Dtype>* op = new VectorMatrixMulOp<Dtype>(); 
        op_node->setOp(op);
    }
    if (opNodeInfo[2] == "MatrixVectorMulOp") {
        std::cout << "Create_Op: MatrixVectorMulOp!\n";
        MatrixVectorMulOp<Dtype>* op = new MatrixVectorMulOp<Dtype>();
        op_node->setOp(op);
    }

    // if (result[2] == "VectorVectorInnerProductOp")
    //     VectorVectorInnerProductOp<Dtype>* op = new VectorVectorInnerProductOp<Dtype>();

    // if (result[2] == "ScalarAddOp")
    //     ScalarAddOp<Dtype>* op = new ScalarAddOp<Dtype>();
    // if (result[2] == "ScalarMaxOp")
    //     ScalarMaxOp<Dtype>* op = new ScalarMaxOp<Dtype>();
    // if (result[2] == "ScalarExpOp")
    //     ScalarExpOp<Dtype>* op = new ScalarExpOp<Dtype>();
    // if (result[2] == "ScalarNegOp")
    //     ScalarNegOp<Dtype>* op = new ScalarNegOp<Dtype>();
    // if (result[2] == "ScalarDivOp")
    //     ScalarDivOp<Dtype>* op = new ScalarDivOp<Dtype>();
    // if (result[2] == "ScalarLogOp")
    //     ScalarLogOp<Dtype>* op = new ScalarLogOp<Dtype>();

    // // DL Ops
    // if (result[2] == "MatrixMatrixFCOp")
    //     MatrixMatrixFCOp<Dtype>*   op = new MatrixMatrixFCOp<Dtype>();
    // if (result[2] == "MatrixTanhOp")
    //     MatrixTanhOp<Dtype>*       op = new MatrixTanhOp<Dtype>();
    // if (result[2] == "MatrixSoftmaxOp")
    //     MatrixSoftmaxOp<Dtype>*    op = new MatrixSoftmaxOp<Dtype>();
    // if (result[2] == "MatrixLogNegLossOp")
    //     MatrixLogNegLossOp<Dtype>* op = new MatrixLogNegLossOp<Dtype>();

    // if (result[2] == "VectorTanhOp")
    //     VectorTanhOp<Dtype>*       op = new VectorTanhOp<Dtype>();
    // if (result[2] == "VectorSoftmaxOp")
    //     VectorSoftmaxOp<Dtype>*    op = new VectorSoftmaxOp<Dtype>();
    // if (result[2] == "VectorLogNegLossOp")
    //     VectorLogNegLossOp<Dtype>* op = new VectorLogNegLossOp<Dtype>();

    // if (result[2] == "ScalarTanhOp")
    //     ScalarTanhOp<Dtype>* op = new ScalarTanhOp<Dtype>();

    // // Tensor OPs
    // if (result[2] == "MatrixDuplicateOp")
    //     MatrixDuplicateOp<Dtype>* op = new MatrixDuplicateOp<Dtype>();
    // if (result[2] == "MatrixSplitOp")
    //     MatrixSplitOp<Dtype>*     op = new MatrixSplitOp<Dtype>();
    // if (result[2] == "MatrixConcatOp")
    //     MatrixConcatOp<Dtype>*    op = new MatrixConcatOp<Dtype>();
    // if (result[2] == "MatrixTransposeOp")
    //     MatrixTransposeOp<Dtype>* op = new MatrixTransposeOp<Dtype>();
    // if (result[2] == "MatrixDescendOp")
    //     MatrixDescendOp<Dtype>*   op = new MatrixDescendOp<Dtype>();

    // if (result[2] == "VectorDuplicateOp")
    //     VectorDuplicateOp<Dtype>* op = new VectorDuplicateOp<Dtype>();
    // if (result[2] == "VectorSplitOp")
    //     VectorSplitOp<Dtype>*     op = new VectorSplitOp<Dtype>();
    // if (result[2] == "VectorConcatOp")
    //     VectorConcatOp<Dtype>*    op = new VectorConcatOp<Dtype>();
    // if (result[2] == "VectorAscendOp")
    //     VectorAscendOp<Dtype>*    op = new VectorAscendOp<Dtype>();
    // if (result[2] == "VectorDescendOp")
    //     VectorDescendOp<Dtype>*   op = new VectorDescendOp<Dtype>();

    // if (result[2] == "ScalarDuplicateOp")
    //     ScalarDuplicateOp<Dtype>* op = new ScalarDuplicateOp<Dtype>();
    // if (result[2] == "ScalarAscendOp")
    //     ScalarAscendOp<Dtype>*    op = new ScalarAscendOp<Dtype>();

    // op_node->setOp(op);

    return op_node;
}

template<typename Dtype>
void Link_Upper(std::vector<std::string>& linkInfo, std::vector<IRNode*>& IRNodeBuff) {

    int self;
    std::vector<int> upper;

    // get the info of "self"
    for (int i = 0; i < (int)IRNodeBuff.size(); ++i) {
        if (linkInfo[1] == IRNodeBuff[i]->name()) 
            self = i;
    }

    // get the info of "upper[]"
    for (int i = 2; i < (int)linkInfo.size(); ++i) {
        for (int j = 0; j < (int)IRNodeBuff.size(); ++j) {
            if (linkInfo[i] == IRNodeBuff[j]->name())
                upper.push_back(j);
        }
    }

    for (int j = 0; j < (int)upper.size(); ++j) {
        IRNodeBuff[self]->exlinkUpperNode(IRNodeBuff[upper[j]]);
    }   
}

template<typename Dtype>
// void AddIRnodeToBuff(std::vector<IRNode*>& IRNodeBuff, std::string Input_str) {
void HCI(std::vector<IRNode*>& IRNodeBuff, std::string Input_str) {

    drop_mark(Input_str, " ");     // drop " "

    std::vector<std::string> InputInfo = str_split(Input_str, ",");

    if (InputInfo[0] == "TENSOR") {

        // convert InputInfo<string>[1, ...] to t_dim<int>[]
        std::vector<unsigned long> *t_dim = new std::vector<unsigned long>();
        for (int i = 2; i < (int)InputInfo.size(); ++i) {
            unsigned long tmp = atoi(InputInfo[i].c_str()); // 666 + i;// stoi(result[i]);
            // convertFromString(tmp, result);
            t_dim->push_back(tmp);
        }

        IRNodeBuff.push_back(create_TensorNode<Dtype>(InputInfo, t_dim));

    } else if (InputInfo[0] == "OP") {
        
        IRNodeBuff.push_back(create_OpNode<Dtype>(InputInfo));

    } else if (InputInfo[0] == "LINKUPPER") {

        Link_Upper<Dtype>(InputInfo, IRNodeBuff);
        
    } else {

        std::cout << "Input format error!" << std::endl;
    }
}


} // namespace swc

#endif /* !DOTGEN_H_ */
