/*************************************************************************
	> File Name: testMLP.cpp
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Wed 05 Dec 2018 03:34:34 AM UTC
 ************************************************************************/

#include<iostream>
#include "SWC.h"

using namespace swc;
using namespace std;

#define Dtype float

int main(){
    cout<<"In test MLP main"<<endl;
    //============================
    // Example of 1 FC layer:
    //  T:data_0   T:weight_0
    //     \       /
    //      \     /
    //        O:FC_0
    //         |
    //      T:data_1    
    //         |
    //      O:Tanh_1
    //         |
    //      T:data_2
    //=============================


    //define IR nodes
    TensorNode<Dtype>* dataTensorNode_0 = new TensorNode<Dtype>();
    TensorNode<Dtype>* weightTensorNode_0= new TensorNode<Dtype>();
    TensorNode<Dtype>* dataTensorNode_1= new TensorNode<Dtype>();
    TensorNode<Dtype>* dataTensorNode_2= new TensorNode<Dtype>();
    OpNode<Dtype>* fcOpNode_0 = new OpNode<Dtype>();
    OpNode<Dtype>* tanhOpNode_1 = new OpNode<Dtype>();

    //Init IR nodes with tensors/Ops 
    //
    //==============================
    //Init tensor nodes as following:
    //--init TensorShape:
    TensorShape* dataTensorShape_0 = new TensorShape(
            new vector<unsigned long>({1000,1000}));
    //--init Tensor
    Tensor<Dtype>* dataTensor_0 = new Tensor<Dtype>(dataTensorShape_0);
    //--set tensor in tensor node
    dataTensorNode_0->setTensor(dataTensor_0);
    //=============================
    TensorShape* weightTensorShape_0 = new TensorShape(
            new vector<unsigned long>({1000,1000}));
    Tensor<Dtype>* weightTensor_0 = new Tensor<Dtype>(weightTensorShape_0);
    weightTensorNode_0->setTensor(weightTensor_0); 

    TensorShape* dataTensorShape_1 = new TensorShape(
            new vector<unsigned long>({1000,1000}));
    Tensor<Dtype>* dataTensor_1 = new Tensor<Dtype>(dataTensorShape_1);
    dataTensorNode_1->setTensor(dataTensor_1); 

    TensorShape* dataTensorShape_2 = new TensorShape(
            new vector<unsigned long>({1000,1000}));
    Tensor<Dtype>* dataTensor_2 = new Tensor<Dtype>(dataTensorShape_2);
    dataTensorNode_2->setTensor(dataTensor_2); 
    //
    //=============================
    //Init op nodes as following:
    //--init Op:
    MatrixMatrixFCOp<Dtype>* fcOp_0 = new MatrixMatrixFCOp<Dtype>();
    //--set Op in Op node 
    fcOpNode_0->setOp(fcOp_0);
    //=============================
    //
    MatrixTanhOp<Dtype>* tanhOp_1 = new MatrixTanhOp<Dtype>();
    tanhOpNode_1->setOp(tanhOp_1);
    
    //Init father/child nodes and name for each IR node 
    vector<IRNode*>* fatherNodes = new vector<IRNode*>();
    vector<IRNode*>* childNodes = new vector<IRNode*>();
    childNodes->push_back(fcOpNode_0);
    dataTensorNode_0->init(fatherNodes, childNodes, string("Data_0"));

    fatherNodes = new vector<IRNode*>();
    childNodes = new vector<IRNode*>();
    childNodes->push_back(fcOpNode_0);
    weightTensorNode_0->init(fatherNodes, childNodes, string("Weight_0"));

    fatherNodes = new vector<IRNode*>();
    childNodes = new vector<IRNode*>();
    fatherNodes->push_back(dataTensorNode_0);
    fatherNodes->push_back(weightTensorNode_0);
    childNodes->push_back(dataTensorNode_1);
    fcOpNode_0->init(fatherNodes, childNodes, string("FC_0"));

    fatherNodes = new vector<IRNode*>();
    childNodes = new vector<IRNode*>();
    fatherNodes->push_back(fcOpNode_0);
    childNodes->push_back(tanhOpNode_1);
    dataTensorNode_1->init(fatherNodes, childNodes, string("Data_1"));

    fatherNodes = new vector<IRNode*>();
    childNodes = new vector<IRNode*>();
    fatherNodes->push_back(dataTensorNode_1);
    childNodes->push_back(dataTensorNode_2);
    tanhOpNode_1->init(fatherNodes,childNodes, string("Tanh_1"));

    fatherNodes = new vector<IRNode*>();
    childNodes = new vector<IRNode*>();
    fatherNodes->push_back(tanhOpNode_1);
    dataTensorNode_2->init(fatherNodes,childNodes, string("Data_2"));


    //define IR graph
    IRGraph<Dtype>* MLPLayer = new IRGraph<Dtype>();
    MLPLayer->pushTensorNode(dataTensorNode_0);
    MLPLayer->pushTensorNode(weightTensorNode_0);
    MLPLayer->pushTensorNode(dataTensorNode_1);
    MLPLayer->pushTensorNode(dataTensorNode_2);
    MLPLayer->pushOpNode(fcOpNode_0);
    MLPLayer->pushOpNode(tanhOpNode_1);

    printf ("Generate MLP layer done!\n");

    for (int i = 0; i < MLPLayer->ternsorNodeNum(); i++) {
        printf("ID:%d, ", i);
        printf("Name:%s, ", MLPLayer->getTensorNode(i)->name().c_str());
        printf("in:%d, ", MLPLayer->getTensorNode(i)->parentNum());
        printf("out:%d\n", MLPLayer->getTensorNode(i)->childNum());
    }

    for (int i = 0; i < MLPLayer->opNodeNum(); i++) {
        printf("ID:%d, ", i);
        printf("Name:%s, ", MLPLayer->getOpNode(i)->name().c_str());
        printf("in:%d, ", MLPLayer->getOpNode(i)->parentNum());
        printf("out:%d\n", MLPLayer->getOpNode(i)->childNum());
    }
    return 0;
}
