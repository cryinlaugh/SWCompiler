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


    //define tensor nodes
    TensorNode<Dtype>* dataTensorNode_0 = new TensorNode<Dtype>("Data_0");
    //Init tensor nodes as following:
    //--init TensorShape:
    TensorShape* dataTensorShape_0 = new TensorShape(
            new vector<unsigned long>({1000,1000}));
    //--init Tensor
    Tensor<Dtype>* dataTensor_0 = new Tensor<Dtype>(dataTensorShape_0);
    //--set tensor in tensor node
    dataTensorNode_0->setTensor(dataTensor_0);
    
    TensorNode<Dtype>* weightTensorNode_0= new TensorNode<Dtype>("Weight_0");
    TensorShape* weightTensorShape_0 = new TensorShape(
            new vector<unsigned long>({1000,1000}));
    Tensor<Dtype>* weightTensor_0 = new Tensor<Dtype>(weightTensorShape_0);
    weightTensorNode_0->setTensor(weightTensor_0); 
    
    //define op nodes 
    OpNode<Dtype>* fcOpNode_0 = new OpNode<Dtype>("FC_0");
    //Init op nodes as following:
    //--init Op:
    MatrixMatrixFCOp<Dtype>* fcOp_0 = new MatrixMatrixFCOp<Dtype>();
    //--set Op in Op node 
    fcOpNode_0->setOp(fcOp_0);

    //link upperNode from current node(Parent)
    //Relink upperNode to current node(Child)
    fcOpNode_0->pushParentNode(dataTensorNode_0, weightTensorNode_0);
    dataTensorNode_0->pushChildNode(fcOpNode_0);
    
    TensorNode<Dtype>* dataTensorNode_1= new TensorNode<Dtype>("Data_1");
    TensorShape* dataTensorShape_1 = new TensorShape(
            new vector<unsigned long>({1000,1000}));
    Tensor<Dtype>* dataTensor_1 = new Tensor<Dtype>(dataTensorShape_1);
    dataTensorNode_1->setTensor(dataTensor_1); 
    
    dataTensorNode_1->pushParentNode(fcOpNode_0);
    fcOpNode_0->pushChildNode(dataTensorNode_1);
    
    OpNode<Dtype>* tanhOpNode_1 = new OpNode<Dtype>("Tanh_1");
    MatrixTanhOp<Dtype>* tanhOp_1 = new MatrixTanhOp<Dtype>();
    tanhOpNode_1->setOp(tanhOp_1);

    tanhOpNode_1->pushParentNode(dataTensorNode_1);
    dataTensorNode_1->pushChildNode(tanhOpNode_1);

    TensorNode<Dtype>* dataTensorNode_2= new TensorNode<Dtype>("Data_2");
    TensorShape* dataTensorShape_2 = new TensorShape(
            new vector<unsigned long>({1000,1000}));
    Tensor<Dtype>* dataTensor_2 = new Tensor<Dtype>(dataTensorShape_2);
    dataTensorNode_2->setTensor(dataTensor_2); 

    dataTensorNode_2->pushParentNode(tanhOpNode_1);
    tanhOpNode_1->pushChildNode(dataTensorNode_2);

    //define IR graph
    IRGraph<Dtype>* MLPLayer = new IRGraph<Dtype>();
    MLPLayer->pushTensorNode(dataTensorNode_0,
                            weightTensorNode_0,
                            dataTensorNode_1,
                            dataTensorNode_2);
    MLPLayer->pushOpNode(fcOpNode_0,
                        tanhOpNode_1);

    printf ("Generate MLP layer done!\n");

    dotGen(*MLPLayer);

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
