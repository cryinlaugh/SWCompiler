/*************************************************************************
	> File Name: testMLPTrain.cpp
	> Author: wayne
	> Mail: singleon11@gmail.com 
	> Created Time: 五  1/25 17:47:52 2019
 ************************************************************************/

#include<iostream>
#include "SWC.h"

using namespace swc;
using namespace std;

#define Dtype float

int main(){
    cout<<"In test MLP main"<<endl;

    TENSOR(Data_0, 8 , 784);
	TENSOR(Weight_0, 784, 512);
	Weight_0_Tensor->setTensorInit(TensorInitType::XAVIER, 784);
	Data_0_Tensor->setTensorInit(TensorInitType::FILE, "mnist_images_8.bin");

	OP(FC_0, MatrixMatrixFCOp);
	LINKUPPER(FC_0, Data_0, Weight_0);
	TENSOR(Data_FC_0, 8 , 512);
	LINKUPPER(Data_FC_0, FC_0);

	OP(Tanh_0, MatrixTanhOp);
	LINKUPPER(Tanh_0, Data_FC_0);
	TENSOR(Data_Tanh_0, 8 , 512);
	LINKUPPER(Data_Tanh_0, Tanh_0);

	TENSOR(Weight_FC_1, 512, 10);
	Weight_FC_1_Tensor->setTensorInit(TensorInitType::XAVIER, 512);
	OP(FC_1, MatrixMatrixFCOp);
	LINKUPPER(FC_1, Data_Tanh_0, Weight_FC_1);
	TENSOR(Data_FC_1, 8 , 10);
	LINKUPPER(Data_FC_1, FC_1);

	OP(Tanh_1, MatrixTanhOp);
	LINKUPPER(Tanh_1, Data_FC_1);
	TENSOR(Data_Tanh_1, 8 , 10);
	LINKUPPER(Data_Tanh_1, Tanh_1);


	OP(Softmax, MatrixSoftmaxOp);
	LINKUPPER(Softmax, Data_Tanh_1);
	TENSOR(Data_Softmax, 8, 10);
	LINKUPPER(Data_Softmax, Softmax);

	OP(PrintSoftmax, PrintMatrixOp);
	LINKUPPER(PrintSoftmax, Data_Softmax);

    // SoftmaxGrad
    TENSOR(Data_Selected, 8);
    Data_Selected_Tensor->setTensorInit(TensorInitType::FILE, "mnist_labels.bin");
	OP(SoftmaxGrad, MatrixSoftmaxGradOp);
	LINKUPPER(SoftmaxGrad, Data_Softmax, Data_Selected);

    TENSOR(Data_SoftmaxGrad, 8, 10);
    LINKUPPER(Data_SoftmaxGrad, SoftmaxGrad);

    // Tanh_1_Grad
    OP(Tanh_1_Grad, MatrixTanhGradOp); 
    LINKUPPER(Tanh_1_Grad, Data_Tanh_1, Data_SoftmaxGrad);

    TENSOR(Data_Tanh_1_Grad, 8, 10);
    LINKUPPER(Data_Tanh_1_Grad, Tanh_1_Grad);

    // Data_Tanh_0_Trans
    OP(Trans_0, MatrixTransOp);
    LINKUPPER(Trans_0, Data_Tanh_0);
    TENSOR(Data_Tanh_0_Trans, 512, 8);
    LINKUPPER(Data_Tanh_0_Trans, Trans_0);

    OP(MM_Grad_1, MatrixMatrixFCOp);
    LINKUPPER(MM_Grad_1, Data_Tanh_0_Trans, Data_Tanh_1_Grad);
    TENSOR(Weight_FC_1_Grad, 512, 10);
    LINKUPPER(Weight_FC_1_Grad, MM_Grad_1);

    OP(Add_1, MatrixAddOp);
    LINKUPPER(Add_1, Weight_FC_1_Grad, Weight_FC_1);
    // LINKUPPER(Weight_FC_1, Add_1);
    // LINKCHILD(Add_1, Weight_FC_1);

    // Data_Weight_FC_1
    OP(Trans_1, MatrixTransOp);
    LINKUPPER(Trans_1, Weight_FC_1);
    TENSOR(Data_Weight_FC_1_Trans, 10, 512);
    LINKUPPER(Data_Weight_FC_1_Trans, Trans_1);

    OP(MM_Grad_2, MatrixMatrixFCOp);
    LINKUPPER(MM_Grad_2, Data_Tanh_1_Grad, Data_Weight_FC_1_Trans); 
    TENSOR(Data_FC1_Grad, 8, 512);
    LINKUPPER(Data_FC1_Grad, MM_Grad_2);

    //Tanh_0_Grad
    OP(Tanh_0_Grad, MatrixTanhGradOp); 
    LINKUPPER(Tanh_0_Grad, Data_Tanh_0, Data_FC1_Grad);

    TENSOR(Data_Tanh_0_Grad, 8, 512);
    LINKUPPER(Data_Tanh_0_Grad, Tanh_0_Grad);

    // Data_0_Trans
    OP(Trans_2, MatrixTransOp);
    LINKUPPER(Trans_2, Data_0);
    TENSOR(Data_0_Trans, 784, 8);
    LINKUPPER(Data_0_Trans, Trans_2);

    OP(MM_Grad_3, MatrixMatrixFCOp);
    LINKUPPER(MM_Grad_3, Data_0_Trans, Data_Tanh_0_Grad);
    TENSOR(Weight_FC_0_Grad, 784, 512);
    LINKUPPER(Weight_FC_0_Grad, MM_Grad_3);

    OP(Add_0, MatrixAddOp);
    LINKUPPER(Add_0, Weight_FC_0_Grad, Weight_0);
    // LINKUPPER(Weight_0, Add_0);
    // LINKCHILD(Add_0, Weight_0);


	//define IR graph
	G(MLPLayer);
	GpT(MLPLayer, Data_0, Weight_0, Data_FC_0, Data_Tanh_0);
    GpT(MLPLayer, Weight_FC_1, Data_FC_1, Data_Tanh_1, Data_Softmax, Data_Selected);
    GpT(MLPLayer, Data_SoftmaxGrad, Data_Tanh_1_Grad, Data_Tanh_0_Trans, Weight_FC_1_Grad, Data_Weight_FC_1_Trans, Data_FC1_Grad);
    GpT(MLPLayer, Data_Tanh_0_Grad, Data_0_Trans, Weight_FC_0_Grad); 

	GpO(MLPLayer, FC_0, Tanh_0, FC_1, Tanh_1, Softmax, PrintSoftmax, MM_Grad_1, MM_Grad_2, MM_Grad_3);
    GpO(MLPLayer, SoftmaxGrad, Tanh_1_Grad, Tanh_0_Grad, Trans_0, Trans_1, Trans_2);
    GpO(MLPLayer, Add_1, Add_0);
	
	// GpO(MLPLayer, FC_0, Tanh_0, FC_1, Tanh_1, Softmax, SoftmaxGrad, Trans_0, Tanh_1_Grad, MM_Grad_1);
 	// GpO(MLPLayer, Trans_1, MM_Grad_2, Tanh_0_Grad, Trans_2, MM_Grad_3);
    

    printf ("Generate MLP layer done!\n");
    MLPLayer->findInOut();
    MLPLayer->updateTopology();
    MLPLayer->updateTopoNodeList();

    // Optimizer is a must because Codegen need label 
    Optimizer<Dtype>* opt = new Optimizer<Dtype>(MLPLayer);
    opt->runOptimizer();

    dotGen(MLPLayer);

    codegen::Codegen<Dtype>* cg = new codegen::Codegen<Dtype>(MLPLayer);
    string code = cg->generate();
    cout << code;

    for (int i = 0; i < MLPLayer->tensorNodeNum(); i++) {
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



