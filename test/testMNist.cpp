/*************************************************************************
	> File Name: testMNist.cpp
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Thu 23 May 2019 08:35:52 AM UTC
 ************************************************************************/

#include<iostream>
#include "SWC.h"


using namespace swc;
using namespace swc::op;
using namespace std;

int main(){


    //============================
    // Example of LeNet-5 network for MNist
    //  
    // 2 convolutional layers + 2 pooling layer + 2 mlp layers
    // ====Conv Layer l1 ====
    // Params: batchsize, inputChannel, outputChannel, inputH, inputW, kernelSizeH, kernelSizeW, paddingH, paddingW, strideH, strideW
    // Input data: batchsize*1*32*32 (data0)
    // Weight data: 6*5*5 conv kernels + 6 bias (weight0, bias0)
    //
    // ====Pool layer l2 ====
    // Params: Max, poolingSizeH, poolingSizeW
    // Input data: batchsize*6*28*28 (data1)
    //
    // ====Relu layer ====
    // Input data: batchsize*6*14*14 (data2)
    // 
    // ====Conv Layer l3 ====
    // Params: batchsize, inputChannel, outputChannel, inputH, inputW, kernelSizeH, kernelSizeW, paddingH, paddingW, strideH, strideW
    // Input data: batchsize*6*14*14 (data3)
    // Weight data: 16*5*5 conv kernels + 16 bias (weight1, bias1)
    // 
    // ====Pool layer l4====
    // Params: Max, batchsize, poolingSizeH, poolingSizeW
    // Input data: batchsize*16*10*10 (data4)
    //
    // ====Relu layer ====
    // Input data: batchsize*16*5*5 (data5)
    //
    // ====Conv(MLP) layer l5====
    // Params: batchsize, inputLen, outputLen
    // Input data: batchsize*16*5*5 = batchsize*400 (data6)
    // Weight data: 400*120 mlp + 120 bias (weight2, bias2)
    //
    // ====Relu layer ====
    // Input data: batchsize*120 (data7)
    //
    // ====MLP layer l6====
    // Params: batchsize, inputLen, outputLen
    // Input data: batchsize*120 (data8)
    // Weight data: 120*84 mlp + 84 bias (weight3, bias3)
    //
    // ====Relu layer ====
    // Input data: batchsize*84 (data9)
    //
    // ====MLP layer l7====
    // Params: batchsize, inputLen, outputLen
    // Input data: batchsize*84 (data10)
    // Weight data: 84*10 mlp (weight4)
    //
    // ====Softmax layer====
    // Params: batchsize, inputLen, outputLen
    // Input data: batchsize*10 (data11)
    // Output data: batchsize*10 (prob)
    //
    //
    //  T:data_0   T:weight_0
    //     \       /
    //      \     /
    //        O:conv_0 -- T:bias_0
    //         |
    //      T:data_1
    //         |
    //      O:maxpool_0
    //         |
    //      T:data_2
    //         |
    //      O:Relu_0
    //         |
    //      T:data_3
    //                  T:weight_1
    //          \       /
    //           \     /
    //          O:conv_1 -- T:bias_1
    //              |
    //          T:data_4
    //              |
    //          O:maxpool_1
    //              |
    //          T:data_5
    //              |
    //          O:Relu_1
    //              |
    //          T:data_6  T:weight_2
    //              \          /
    //               \        /
    //                \      /
    //                 O:mlp_0  -- T:bias_2  
    //                    |
    //                 T:data_7
    //                    |
    //                 O:Relu_2
    //                    |
    //                 T:data_8  T:weight_3
    //                    \       /
    //                     \     /
    //                      O:mlp_1 -- T:bias_3
    //                        |
    //                      T:data_9
    //                        |
    //                      O:Relu_3
    //                        |
    //                      T:data_10  T:weight_4
    //                         \          /
    //                          \        /
    //                            O:mlp_2
    //                              |
    //                            T:data_11
    //                              |
    //                            O:softmax_0
    //                              |
    //                            T:prob
    //
    //=============================

    SWLOG_INFO<<"Start generating graph..."<<endl;

    TENSOR(data0, 256, 1, 32, 32);
    INIT(data0, TensorInitType::FILE, "mnist_images_8.bin");
    TENSOR(weight0, 1, 6, 5, 5);
    INIT(weight0, TensorInitType::XAVIER, 0.2);
    TENSOR(bias0, 6);
    INIT(bias0, TensorInitType::ZERO, 0);
    weight0->setTraining(1);
    bias0->setTraining(1);

    OP(conv0, Conv2dOp); 
    LINKUPPER(conv0, data0, weight0, bias0);

    TENSOR(data1,256, 6,28,28);
    LINKUPPER(data1, conv0);

    OP(pool0, MaxPoolOp);
    LINKUPPER(pool0, data1);

    TENSOR(data2, 256, 6, 14, 14);
    LINKUPPER(data2, pool0);

    OP(relu0, ReluOp);
    LINKUPPER(relu0, data2);

    TENSOR(data3, 256, 6, 14, 14);
    LINKUPPER(data3, relu0);

    TENSOR(weight1, 6,16,5,5);
    INIT(weight1, TensorInitType::XAVIER, 0.2);

    TENSOR(bias1, 16);
    INIT(bias1, TensorInitType::ZERO, 0);
    weight1->setTraining(1);
    bias1->setTraining(1);

    OP(conv1, Conv2dOp); 
    LINKUPPER(conv1, data3, weight1, bias1);

    TENSOR(data4, 256, 16,10,10);
    LINKUPPER(data4, conv1);
    
    OP(pool1, MaxPoolOp);
    LINKUPPER(pool1, data4);

    TENSOR(data5, 256, 16, 5, 5);
    LINKUPPER(data5, pool1);

    OP(relu1, ReluOp);
    LINKUPPER(relu1, data5);

    TENSOR(data6, 256, 16, 5, 5);
    LINKUPPER(data6, relu1);

    TENSOR(weight2, 400, 120);
    INIT(weight2, TensorInitType::XAVIER, 0.1);

    TENSOR(bias2, 120);
    INIT(bias2, TensorInitType::ZERO, 0);

    weight2->setTraining(1);
    bias2->setTraining(1);
    
    OP(mlp0, MatrixMatrixFCBiasOp);
    LINKUPPER(mlp0, data6, weight2, bias2);

    TENSOR(data7, 256, 120);
    LINKUPPER(data7, mlp0);

    OP(relu2, ReluOp);
    LINKUPPER(relu2, data7);

    TENSOR(data8, 256, 120);
    LINKUPPER(data8, relu2);

    TENSOR(weight3, 120, 84);
    INIT(weight3, TensorInitType::XAVIER, 0.2);

    TENSOR(bias3, 84);
    INIT(bias3, TensorInitType::ZERO, 0);

    weight3->setTraining(1);
    bias3->setTraining(1);
    
    OP(mlp1, MatrixMatrixFCBiasOp);
    LINKUPPER(mlp1, data8, weight3, bias3);

    TENSOR(data9, 256, 84);
    LINKUPPER(data9, mlp1);

    OP(relu3, ReluOp);
    LINKUPPER(relu3, data9);

    TENSOR(data10, 256, 84);
    LINKUPPER(data10, relu3);

    TENSOR(weight4, 84, 10);
    INIT(weight4, TensorInitType::XAVIER, 0.3);

    weight4->setTraining(1);
    
    OP(mlp2, MatrixMatrixFCOp);
    LINKUPPER(mlp2, data10, weight4);

    TENSOR(data11, 256, 10);
    LINKUPPER(data11, mlp2);

    TENSOR(label, 256, 10);
    INIT(label, TensorInitType::FILE, "mnist_images_8_label.bin");
    
    OP(softmax0, MatrixSoftmaxOp);
    LINKUPPER(softmax0, data11);
    LINKUPPER(softmax0, label);

    TENSOR(prob, 256, 10);
    TENSOR(loss, 1, 1);
    LINKUPPER(prob, softmax0);
    LINKUPPER(loss, softmax0);


    G(lenet5);
    GpT(lenet5, data0, data1,
            data2, data3, data4,
            data5, data6, data7,
            data8, data9, data10,
            data11, label, 
            weight0, weight1, weight2, weight3, weight4,
            bias0, bias1, bias2, bias3,
            prob, loss);
    GpO(lenet5, conv0, conv1,
            pool0, pool1,
            relu0, relu1, relu2, relu3,
            mlp0, mlp1, mlp2,
            softmax0);

    lenet5->updateTopology();

    TRAIN(lenet5, "SGD");

    lenet5_train->updateTopology();
    
    dotGen(lenet5_train);

    SWLOG_INFO<<"Start generating graph..."<<endl;



}
