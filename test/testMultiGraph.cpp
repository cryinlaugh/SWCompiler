/*************************************************************************
	> File Name: testMultiGraph.cpp
	> Author: wayne
	> Mail:  
	> Created Time: 四  3/21 16:52:59 2019
 ************************************************************************/

#include "SWC.h"
#include <iostream>

using namespace swc;
using namespace swc::op;
using namespace std;

int main() {
    cout << "In test MLP main" << endl;
    //============================
    // Example of 1 FC layer:
    //  T:data0   T:weight0
    //     \       /
    //      \     /
    //        matmul0
    //         |
    //      T:data1
    //         |
    //      O:tanh1
    //         |
    //      T:data2
    //		   |
    //      O:softmax
    //		   |
    //		T: data3
    //=============================

    TENSOR(data0, 8, 784);
    TENSOR(weight0, 784, 512);
    weight0_Tensor->setTensorInit(TensorInitType::XAVIER, 784);
    data0_Tensor->setTensorInit(TensorInitType::FILE, "mnist_images_8.bin");

    OP(gpu0, SubGraphOp);
    OP(gpu1, SubGraphOp);

    LINKUPPER(gpu0, data0, weight0);
    LINKUPPER(gpu1, data0, weight0);

    TENSOR(data2, 8, 512);
    LINKUPPER(data2, gpu0, gpu1);

    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data2);

    TENSOR(data3, 8, 512);
    LINKUPPER(data3, softmax);

    Device cpu0;
    Device dev_gpu0;
    dev_gpu0.type = DeviceType::GPU;
    dev_gpu0.id = 0;
    Device dev_gpu1;
    dev_gpu1.type = DeviceType::GPU;
    dev_gpu1.id = 1;
    //-----------GPU0-------------------------------------
    TensorNode *data0_rep0 = new TensorNode("data0");
    data0_rep0->setTensor(data0->getTensor());
    TensorNode *weight0_rep0 = new TensorNode("weight0");
    weight0_rep0->setTensor(weight0->getTensor());

    OP(scatter00, ScatterOp);
    OP(scatter01, ScatterOp);
    scatter01->setRunOnce();
    LINKUPPER(scatter00, data0_rep0);
    LINKUPPER(scatter01, weight0_rep0);

    TENSOR(data0_gpu0, 4, 784);
    TENSOR(weight0_gpu0, 784, 512);
    weight0_gpu0_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    LINKUPPER(data0_gpu0, scatter00);
    LINKUPPER(weight0_gpu0, scatter01);

    OP(matmul0_gpu0, MatrixMatrixMulOp);
    LINKUPPER(matmul0_gpu0, data0_gpu0, weight0_gpu0);
    TENSOR(data1_gpu0, 4, 512);
    LINKUPPER(data1_gpu0, matmul0_gpu0);

    OP(tanh0_gpu0, MatrixTanhOp);
    LINKUPPER(tanh0_gpu0, data1_gpu0);
    TENSOR(data2_gpu0, 4, 512);
    LINKUPPER(data2_gpu0, tanh0_gpu0);

    OP(gather0, GatherOp);
    LINKUPPER(gather0, data2_gpu0);

    TensorNode *data2_rep0 = new TensorNode("data2");
    data2_rep0->setTensor(data2->getTensor());
    LINKUPPER(data2_rep0, gather0);

    IRGraph *subGraph0 = new IRGraph();
    subGraph0->pushTensorNode(data0_rep0, weight0_rep0, data0_gpu0,
                              weight0_gpu0, data1_gpu0, data2_gpu0, data2_rep0);
    subGraph0->pushOpNode(scatter00, scatter01, matmul0_gpu0, tanh0_gpu0,
                          gather0);

    subGraph0->setDeviceLabel(dev_gpu0);
    data0_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    weight0_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    data2_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    //-----------GPU1-------------------------------------
    TensorNode *data0_rep1 = new TensorNode("data0");
    data0_rep1->setTensor(data0->getTensor());
    TensorNode *weight0_rep1 = new TensorNode("weight0");
    weight0_rep1->setTensor(weight0->getTensor());

    OP(scatter10, ScatterOp);
    OP(scatter11, ScatterOp);
    scatter11->setRunOnce();
    LINKUPPER(scatter10, data0_rep1);
    LINKUPPER(scatter11, weight0_rep1);

    TENSOR(data0_gpu1, 4, 784);
    TENSOR(weight0_gpu1, 784, 512);
    weight0_gpu1_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    LINKUPPER(data0_gpu1, scatter10);
    LINKUPPER(weight0_gpu1, scatter11);

    OP(matmul0_gpu1, MatrixMatrixMulOp);
    LINKUPPER(matmul0_gpu1, data0_gpu1, weight0_gpu1);
    TENSOR(data1_gpu1, 4, 512);
    LINKUPPER(data1_gpu1, matmul0_gpu1);

    OP(tanh0_gpu1, MatrixTanhOp);
    LINKUPPER(tanh0_gpu1, data1_gpu1);
    TENSOR(data2_gpu1, 4, 512);
    LINKUPPER(data2_gpu1, tanh0_gpu1);

    OP(gather1, GatherOp);
    LINKUPPER(gather1, data2_gpu1);

    TensorNode *data2_rep1 = new TensorNode("data2");
    data2_rep1->setTensor(data2->getTensor());
    LINKUPPER(data2_rep1, gather1);

    IRGraph *subGraph1 = new IRGraph();
    subGraph1->pushTensorNode(data0_rep1, weight0_rep1, data0_gpu1,
                              weight0_gpu1, data1_gpu1, data2_gpu1, data2_rep1);
    subGraph1->pushOpNode(scatter10, scatter11, matmul0_gpu1, tanh0_gpu1,
                          gather1);

    subGraph1->setDeviceLabel(dev_gpu1);
    data0_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    weight0_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    data2_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);

    gpu0_Op->setGraph(subGraph0);
    gpu1_Op->setGraph(subGraph1);

    IRGraph *MLPLayer = new IRGraph();
    MLPLayer->pushTensorNode(data0, weight0, data2, data3);
    MLPLayer->pushOpNode(gpu0, gpu1, softmax);
    printf("Generate MLP layer done!\n");

    // MLPLayer->updateTopoNodeList();
    // // Optimizer is a must because Codegen need label
    // Optimizer* opt = new Optimizer(MLPLayer);
    // opt->runOptimizer();

    subGraph1->updateTopology();
    pass::Optimizer *opt = new pass::Optimizer(subGraph1);
    opt->runOptimizer();
    // dotGen(subGraph1);

    subGraph0->updateTopology();
    opt->setGraph(subGraph0);
    opt->runOptimizer();

    MLPLayer->updateTopology();
    opt->setGraph(MLPLayer);
    opt->runOptimizer();
    dotGen(MLPLayer);

    codegen::Codegen *cg = new codegen::Codegen(MLPLayer);
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
