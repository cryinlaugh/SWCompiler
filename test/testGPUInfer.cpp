#include <iostream>

#include "SWC.h"
#include "diff/AutoDiff.h"

#define Dtype float

using namespace swc;
using namespace std;

int main() {
    //============================
    // Example of 2 FC layer:
    //  T:data0   T:weight0
    //     \       /
    //      \     /
    //        O:fc_0 -- T:bias0
    //         |
    //      T:data_1
    //         |
    //      O:tanh_0
    //         |
    //      T:data2
    //                  T:weight_1
    //          \       /
    //           \     /
    //          O:fc_1 -- T:bias_1
    //              |
    //          T:data_3
    //              |
    //          O: softmax
    //              |
    //          T:data_4
    //=============================

    TENSOR(data0, 8, 784);
    TENSOR(weight0, 784, 512);
    TENSOR(bias0, 512);
    data0_Tensor->setTensorInit(TensorInitType::FILE, "mnist_images_8.bin");
    weight0_Tensor->setTensorInit(TensorInitType::FILE, "mlp_weight_0.bin");
    bias0_Tensor->setTensorInit(TensorInitType::FILE, "mlp_bias_0.bin");

    //=======================================
    // run fc0 and tanh0 on gpu
    OP(gpu0, SubGraphOp);
    OP(gpu1, SubGraphOp);

    LINKUPPER(gpu0, data0, weight0, bias0);
    LINKUPPER(gpu1, data0, weight0, bias0);

    TENSOR(data2, 8, 512);
    LINKUPPER(data2, gpu0, gpu1);

    // define IR graph
    G(mlp);
    GpT(mlp, data0, data2, weight0, bias0);
    GpO(mlp, gpu0, gpu1);

    TENSOR(weight_1, 512, 10);
    TENSOR(bias_1, 10);
    weight_1_Tensor->setTensorInit(TensorInitType::FILE, "mlp_weight_1.bin");
    bias_1_Tensor->setTensorInit(TensorInitType::FILE, "mlp_bias_1.bin");

    OP(fc_1, MatrixMatrixFCOp);
    LINKUPPER(fc_1, data2, weight_1, bias_1);

    TENSOR(data_3, 8, 10);
    LINKUPPER(data_3, fc_1);

    Tensor *labelt = new Tensor({8}, DataType::Int32_t);
    TensorNode *labeln = new TensorNode("selected", labelt);

    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data_3, labeln);

    TENSOR(data_4, 8, 10);
    LINKUPPER(data_4, softmax);

    GpT(mlp, data_3, data_4, weight_1, bias_1, labeln);
    GpO(mlp, fc_1, softmax);

    //==================================
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
    TensorNode *bias0_rep0 = new TensorNode("bias_0");
    bias0_rep0->setTensor(bias0->getTensor());

    OP(scatter00, ScatterOp);
    OP(scatter01, ScatterOp);
    OP(scatter02, ScatterOp);
    scatter01->setRunOnce();
    scatter02->setRunOnce();
    LINKUPPER(scatter00, data0_rep0);
    LINKUPPER(scatter01, weight0_rep0);
    LINKUPPER(scatter02, bias0_rep0);

    TENSOR(data0_gpu0, 4, 784);
    TENSOR(weight0_gpu0, 784, 512);
    TENSOR(bias0_gpu0, 512);
    weight0_gpu0_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    bias0_gpu0_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    LINKUPPER(data0_gpu0, scatter00);
    LINKUPPER(weight0_gpu0, scatter01);
    LINKUPPER(bias0_gpu0, scatter02);

    OP(matmul0_gpu0, MatrixMatrixFCOp);
    LINKUPPER(matmul0_gpu0, data0_gpu0, weight0_gpu0, bias0_gpu0);
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
    subGraph0->pushTensorNode(data0_rep0, weight0_rep0, bias0_rep0, data0_gpu0,
                              weight0_gpu0, bias0_gpu0, data1_gpu0, data2_gpu0,
                              data2_rep0);
    subGraph0->pushOpNode(scatter00, scatter01, scatter02, matmul0_gpu0,
                          tanh0_gpu0, gather0);

    subGraph0->setDeviceLabel(dev_gpu0);
    data0_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    weight0_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    bias0_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    data2_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    //-----------GPU1-------------------------------------
    TensorNode *data0_rep1 = new TensorNode("data0");
    data0_rep1->setTensor(data0->getTensor());
    TensorNode *weight0_rep1 = new TensorNode("weight0");
    weight0_rep1->setTensor(weight0->getTensor());
    TensorNode *bias0_rep1 = new TensorNode("bias0");
    bias0_rep1->setTensor(bias0->getTensor());

    OP(scatter10, ScatterOp);
    OP(scatter11, ScatterOp);
    OP(scatter12, ScatterOp);
    auto *sop = (ScatterOp *)scatter10->getOp();
    sop->setOffset(4 * 784);
    scatter11->setRunOnce();
    scatter12->setRunOnce();
    LINKUPPER(scatter10, data0_rep1);
    LINKUPPER(scatter11, weight0_rep1);
    LINKUPPER(scatter12, bias0_rep1);

    TENSOR(data0_gpu1, 4, 784);
    TENSOR(weight0_gpu1, 784, 512);
    TENSOR(bias0_gpu1, 512);
    weight0_gpu1_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    bias0_gpu1_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    LINKUPPER(data0_gpu1, scatter10);
    LINKUPPER(weight0_gpu1, scatter11);
    LINKUPPER(bias0_gpu1, scatter12);

    OP(matmul0_gpu1, MatrixMatrixFCOp);
    LINKUPPER(matmul0_gpu1, data0_gpu1, weight0_gpu1, bias0_gpu1);
    TENSOR(data1_gpu1, 4, 512);
    LINKUPPER(data1_gpu1, matmul0_gpu1);

    OP(tanh0_gpu1, MatrixTanhOp);
    LINKUPPER(tanh0_gpu1, data1_gpu1);
    TENSOR(data2_gpu1, 4, 512);
    LINKUPPER(data2_gpu1, tanh0_gpu1);

    OP(gather1, GatherOp);
    auto *gop = (ScatterOp *)gather1->getOp();
    gop->setOffset(4 * 512);
    LINKUPPER(gather1, data2_gpu1);

    TensorNode *data2_rep1 = new TensorNode("data2");
    data2_rep1->setTensor(data2->getTensor());
    LINKUPPER(data2_rep1, gather1);

    IRGraph *subGraph1 = new IRGraph();
    subGraph1->pushTensorNode(data0_rep1, weight0_rep1, bias0_rep1, data0_gpu1,
                              weight0_gpu1, bias0_gpu1, data1_gpu1, data2_gpu1,
                              data2_rep1);
    subGraph1->pushOpNode(scatter10, scatter11, scatter12, matmul0_gpu1,
                          tanh0_gpu1, gather1);

    subGraph1->setDeviceLabel(dev_gpu1);
    data0_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    weight0_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    bias0_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    data2_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);

    gpu0_Op->setGraph(subGraph0);
    gpu1_Op->setGraph(subGraph1);

    subGraph1->updateTopology();
    Optimizer *opt = new Optimizer(subGraph1);
    opt->runOptimizer();

    subGraph0->updateTopology();
    opt->setGraph(subGraph0);
    opt->runOptimizer();

    mlp->updateTopology();
    opt->setGraph(mlp);
    opt->runOptimizer();

    dotGen(mlp);

    codegen::Codegen *cg = new codegen::Codegen(mlp);
    string code = cg->generate();
    cout << code;

    return 0;
}
