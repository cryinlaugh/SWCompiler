#include <iostream>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace std;

int main() {
    //============================
    // Example of 2-layer
    // fully connected network:
    // data parallel, fc0 and tanh0
    // run on GPU0 and GPU1
    //
    //  T:data0   T:weight0
    //     \       /
    //      \     /
    //        O:fc0 -- T:bias0
    //         |
    //      T:data1
    //         |
    //      O:tanh0
    //         |
    //      T:data2
    //                  T:weight1
    //          \       /
    //           \     /
    //          O:fc1 -- T:bias1
    //              |
    //          T:data3
    //              |
    //          O: softmax
    //              |
    //          T:data4
    //=============================

    TENSOR(data0, 8, 784);
    TENSOR(weight0, 784, 512);
    TENSOR(bias0, 512);
    data0_Tensor->setTensorInit(TensorInitType::FILE,
                                "input/mnist_images_8.bin");
    weight0_Tensor->setTensorInit(TensorInitType::FILE,
                                  "input/mlp_weight0.bin");
    bias0_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias0.bin");

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

    TENSOR(weight1, 512, 10);
    TENSOR(bias1, 10);
    weight1_Tensor->setTensorInit(TensorInitType::FILE,
                                  "input/mlp_weight1.bin");
    bias1_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias1.bin");

    OP(fc1, MatrixMatrixFCBiasOp);
    LINKUPPER(fc1, data2, weight1, bias1);

    TENSOR(data3, 8, 10);
    LINKUPPER(data3, fc1);

    Tensor *labelt = new Tensor({8}, DataType::Int32_t);
    TensorNode *labeln = new TensorNode("selected", labelt);

    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data3, labeln);

    TENSOR(data4, 8, 10);
    LINKUPPER(data4, softmax);

    GpT(mlp, data3, data4, weight1, bias1, labeln);
    GpO(mlp, fc1, softmax);

    auto *argmax_o = new OpNode("argmax", new ArgMaxOp(3));
    argmax_o->exlinkUpperNode(data4);
    auto *top3_t =
        new TensorNode("top3", new Tensor({8, 3}, DataType::Int32_t), argmax_o);
    auto *print_o = new OpNode("print", new DebugOp());
    print_o->exlinkUpperNode(top3_t);

    mlp->pushOpNode(argmax_o, print_o);
    mlp->pushTensorNode(top3_t);
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

    OP(fc0_gpu0, MatrixMatrixFCBiasOp);
    LINKUPPER(fc0_gpu0, data0_gpu0, weight0_gpu0, bias0_gpu0);
    TENSOR(data1_gpu0, 4, 512);
    LINKUPPER(data1_gpu0, fc0_gpu0);

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
    subGraph0->pushOpNode(scatter00, scatter01, scatter02, fc0_gpu0, tanh0_gpu0,
                          gather0);

    // set these IRNode::_isExternal true to avoid
    // labeling them dev_gpu0
    data0_rep0->setExternal(true);
    weight0_rep0->setExternal(true);
    bias0_rep0->setExternal(true);
    data2_rep0->setExternal(true);

    subGraph0->setDeviceLabel(dev_gpu0);
    // data0_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    // weight0_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    // bias0_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    // data2_rep0->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
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

    OP(fc0_gpu1, MatrixMatrixFCBiasOp);
    LINKUPPER(fc0_gpu1, data0_gpu1, weight0_gpu1, bias0_gpu1);
    TENSOR(data1_gpu1, 4, 512);
    LINKUPPER(data1_gpu1, fc0_gpu1);

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
    subGraph1->pushOpNode(scatter10, scatter11, scatter12, fc0_gpu1, tanh0_gpu1,
                          gather1);

    // set these IRNode::_isExternal true to avoid
    // labeling them dev_gpu1
    data0_rep1->setExternal(true);
    weight0_rep1->setExternal(true);
    bias0_rep1->setExternal(true);
    data2_rep1->setExternal(true);

    subGraph1->setDeviceLabel(dev_gpu1);
    // data0_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    // weight0_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    // bias0_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);
    // data2_rep1->getLabel()->setDeviceLabel(cpu0.type, cpu0.id);

    //==================================
    gpu0_Op->setGraph(subGraph0);
    gpu1_Op->setGraph(subGraph1);

    mlp->updateTopology();
    pass::Optimizer *opt = new pass::Optimizer(mlp);
    opt->runOptimizer();

    subGraph0->updateTopology();
    opt->setGraph(subGraph0);
    opt->runOptimizer();

    subGraph1->updateTopology();
    opt->setGraph(subGraph1);
    opt->runOptimizer();

    //==================================
    dotGen(mlp);
    dotGen(subGraph0, "subGraph0.dot");
    dotGen(subGraph1, "subGraph1.dot");

    //==================================
    // nvcc -ccbin g++ -lcublas Graph.cu
    CodegenConfig config;
    config.cuda = true;
    config.cuda_stream = true;
    config.cublas = true;
    codegen::Codegen *cg = new codegen::Codegen(mlp, config);
    string code = cg->generate();
    cout << code;

    return 0;
}
