#include <iostream>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace std;

int main() {
    //============================
    // Example of 2-layer
    // Fully Connected network:
    // data parallel, fc0 and tanh0
    // run on different MPI processes.
    //
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
    weight0_Tensor->setTensorInit(TensorInitType::FILE, "mlp_weight0.bin");
    bias0_Tensor->setTensorInit(TensorInitType::FILE, "mlp_bias0.bin");

    //====================================================
    OP(cpu1, SubGraphOp);
    OP(cpu2, SubGraphOp);

    LINKUPPER(cpu1, data0, weight0, bias0);
    LINKUPPER(cpu2, data0, weight0, bias0);

    TENSOR(data2, 8, 512);
    LINKUPPER(data2, cpu1, cpu2);

    // define IR graph
    G(mlp);
    GpT(mlp, data0, data2, weight0, bias0);
    GpO(mlp, cpu1, cpu2);

    TENSOR(weight_1, 512, 10);
    TENSOR(bias_1, 10);
    weight_1_Tensor->setTensorInit(TensorInitType::FILE, "mlp_weight1.bin");
    bias_1_Tensor->setTensorInit(TensorInitType::FILE, "mlp_bias1.bin");

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

    OpNode *argmax = new OpNode("argmax", new ArgMaxOp(3));
    argmax->exlinkUpperNode(data_4);

    TensorNode *top3_idx =
        new TensorNode("top3", new Tensor({8, 3}, DataType::Int32_t), argmax);

    OpNode *print_top3 = new OpNode("print_top3", new DebugOp());
    print_top3->exlinkUpperNode(top3_idx);

    GpT(mlp, data_3, data_4, weight_1, bias_1, labeln);
    GpO(mlp, fc_1, softmax);

    mlp->pushOpNode(argmax, print_top3);
    mlp->pushTensorNode(top3_idx);

    //====================================================
    Device dev_cpu0;
    Device dev_cpu1;
    dev_cpu1.id = 1;
    Device dev_cpu2;
    dev_cpu2.id = 2;

    //-----------CPU1-------------------------------------
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

    TENSOR(data0_cpu1, 4, 784);
    TENSOR(weight0_cpu1, 784, 512);
    TENSOR(bias0_cpu1, 512);
    weight0_cpu1_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    bias0_cpu1_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    LINKUPPER(data0_cpu1, scatter00);
    LINKUPPER(weight0_cpu1, scatter01);
    LINKUPPER(bias0_cpu1, scatter02);

    OP(matmul0_cpu1, MatrixMatrixFCOp);
    LINKUPPER(matmul0_cpu1, data0_cpu1, weight0_cpu1, bias0_cpu1);
    TENSOR(data1_cpu1, 4, 512);
    LINKUPPER(data1_cpu1, matmul0_cpu1);

    OP(tanh0_cpu1, MatrixTanhOp);
    LINKUPPER(tanh0_cpu1, data1_cpu1);
    TENSOR(data2_cpu1, 4, 512);
    LINKUPPER(data2_cpu1, tanh0_cpu1);

    OP(gather0, GatherOp);
    LINKUPPER(gather0, data2_cpu1);

    TensorNode *data2_rep0 = new TensorNode("data2");
    data2_rep0->setTensor(data2->getTensor());
    LINKUPPER(data2_rep0, gather0);

    IRGraph *subGraph0 = new IRGraph();
    subGraph0->pushTensorNode(data0_rep0, weight0_rep0, bias0_rep0, data0_cpu1,
                              weight0_cpu1, bias0_cpu1, data1_cpu1, data2_cpu1,
                              data2_rep0);
    subGraph0->pushOpNode(scatter00, scatter01, scatter02, matmul0_cpu1,
                          tanh0_cpu1, gather0);

    data0_rep0->setExternal(true);
    weight0_rep0->setExternal(true);
    bias0_rep0->setExternal(true);
    data2_rep0->setExternal(true);
    subGraph0->setDeviceLabel(dev_cpu1);
    //-----------CPU1-------------------------------------
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

    TENSOR(data0_cpu2, 4, 784);
    TENSOR(weight0_cpu2, 784, 512);
    TENSOR(bias0_cpu2, 512);
    weight0_cpu2_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    bias0_cpu2_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    LINKUPPER(data0_cpu2, scatter10);
    LINKUPPER(weight0_cpu2, scatter11);
    LINKUPPER(bias0_cpu2, scatter12);

    OP(matmul0_cpu2, MatrixMatrixFCOp);
    LINKUPPER(matmul0_cpu2, data0_cpu2, weight0_cpu2, bias0_cpu2);
    TENSOR(data1_cpu2, 4, 512);
    LINKUPPER(data1_cpu2, matmul0_cpu2);

    OP(tanh0_cpu2, MatrixTanhOp);
    LINKUPPER(tanh0_cpu2, data1_cpu2);
    TENSOR(data2_cpu2, 4, 512);
    LINKUPPER(data2_cpu2, tanh0_cpu2);

    OP(gather1, GatherOp);
    auto *gop = (ScatterOp *)gather1->getOp();
    gop->setOffset(4 * 512);
    LINKUPPER(gather1, data2_cpu2);

    TensorNode *data2_rep1 = new TensorNode("data2");
    data2_rep1->setTensor(data2->getTensor());
    LINKUPPER(data2_rep1, gather1);

    IRGraph *subGraph1 = new IRGraph();
    subGraph1->pushTensorNode(data0_rep1, weight0_rep1, bias0_rep1, data0_cpu2,
                              weight0_cpu2, bias0_cpu2, data1_cpu2, data2_cpu2,
                              data2_rep1);
    subGraph1->pushOpNode(scatter10, scatter11, scatter12, matmul0_cpu2,
                          tanh0_cpu2, gather1);

    data0_rep1->setExternal(true);
    weight0_rep1->setExternal(true);
    bias0_rep1->setExternal(true);
    data2_rep1->setExternal(true);

    subGraph1->setDeviceLabel(dev_cpu2);
    //====================================================

    cpu1_Op->setGraph(subGraph0);
    cpu2_Op->setGraph(subGraph1);

    mlp->updateTopology();
    pass::Optimizer *opt = new pass::Optimizer(mlp);
    opt->runOptimizer();

    subGraph0->updateTopology();
    opt->setGraph(subGraph0);
    opt->runOptimizer();

    subGraph1->updateTopology();
    opt->setGraph(subGraph1);
    opt->runOptimizer();
    //====================================================

    dotGen(mlp);
    dotGen(subGraph0, "subGraph0.dot");
    dotGen(subGraph1, "subGraph1.dot");

    //====================================================
    CodegenConfig config;
    config.flag_MPI = true;
    codegen::Codegen *cg = new codegen::Codegen(mlp);
    string code = cg->generate();
    cout << code;

    return 0;
}
