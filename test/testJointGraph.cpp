/*************************************************************************
    > File Name: testJointGraph.cpp
    > Author: wayne
    > Mail:
    > Created Time: 二  7/23 22:38:52 2019
 ************************************************************************/

#include <iostream>
using namespace std;
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

    DYOP(scatter0, ScatterOp, 0, 2);
    DYOP(scatter1, ScatterOp, -1, 2);
    DYOP(scatter2, ScatterOp, -1, 2);
    scatter1->setRunOnce();
    scatter2->setRunOnce();
    LINKUPPER(scatter0, data0);
    LINKUPPER(scatter1, weight0);
    LINKUPPER(scatter2, bias0);

    TENSOR(data0_p, 4, 784);
    TENSOR(weight0_p, 784, 512);
    TENSOR(bias0_p, 512);
    weight0_p_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    bias0_p_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
    LINKUPPER(data0_p, scatter0);
    LINKUPPER(weight0_p, scatter1);
    LINKUPPER(bias0_p, scatter2);

    // OP(fc0_p, MatrixMatrixFCBiasOp);
    // LINKUPPER(fc0_p, data0_p, weight0_p, bias0_p);
    // TENSOR(data1_p, 4, 512);
    // LINKUPPER(data1_p, fc0_p);
    OP(fc0_p_mm, MatrixMatrixMulOp);
    LINKUPPER(fc0_p_mm, data0_p, weight0_p);
    TENSOR(fc0_p_mm_out, 4, 512);
    LINKUPPER(fc0_p_mm_out, fc0_p_mm);
    OP(fc0_p_add, MatrixVectorAddOp);
    LINKUPPER(fc0_p_add, fc0_p_mm_out, bias0_p);
    TENSOR(data1_p, 4, 512);
    LINKUPPER(data1_p, fc0_p_add);

    OP(tanh0_p, MatrixTanhOp);
    LINKUPPER(tanh0_p, data1_p);
    TENSOR(data2_p, 4, 512);
    LINKUPPER(data2_p, tanh0_p);

    DYOP(gather0, GatherOp, 0, 2);
    LINKUPPER(gather0, data2_p);


    TENSOR(data2, 8, 512);
    LINKUPPER(data2, gather0);


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
    TensorNode *label = new TensorNode("selected", labelt);

    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data3, label);

    TENSOR(data4, 8, 10);
    LINKUPPER(data4, softmax);

    OpNode *argmax = new OpNode("argmax", new ArgMaxOp(3));
    argmax->exlinkUpperNode(data4);

    TensorNode *top3_idx =
        new TensorNode("top3", new Tensor({8, 3}, DataType::Int32_t), argmax);

    OpNode *print_top3 = new OpNode("print_top3", new DebugOp());
    print_top3->exlinkUpperNode(top3_idx);

    // define IR graph
    G(mlp);
    GpT(mlp, data0, weight0, bias0,
            data0_p, weight0_p, bias0_p,
            fc0_p_mm_out,
            data1_p, data2_p,
            data2, data3, weight1, bias1,
            data4, label, top3_idx);

    std::vector<TensorNode*> parallel_tnodes{data0_p, weight0_p, bias0_p,
            fc0_p_mm_out, data1_p, data2_p};
    for(auto *tn : parallel_tnodes) {
        tn->getLabel()->setDeviceLabel(INT_MAX, DeviceType::CPU, 0);
    }

    GpO(mlp, scatter0, scatter1, scatter2,
            fc0_p_mm, fc0_p_add,
            tanh0_p, gather0,
            fc1, softmax, argmax, print_top3);

    //====================================================
    mlp->findInOut();
    mlp->updateTopology();
    pass::Optimizer *opt = new pass::Optimizer(mlp);
    opt->runOptimizer();
    //====================================================
    dotGen(mlp);
    //====================================================
    CodegenConfig config;
    config.mpi = true;
    codegen::ParallelCodegen *cg = new codegen::ParallelCodegen(mlp, config);
    string code = cg->generate();
    cout << code;

    return 0;
}
