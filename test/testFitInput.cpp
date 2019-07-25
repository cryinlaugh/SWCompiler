#include <iostream>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace std;

int main() {
    //============================
    // Example of 2-layer
    // fully connected network:
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

    //--------1st FC layer------------------------------------
    auto *data0 = new TensorNode("data0", {8, 784});
    data0->getTensor()->setTensorInit(TensorInitType::FILE,
                                      "input/mnist_images_8.bin");
    // we can only specify output neuron num 512
    // then infer wieght & bias dims
    // as long as we are creating a FC layer
    auto *weight0 = new TensorNode("weight0", {784, 512});
    auto *bias0 = new TensorNode("bias0", {512});

    auto *fc0 = new OpNode("fc0", new MatrixMatrixFCBiasOp());
    // order of input must be fixed
    // better not to expose this  to user
    fc0->exlinkUpperNode(data0, weight0, bias0);

    // dims of data1 should be inferred
    auto *data1 = new TensorNode("data1", fc0);

    auto *tanh0 = new OpNode("tanh0", new MatrixTanhOp());
    tanh0->exlinkUpperNode(data1);

    auto *data2 = new TensorNode("data2", tanh0);

    //--------2nd FC layer------------------------------------
    // How do you know that dim[0] of weight1 should be 512 ?
    // if top ops are like conv, pool
    auto *weight1 = new TensorNode("weight1", {0, 10});
    auto *bias1 = new TensorNode("bias1", {10});

    auto *fc1 = new OpNode("fc1", new MatrixMatrixFCBiasOp());
    fc1->exlinkUpperNode(data2, weight1, bias1);

    auto *data3 = new TensorNode("data3", fc1);

    auto *softmax = new OpNode("softmax", new MatrixSoftmaxOp());
    softmax->exlinkUpperNode(data3);

    auto *data4 = new TensorNode("data4", softmax);

    IRGraph *mlp = new IRGraph();
    mlp->pushTensorNode(data0, weight0, bias0, data1, data2, weight1, bias1,
                        data3, data4);
    mlp->pushOpNode(fc0, tanh0, fc1, softmax);

    // initialize tensor dimensions
    mlp->initTensorNodes();

    mlp->findInOut();
    mlp->updateTopology();

    pass::Optimizer *opt = new pass::Optimizer(mlp);
    opt->runOptimizer();
    dotGen(mlp);

    return 0;
}
