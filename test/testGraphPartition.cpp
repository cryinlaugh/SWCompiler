#include <iostream>

#include "SWC.h"
#include "diff/AutoDiff.h"

#define Dtype float

using namespace swc;
using namespace swc::op;
using namespace std;

int main() {
    //============================
    // Example of 2 FC layer:
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
    data0_Tensor->setTensorInit(TensorInitType::FILE, "input/mnist_images_8.bin");
    weight0_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_weight0.bin");
    bias0_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias0.bin");

    OP(fc0, MatrixMatrixFCOp);
    LINKUPPER(fc0, data0, weight0, bias0);

    TENSOR(data1, 8, 512);
    LINKUPPER(data1, fc0);

    OP(tanh0, MatrixTanhOp);
    LINKUPPER(tanh0, data1);

    TENSOR(data2, 8, 512);
    LINKUPPER(data2, tanh0);

    // define IR graph
    G(mlp);
    GpT(mlp, data0, data1, data2, weight0, bias0);
    GpO(mlp, fc0, tanh0);

    TENSOR(weight1, 512, 10);
    TENSOR(bias1, 10);
    weight1_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_weight1.bin");
    bias1_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias1.bin");

    OP(fc1, MatrixMatrixFCOp);
    LINKUPPER(fc1, data2, weight1, bias1);

    TENSOR(data3, 8, 10);
    LINKUPPER(data3, fc1);

    Tensor *labelt = new Tensor({8}, DataType::Int32_t);
    TensorNode *labeln = new TensorNode("selected", labelt);
    // labelt->setTensorInit(TensorInitType::FILE, "mnist_labels.bin");

    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data3, labeln);

    TENSOR(data4, 8, 10);
    LINKUPPER(data4, softmax);

    GpT(mlp, data3, data4, weight1, bias1, labeln);
    GpO(mlp, fc1, softmax);

    CHECKT(data0);
    CHECKT(weight0);
    CHECKO(fc0);
    CHECKT(data1);
    CHECKO(tanh0);
    CHECKT(data2);
    CHECKG(mlp);

    bool res =
        mlp->buildSubGraphs(data0, data2, ParallelStrategy::SLICE, 0, 2);
    if (res) {
        std::cout << "build SubGraph Ok\n";
    }

    Device cpu0;
    Device dev_gpu[2];
    dev_gpu[0].type = DeviceType::GPU;
    dev_gpu[0].id = 0;
    dev_gpu[1].type = DeviceType::GPU;
    dev_gpu[1].id = 1;

    mlp->updateTopology();
    pass::Optimizer *opt = new pass::Optimizer(mlp);
    opt->runOptimizer();
    dotGen(mlp);

    int dev_id = 0;
    for (int i = 0; i < mlp->opNodeNum(); i++) {
        auto *node = mlp->getOpNode(i);
        if (auto *op = dynamic_cast<SubGraphOp *>(node->getOp())) {
            std::cout << "subGraph Node " << node->name() << "\n";
            auto *subG = op->getGraph();

            // Be aware of implementation of IRGraph::setDeviceLabel()
            // external TensorNodes will not be labeled
            // since they are mirror of cpu Main IRGraph nodes.

            subG->setDeviceLabel(dev_gpu[dev_id++]);

            subG->updateTopology();
            opt->setGraph(subG);
            opt->runOptimizer();
            dotGen(subG, node->name() + ".dot");
        }
    }

    codegen::Codegen *cg = new codegen::Codegen(mlp);
    string code = cg->generate();
    cout << code;

    SWLOG_DEBUG(10) << "debug test 10\n";

    return 0;
}
