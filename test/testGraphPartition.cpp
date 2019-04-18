#include <iostream>

#include "SWC.h"
#include "diff/AutoDiff.h"

#define Dtype float

using namespace swc;
using namespace std;

int main() {
    //============================
    // Example of 1 FC layer:
    //  T:data_0   T:weight_0
    //     \       /
    //      \     /
    //        O:fc_0 -- T:bias_0
    //         |
    //      T:data_1
    //         |
    //      O:tanh_0
    //         |
    //      T:data_2
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

    TENSOR(data_0, 8, 784);
    TENSOR(weight_0, 784, 512);
    TENSOR(bias_0, 512);
    data_0_Tensor->setTensorInit(TensorInitType::FILE, "mnist_images_8.bin");
    weight_0_Tensor->setTensorInit(TensorInitType::FILE, "mlp_weight_0.bin");
    bias_0_Tensor->setTensorInit(TensorInitType::FILE, "mlp_bias_0.bin");


    OP(fc_0, MatrixMatrixFCOp);
    LINKUPPER(fc_0, data_0, weight_0, bias_0);

    TENSOR(data_1, 8, 512);
    LINKUPPER(data_1, fc_0);

    OP(tanh_0, MatrixTanhOp);
    LINKUPPER(tanh_0, data_1);

    TENSOR(data_2, 8, 512);
    LINKUPPER(data_2, tanh_0);

    // define IR graph
    G(mlp);
    GpT(mlp, data_0, data_1, data_2, weight_0, bias_0);
    GpO(mlp, fc_0, tanh_0);

    TENSOR(weight_1, 512, 10);
    TENSOR(bias_1, 10);
    weight_1_Tensor->setTensorInit(TensorInitType::FILE, "mlp_weight_1.bin");
    bias_1_Tensor->setTensorInit(TensorInitType::FILE, "mlp_bias_1.bin");

    OP(fc_1, MatrixMatrixFCOp);
    LINKUPPER(fc_1, data_2, weight_1, bias_1);

    TENSOR(data_3, 8, 10);
    LINKUPPER(data_3, fc_1);

    Tensor *labelt = new Tensor({8}, DataType::Int32_t);
    TensorNode *labeln = new TensorNode("selected", labelt);
    // labelt->setTensorInit(TensorInitType::FILE, "mnist_labels.bin");

    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data_3, labeln);

    TENSOR(data_4, 8, 10);
    LINKUPPER(data_4, softmax);


    GpT(mlp, data_3, data_4, weight_1, bias_1, labeln);
    GpO(mlp, fc_1, softmax);

    CHECKT(data_0);
    CHECKT(weight_0);
    CHECKO(fc_0);
    CHECKT(data_1);
    CHECKO(tanh_0);
    CHECKT(data_2);
    CHECKG(mlp);

    bool res = mlp->buildSubGraph(data_0, data_2,
            ParallelStrategy::SLICE, 0, 1);
    if(res){
        std::cout << "build SubGraph Ok\n";
    }

    Device cpu0;
    Device dev_gpu0; dev_gpu0.type=DeviceType::GPU; dev_gpu0.id=0;

    mlp->updateTopology();
    Optimizer* opt = new Optimizer(mlp);
    opt->runOptimizer();
    dotGen(mlp);

    for(int i=0; i<mlp->opNodeNum(); i++){
        auto *node = mlp->getOpNode(i);
        if(auto *op = dynamic_cast<SubGraphOp*>(node->getOp())){
            std::cout << "subGraph Node " << node->name() << "\n";
            auto *subG = op->getGraph();

            // Be aware of implementation of IRGraph::setDeviceLabel()
            // external TensorNodes will not be labeled
            // since they are mirror of cpu Main IRGraph nodes.
            subG->setDeviceLabel(dev_gpu0);

            subG->updateTopology();
            opt->setGraph(subG);
            opt->runOptimizer();
            dotGen(subG, "subG.dot");
        }
    }

    codegen::Codegen *cg = new codegen::Codegen(mlp);
    string code = cg->generate();
    cout << code;

    return 0;
}
