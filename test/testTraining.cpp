#include <iostream>

#include "SWC.h"
#include "diff/AutoDiff.h"

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
    data_0_Tensor->setTensorInit(TensorInitType::FILE, "mnist_images.bin");
    weight_0_Tensor->setTensorInit(TensorInitType::XAVIER, 784);
    bias_0_Tensor->setTensorInit(TensorInitType::CONSTANT, 0);
    weight_0->setTraining(1);
    bias_0->setTraining(1);


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
    weight_1_Tensor->setTensorInit(TensorInitType::XAVIER, 512);
    bias_1_Tensor->setTensorInit(TensorInitType::CONSTANT, 0);
    weight_1->setTraining(1);
    bias_1->setTraining(1);

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

    mlp->updateTopology();


    SWLOG_DEBUG << "Start doing optimization on mlp." << std::endl;
    PassManager passManager;
    RenamingNodePass renamingpass(mlp);
    passManager.add((OptimizePass *)&renamingpass);
    LabelingPass labelingpass(mlp);
    passManager.add((OptimizePass *)&labelingpass);
    passManager.run();
    SWLOG_DEBUG << "Done doing optimization on mlp." << std::endl;

    /*
    Optimizer *opt = new Optimizer(mlp);
    opt->runOptimizer();
    dotGen(mlp);
    */

    TrainingProfile profile;
    profile.batch = data_0->getDims()[0];
    
    IRGraph *net = getTrainNet(mlp, profile);
    net->updateTopology();

    renamingpass.setGraph(net);
    labelingpass.setGraph(net);
    LoweringPass loweringpass(net);
    passManager.add((OptimizePass *)&renamingpass);
    passManager.add((OptimizePass *)&labelingpass);
    passManager.add((OptimizePass *)&loweringpass);
    passManager.add((OptimizePass *)&labelingpass);
    passManager.run();

    CHECKG(net);

    dotGen(net);

    codegen::Codegen *cg = new codegen::Codegen(net);
    string code = cg->generate();
    cout << code;

    return 0;
}
