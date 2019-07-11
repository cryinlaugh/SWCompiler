#include <iostream>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace swc::pass;
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
    // data0_Tensor->setTensorInit(TensorInitType::FILE, "mnist_images.bin");
    weight0_Tensor->setTensorInit(TensorInitType::XAVIER, 784);
    bias0_Tensor->setTensorInit(TensorInitType::CONSTANT, 0);
    weight0->setTraining(1);
    bias0->setTraining(1);
    weight0->setExternal(true);
    bias0->setExternal(true);

    OP(fc0, MatrixMatrixFCBiasOp);
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
    weight1_Tensor->setTensorInit(TensorInitType::XAVIER, 512);
    bias1_Tensor->setTensorInit(TensorInitType::CONSTANT, 0);
    weight1->setTraining(1);
    bias1->setTraining(1);

    OP(fc1, MatrixMatrixFCBiasOp);
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

    mlp->updateTopology();

    SWLOG_INFO << "Start doing optimization on mlp." << std::endl;
    PassManager passManager;
    RenamingNodePass renamingpass(mlp);
    passManager.add((OptimizePass *)&renamingpass);
    LabelingPass labelingpass(mlp);
    passManager.add((OptimizePass *)&labelingpass);
    passManager.run();
    SWLOG_INFO << "Done doing optimization on mlp." << std::endl;

    /*
    Optimizer *opt = new Optimizer(mlp);
    opt->runOptimizer();
    dotGen(mlp);
    */

    TrainingConfig profile;
    profile.batch = data0->getDims()[0];

    IRGraph *net = getTrainNet(mlp, profile);

    TensorNode *data_input = (TensorNode *)net->getNodeByName("data0");
    TensorNode *label_input = (TensorNode *)net->getNodeByName("selected");

    net->setTrainDataNodes(label_input, data_input);
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

    CodegenConfig config;

    config.train_mode = true;
    config.train_config.train_data_file = "mnist_labels_images.bin";
    config.train_config.label_bytes = BytesProto::ONE_BYTE_AS_INT;
    config.train_config.data_bytes = BytesProto::FOUR_BYTES_AS_FLOAT;
    config.train_config.train_data_samples = 60000;

    codegen::Codegen *cg = new codegen::Codegen(net, config);

    string code = cg->generate();
    // cout << code;

    return 0;
}
