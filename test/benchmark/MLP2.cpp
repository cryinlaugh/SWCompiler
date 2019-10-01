#include <iostream>
#include <ctime>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace swc::pass;
using namespace std;

#define MINIBATCH 128 

int main() {
    //============================
    // Example of 2 FC layer:
    //  T:input T:fc0_w
    //     \       /
    //      \     /
    //        O:fc0_o
    //         |
    //      T: fc0
    //         |
    //      O:tanh0_o_o
    //         |
    //      T: tanh0_o
    //                  T:fc1_w
    //          \       /
    //           \     /
    //          O:fc1_o
    //              |
    //          T: fc1
    //              |
    //          O: softmax_o
    //              |
    //          T: softmax
    //=============================

    TENSOR(input, MINIBATCH, 784);
    TENSOR(fc0_w, 784, 512);
    fc0_w_Tensor->setTensorInit(TensorInitType::XAVIER, 784);
    fc0_w->setTraining(1);
    // setExternal can be depreciated when we do autodiff
    // directly on IRGraph

    OP(fc0_o, MatrixMatrixFCOp);
    LINKUPPER(fc0_o, input, fc0_w);

    TENSOR(fc0, 0);
    LINKUPPER(fc0, fc0_o);

    OP(tanh0_o, MatrixTanhOp);
    LINKUPPER(tanh0_o, fc0);

    TENSOR(tanh0, 0);
    LINKUPPER(tanh0, tanh0_o);


    TENSOR(fc1_w, 0, 10);
    fc1_w_Tensor->setTensorInit(TensorInitType::XAVIER, 512);
    fc1_w->setTraining(1);

    OP(fc1_o, MatrixMatrixFCOp);
    LINKUPPER(fc1_o, tanh0, fc1_w);

    TENSOR(fc1, 0);
    LINKUPPER(fc1, fc1_o);

    Tensor *labelt = new Tensor({MINIBATCH}, DataType::Int32_t);
    TensorNode *label = new TensorNode("selected", labelt);

    OP(softmax_o, MatrixSoftmaxWithLossOp);
    LINKUPPER(softmax_o, fc1, label);
    TENSOR(softmax, 0);
    LINKUPPER(softmax, softmax_o);
    TENSOR(loss, 1);
    LINKUPPER(loss, softmax_o);


    // define IR graph
    G(mlp);
    GpT(mlp, input, fc0_w, fc0, tanh0, fc1_w, fc1,
        label, softmax, loss);
    GpO(mlp, fc0_o, tanh0_o, fc1_o, softmax_o);


    mlp->findInOut();
    mlp->updateTopology();

    mlp->initTensorNodes();

    mlp->setTrainDataNodes(label, input);
    mlp->addDisplayTensorNodes(loss);


    Config config;
    config.train_mode = true;
    // config.mkldnn = true;
    config.mpi = true;
    config.mpi_size = 4;
    config.train_config.optimizer = "sgd";
    config.train_config.train_data_file = "mnist_labels_images.bin";
    config.train_config.label_bytes = BytesProto::ONE_BYTE_AS_INT;
    config.train_config.data_bytes = BytesProto::FOUR_BYTES_AS_FLOAT;
    config.train_config.train_data_samples = 60000;
    // config.train_config.snapshot = 1000;
    config.train_config.max_iters = 100;
    config.train_config.display = 50;
    //config.compute_op_annotation = true;
    //config.comm_op_annotation = true;
    config.parallel_preference = COMM_SAVING;
    // config.parallel_preference = MEM_SAVING;
    // config.force_data_parallel = true;
    config.benchmark = true;

    mlp->setConfig(config);


    dotGen(mlp, "mlp_def.dot");


    Engine engine(mlp);
    engine.compile();

    dotGen(mlp, "mlp_train.dot");

    cout << mlp->getCommTrace() << "\n";
    cout << mlp->getCommCost() << "\n";

    string code = engine.genCode();
    // cout << code;

    return 0;
}
