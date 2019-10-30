#include <iostream>
#include <ctime>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace swc::pass;
using namespace std;

#define MINIBATCH 128 

int main() {

    TENSOR(input, MINIBATCH, 784);
    TENSOR(fc0_w, 0, 256);
    fc0_w_Tensor->setTensorInit(TensorInitType::XAVIER, 784);
    fc0_w->setTraining(1);

    OP(fc0_o, MatrixMatrixFCOp);
    LINKUPPER(fc0_o, input, fc0_w);

    TENSOR(fc0, 0);
    LINKUPPER(fc0, fc0_o);

    OP(tanh0_o, MatrixTanhOp);
    LINKUPPER(tanh0_o, fc0);

    TENSOR(tanh0, 0);
    LINKUPPER(tanh0, tanh0_o);


    TENSOR(fc1_w, 0, 256);
    fc1_w_Tensor->setTensorInit(TensorInitType::XAVIER, 256);
    fc1_w->setTraining(1);

    OP(fc1_o, MatrixMatrixFCOp);
    LINKUPPER(fc1_o, tanh0, fc1_w);

    TENSOR(fc1, 0);
    LINKUPPER(fc1, fc1_o);

    OP(tanh1_o, MatrixTanhOp);
    LINKUPPER(tanh1_o, fc1);

    TENSOR(tanh1, 0);
    LINKUPPER(tanh1, tanh1_o);

    TENSOR(fc2_w, 0, 10);
    fc2_w_Tensor->setTensorInit(TensorInitType::XAVIER, 256);
    fc2_w->setTraining(1);

    OP(fc2_o, MatrixMatrixFCOp);
    LINKUPPER(fc2_o, tanh1, fc2_w);

    TENSOR(fc2, 0);
    LINKUPPER(fc2, fc2_o);

    Tensor *labelt = new Tensor({MINIBATCH}, DataType::Int32_t);
    TensorNode *label = new TensorNode("selected", labelt);

    OP(softmax_o, MatrixSoftmaxWithLossOp);
    LINKUPPER(softmax_o, fc2, label);
    TENSOR(softmax, 0);
    LINKUPPER(softmax, softmax_o);
    TENSOR(loss, 1);
    LINKUPPER(loss, softmax_o);


    // define IR graph
    G(mlp);
    GpT(mlp, input, fc0_w, fc0, tanh0, fc1_w, fc1, tanh1, fc2_w, fc2,
        label, softmax, loss);
    GpO(mlp, fc0_o, tanh0_o, fc1_o, tanh1_o, fc2_o, softmax_o);


    mlp->findInOut();
    mlp->updateTopology();

    mlp->initTensorNodes();

    mlp->setTrainDataNodes(label, input);
    mlp->addDisplayTensorNodes(loss);


    Config config;

    config.train_mode = true;
    // config.mkldnn = true;
    config.mpi = true;
    config.mpi_size = 32;

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
    
    //config.parallel_preference = COMM_SAVING;
    config.parallel_preference = MEM_SAVING;
     
    /*when benchmark enabled, disable emit some code*/
    config.benchmark = true;
    /* not do lowering for node liek FC, FCGrad etc.*/
    config.enable_lowering = false;

    /* about parallel strategy*/
    config.force_data_parallel = true;
    // config.geneticalgo_opt_parallel = true;
    // config.handcraft_parallel = true;

    mlp->setConfig(config);

    dotGen(mlp, "mlp_def.dot");

    Engine engine(mlp);
    engine.compile();

    dotGen(mlp, "mlp_train.dot");

    cout << mlp->getCommTrace() << "\n";
    cout << mlp->getCommCost() << "\n";

    string code = engine.genCode();

    return 0;
}
