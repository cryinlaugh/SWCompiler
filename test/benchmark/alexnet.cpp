/*************************************************************************
	> File Name: test/testVGGTrain.cpp
	> Author: wayne
	> Mail:  
	> Created Time: Mon 23 Sep 2019 01:22:47 PM UTC
 ************************************************************************/

#include "SWC.h"
#include <iostream>

using namespace swc;
using namespace swc::op;
using namespace swc::pass;
using namespace std;

#define MINIBATCH 8 

int main()
{
    TENSOR(data, MINIBATCH, 32, 32, 3);

    TENSOR(conv1_w, 96, 3, 3, 3);
    TENSOR(conv1_b, 96);
    INIT(conv1_w, TensorInitType::XAVIER, 3*3*3); // fanIn
    INIT(conv1_b, TensorInitType::CONSTANT, 0);
    conv1_w->setTraining(1);
    conv1_b->setTraining(1);
    vector<size_t> conv1_kernels{3, 3};
    vector<size_t> conv1_strides{1, 1};
    vector<size_t> conv1_pads{1, 1, 1, 1};
    DYOP(conv1_o, Conv2dOp, conv1_kernels, conv1_strides, conv1_pads);
    LINKUPPER(conv1_o, data, conv1_w, conv1_b);
    TENSOR(conv1, 0);
    LINKUPPER(conv1, conv1_o);

    OP(relu1_o, ReluOp);
    LINKUPPER(relu1_o, conv1);
    TENSOR(relu1, 0);
    LINKUPPER(relu1, relu1_o);


    vector<size_t> pool1_kernels{2, 2};
    vector<size_t> pool1_strides{2, 2};
    vector<size_t> pool1_pads{0, 0, 0, 0};
    DYOP(pool1_o, MaxPoolOp, pool1_kernels, pool1_strides, pool1_pads);
    LINKUPPER(pool1_o, relu1);
    TENSOR(pool1, 0);
    LINKUPPER(pool1, pool1_o);

    OP(lrn1_o, LRNOp);
    LINKUPPER(lrn1_o, pool1);
    TENSOR(lrn1, 0);
    LINKUPPER(lrn1, lrn1_o);

    TENSOR(conv2_w, 256, 3, 3, 96);
    TENSOR(conv2_b, 256);
    INIT(conv2_w, TensorInitType::XAVIER, 3*3*96); // fanIn
    INIT(conv2_b, TensorInitType::CONSTANT, 0);
    conv2_w->setTraining(1);
    conv2_b->setTraining(1);
    vector<size_t> conv2_kernels{3, 3};
    vector<size_t> conv2_strides{1, 1};
    vector<size_t> conv2_pads{1, 1, 1, 1};
    DYOP(conv2_o, Conv2dOp, conv2_kernels, conv2_strides, conv2_pads);
    LINKUPPER(conv2_o, lrn1, conv2_w, conv2_b);
    TENSOR(conv2, 0);
    LINKUPPER(conv2, conv2_o);

    OP(relu2_o, ReluOp);
    LINKUPPER(relu2_o, conv2);
    TENSOR(relu2, 0);
    LINKUPPER(relu2, relu2_o);


    vector<size_t> pool2_kernels{2, 2};
    vector<size_t> pool2_strides{2, 2};
    vector<size_t> pool2_pads{0, 0, 0, 0};
    DYOP(pool2_o, MaxPoolOp, pool2_kernels, pool2_strides, pool2_pads);
    LINKUPPER(pool2_o, relu2);
    TENSOR(pool2, 0);
    LINKUPPER(pool2, pool2_o);

    OP(lrn2_o, LRNOp);
    LINKUPPER(lrn2_o, pool2);
    TENSOR(lrn2, 0);
    LINKUPPER(lrn2, lrn2_o);

    TENSOR(conv3_w, 384, 3, 3, 256);
    TENSOR(conv3_b, 384);
    INIT(conv3_w, TensorInitType::XAVIER, 3*3*256); // fanIn
    INIT(conv3_b, TensorInitType::CONSTANT, 0);
    conv3_w->setTraining(1);
    conv3_b->setTraining(1);
    vector<size_t> conv3_kernels{3, 3};
    vector<size_t> conv3_strides{1, 1};
    vector<size_t> conv3_pads{1, 1, 1, 1};
    DYOP(conv3_o, Conv2dOp, conv3_kernels, conv3_strides, conv3_pads);
    LINKUPPER(conv3_o, lrn2, conv3_w, conv3_b);
    TENSOR(conv3, 0);
    LINKUPPER(conv3, conv3_o);

    OP(relu3_o, ReluOp);
    LINKUPPER(relu3_o, conv3);
    TENSOR(relu3, 0);
    LINKUPPER(relu3, relu3_o);


    TENSOR(conv4_w, 384, 3, 3, 384);
    TENSOR(conv4_b, 384);
    INIT(conv4_w, TensorInitType::XAVIER, 3*3*384); // fanIn
    INIT(conv4_b, TensorInitType::CONSTANT, 0);
    conv4_w->setTraining(1);
    conv4_b->setTraining(1);
    vector<size_t> conv4_kernels{3, 3};
    vector<size_t> conv4_strides{1, 1};
    vector<size_t> conv4_pads{1, 1, 1, 1};
    DYOP(conv4_o, Conv2dOp, conv4_kernels, conv4_strides, conv4_pads);
    LINKUPPER(conv4_o, relu3, conv4_w, conv4_b);
    TENSOR(conv4, 0);
    LINKUPPER(conv4, conv4_o);

    OP(relu4_o, ReluOp);
    LINKUPPER(relu4_o, conv4);
    TENSOR(relu4, 0);
    LINKUPPER(relu4, relu4_o);


    TENSOR(conv5_w, 256, 3, 3, 384);
    TENSOR(conv5_b, 256);
    INIT(conv5_w, TensorInitType::XAVIER, 3*3*384); // fanIn
    INIT(conv5_b, TensorInitType::CONSTANT, 0);
    conv5_w->setTraining(1);
    conv5_b->setTraining(1);
    vector<size_t> conv5_kernels{3, 3};
    vector<size_t> conv5_strides{1, 1};
    vector<size_t> conv5_pads{1, 1, 1, 1};
    DYOP(conv5_o, Conv2dOp, conv5_kernels, conv5_strides, conv5_pads);
    LINKUPPER(conv5_o, relu4, conv5_w, conv5_b);
    TENSOR(conv5, 0);
    LINKUPPER(conv5, conv5_o);

    OP(relu5_o, ReluOp);
    LINKUPPER(relu5_o, conv5);
    TENSOR(relu5, 0);
    LINKUPPER(relu5, relu5_o);


    TENSOR(fc6_w, 0, 1024);
    TENSOR(fc6_b, 1024);
    INIT(fc6_w, TensorInitType::XAVIER, 16384); // fanIn 8*8*256
    INIT(fc6_b, TensorInitType::CONSTANT, 0);
    fc6_w->setTraining(1);
    fc6_b->setTraining(1);
    OP(fc6_o, MatrixMatrixFCBiasOp);
    LINKUPPER(fc6_o, relu5, fc6_w, fc6_b);
    TENSOR(fc6, 0);
    LINKUPPER(fc6, fc6_o);

    OP(relu6_o, ReluOp);
    LINKUPPER(relu6_o, fc6);
    TENSOR(relu6, 0);
    LINKUPPER(relu6, relu6_o);

    TENSOR(dropout6_mask, 0);
    DYOP(dropout6_o, DropoutOp, 0.5);
    LINKUPPER(dropout6_o, relu6, dropout6_mask);
    TENSOR(dropout6, 0);
    LINKUPPER(dropout6, dropout6_o);

    TENSOR(fc7_w, 0, 1024);
    TENSOR(fc7_b, 1024);
    INIT(fc7_w, TensorInitType::XAVIER, 1024); // fanIn
    INIT(fc7_b, TensorInitType::CONSTANT, 0);
    fc7_w->setTraining(1);
    fc7_b->setTraining(1);
    OP(fc7_o, MatrixMatrixFCBiasOp);
    LINKUPPER(fc7_o, dropout6, fc7_w, fc7_b);
    TENSOR(fc7, 0);
    LINKUPPER(fc7, fc7_o);

    OP(relu7_o, ReluOp);
    LINKUPPER(relu7_o, fc7);
    TENSOR(relu7, 0);
    LINKUPPER(relu7, relu7_o);

    TENSOR(dropout7_mask, 0);
    DYOP(dropout7_o, DropoutOp, 0.5);
    LINKUPPER(dropout7_o, relu7, dropout7_mask);
    TENSOR(dropout7, 0);
    LINKUPPER(dropout7, dropout7_o);


    TENSOR(fc8_w, 0, 10);
    TENSOR(fc8_b, 10);
    INIT(fc8_w, TensorInitType::XAVIER, 1024); // fanIn
    INIT(fc8_b, TensorInitType::CONSTANT, 0);
    fc8_w->setTraining(1);
    fc8_b->setTraining(1);
    OP(fc8_o, MatrixMatrixFCBiasOp);
    LINKUPPER(fc8_o, dropout7, fc8_w, fc8_b);
    TENSOR(fc8, 0);
    LINKUPPER(fc8, fc8_o);


    Tensor *label_t = new Tensor({MINIBATCH}, DataType::Int32_t);
    TensorNode *label = new TensorNode("selected", label_t);

    OP(softmax, MatrixSoftmaxWithLossOp);
    LINKUPPER(softmax, fc8, label);
    TENSOR(prob, 0);
    LINKUPPER(prob, softmax);
    TENSOR(loss, 1);
    LINKUPPER(loss, softmax);

    G(alexnet);
    GpT(alexnet,
            data, conv1_w, conv1_b, conv1, relu1, pool1, lrn1, 
            conv2_w, conv2_b, conv2, relu2, pool2, lrn2, 
            conv3_w, conv3_b, conv3, relu3,
            conv4_w, conv4_b, conv4, relu4,
            conv5_w, conv5_b, conv5, relu5,
            fc6, fc6_w, fc6_b, relu6, dropout6, dropout6_mask,
            fc7, fc7_w, fc7_b, relu7, dropout7, dropout7_mask,
            fc8, fc8_w, fc8_b,
    		label, prob, loss);
    GpO(alexnet,
            conv1_o, relu1_o, pool1_o, lrn1_o,
            conv2_o, relu2_o, pool2_o, lrn2_o,
            conv3_o, relu3_o,
            conv4_o, relu4_o,
            conv5_o, relu5_o,
            fc6_o, relu6_o, dropout6_o,
            fc7_o, relu7_o, dropout7_o,
            fc8_o,
            softmax);

    alexnet->findInOut();
    alexnet->updateTopology();

    alexnet->initTensorNodes();


    alexnet->setTrainDataNodes(label, data);
    alexnet->addDisplayTensorNodes(loss);

    Config config;

    config.train_mode = true;
    // config.mkldnn = true;
    config.mpi = true;
    config.mpi_size = 8;

    config.train_config.optimizer = "sgd";
    config.train_config.train_data_file = "mnist_labels_images.bin";
    config.train_config.label_bytes = BytesProto::ONE_BYTE_AS_INT;
    config.train_config.data_bytes = BytesProto::FOUR_BYTES_AS_FLOAT;
    config.train_config.train_data_samples = 60000;
    // config.train_config.snapshot = 1000;
    config.train_config.max_iters = 100;
    config.train_config.display = 50;

    // config.compute_op_annotation = true;
    // config.comm_op_annotation = true;
    
    config.parallel_preference = COMM_SAVING;
    // config.parallel_preference = MEM_SAVING;
     
    /*when benchmark enabled, disable emit some code*/
    config.benchmark = true;
    /* not do lowering for node liek FC, FCGrad etc.*/
    config.enable_lowering = false;

    /* about parallel strategy*/
    // config.force_data_parallel = true;
    // config.geneticalgo_opt_parallel = true;
    // config.handcraft_parallel = true;

    // optimzer
    config.decentralized_optimizer = true;

    alexnet->setConfig(config);
    std::cout << "alexnet_b" << MINIBATCH << "_p" << config.mpi_size << "\n";

    svgGen(alexnet, "alexnet_infer.dot");

    Engine engine(alexnet);
    engine.compile();

    dotGen(alexnet, "alexnet_train.dot");
    cout << alexnet->getCommTrace() << "\n";
    cout << alexnet->getCommCost() << "\n";

    string code = engine.genCode();
    // cout << code << "\n";

    return 0;
}
