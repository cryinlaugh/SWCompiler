#include "SWC.h"
#include <iostream>

using namespace swc;
using namespace swc::op;
using namespace swc::pass;
using namespace std;

#define MINIBATCH 8 

int main()
{
    TENSOR(data, MINIBATCH, 224, 224, 3);

    TENSOR(conv1_1_w, 64, 3, 3, 3);
    TENSOR(conv1_1_b, 64);
    INIT(conv1_1_w, TensorInitType::XAVIER, 3*3*3); // fanIn
    INIT(conv1_1_b, TensorInitType::CONSTANT, 0);
    conv1_1_w->setTraining(1);
    conv1_1_b->setTraining(1);
    vector<size_t> conv1_1_kernels{3, 3};
    vector<size_t> conv1_1_strides{1, 1};
    vector<size_t> conv1_1_pads{1, 1, 1, 1};
    DYOP(conv1_1_o, Conv2dWithActivationOp, conv1_1_kernels, conv1_1_strides, conv1_1_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv1_1_o, data, conv1_1_w, conv1_1_b);
    TENSOR(conv1_1, 0);
    LINKUPPER(conv1_1, conv1_1_o);

    TENSOR(conv1_2_w, 64, 3, 3, 64);
    TENSOR(conv1_2_b, 64);
    INIT(conv1_2_w, TensorInitType::XAVIER, 3*3*64); // fanIn
    INIT(conv1_2_b, TensorInitType::CONSTANT, 0);
    conv1_2_w->setTraining(1);
    conv1_2_b->setTraining(1);
    vector<size_t> conv1_2_kernels{3, 3};
    vector<size_t> conv1_2_strides{1, 1};
    vector<size_t> conv1_2_pads{1, 1, 1, 1};
    DYOP(conv1_2_o, Conv2dWithActivationOp, conv1_2_kernels, conv1_2_strides, conv1_2_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv1_2_o, conv1_1, conv1_2_w, conv1_2_b);
    TENSOR(conv1_2, 0);
    LINKUPPER(conv1_2, conv1_2_o);

    vector<size_t> pool1_kernels{2, 2};
    vector<size_t> pool1_strides{2, 2};
    vector<size_t> pool1_pads{0, 0, 0, 0};
    DYOP(pool1_o, MaxPoolOp, pool1_kernels, pool1_strides, pool1_pads);
    LINKUPPER(pool1_o, conv1_2);
    TENSOR(pool1, 0);
    LINKUPPER(pool1, pool1_o);


    TENSOR(conv2_1_w, 128, 3, 3, 64);
    TENSOR(conv2_1_b, 128);
    INIT(conv2_1_w, TensorInitType::XAVIER, 3*3*64); // fanIn
    INIT(conv2_1_b, TensorInitType::CONSTANT, 0);
    conv2_1_w->setTraining(1);
    conv2_1_b->setTraining(1);
    vector<size_t> conv2_1_kernels{3, 3};
    vector<size_t> conv2_1_strides{1, 1};
    vector<size_t> conv2_1_pads{1, 1, 1, 1};
    DYOP(conv2_1_o, Conv2dWithActivationOp, conv2_1_kernels, conv2_1_strides, conv2_1_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv2_1_o, pool1, conv2_1_w, conv2_1_b);
    TENSOR(conv2_1, 0);
    LINKUPPER(conv2_1, conv2_1_o);

    TENSOR(conv2_2_w, 128, 3, 3, 128);
    TENSOR(conv2_2_b, 128);
    INIT(conv2_2_w, TensorInitType::XAVIER, 3*3*128); // fanIn
    INIT(conv2_2_b, TensorInitType::CONSTANT, 0);
    conv2_2_w->setTraining(1);
    conv2_2_b->setTraining(1);
    vector<size_t> conv2_2_kernels{3, 3};
    vector<size_t> conv2_2_strides{1, 1};
    vector<size_t> conv2_2_pads{1, 1, 1, 1};
    DYOP(conv2_2_o, Conv2dWithActivationOp, conv2_2_kernels, conv2_2_strides, conv2_2_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv2_2_o, conv2_1, conv2_2_w, conv2_2_b);
    TENSOR(conv2_2, 0);
    LINKUPPER(conv2_2, conv2_2_o);

    vector<size_t> pool2_kernels{2, 2};
    vector<size_t> pool2_strides{2, 2};
    vector<size_t> pool2_pads{0, 0, 0, 0};
    DYOP(pool2_o, MaxPoolOp, pool2_kernels, pool2_strides, pool2_pads);
    LINKUPPER(pool2_o, conv2_2);
    TENSOR(pool2, 0);
    LINKUPPER(pool2, pool2_o);


    TENSOR(conv3_1_w, 256, 3, 3, 128);
    TENSOR(conv3_1_b, 256);
    INIT(conv3_1_w, TensorInitType::XAVIER, 3*3*128); // fanIn
    INIT(conv3_1_b, TensorInitType::CONSTANT, 0);
    conv3_1_w->setTraining(1);
    conv3_1_b->setTraining(1);
    vector<size_t> conv3_1_kernels{3, 3};
    vector<size_t> conv3_1_strides{1, 1};
    vector<size_t> conv3_1_pads{1, 1, 1, 1};
    DYOP(conv3_1_o, Conv2dWithActivationOp, conv3_1_kernels, conv3_1_strides, conv3_1_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv3_1_o, pool2, conv3_1_w, conv3_1_b);
    TENSOR(conv3_1, 0);
    LINKUPPER(conv3_1, conv3_1_o);


    TENSOR(conv3_2_w, 256, 3, 3, 256);
    TENSOR(conv3_2_b, 256);
    INIT(conv3_2_w, TensorInitType::XAVIER, 3*3*256); // fanIn
    INIT(conv3_2_b, TensorInitType::CONSTANT, 0);
    conv3_2_w->setTraining(1);
    conv3_2_b->setTraining(1);
    vector<size_t> conv3_2_kernels{3, 3};
    vector<size_t> conv3_2_strides{1, 1};
    vector<size_t> conv3_2_pads{1, 1, 1, 1};
    DYOP(conv3_2_o, Conv2dWithActivationOp, conv3_2_kernels, conv3_2_strides, conv3_2_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv3_2_o, conv3_1, conv3_2_w, conv3_2_b);
    TENSOR(conv3_2, 0);
    LINKUPPER(conv3_2, conv3_2_o);

    TENSOR(conv3_3_w, 256, 3, 3, 256);
    TENSOR(conv3_3_b, 256);
    INIT(conv3_3_w, TensorInitType::XAVIER, 3*3*256); // fanIn
    INIT(conv3_3_b, TensorInitType::CONSTANT, 0);
    conv3_3_w->setTraining(1);
    conv3_3_b->setTraining(1);
    vector<size_t> conv3_3_kernels{3, 3};
    vector<size_t> conv3_3_strides{1, 1};
    vector<size_t> conv3_3_pads{1, 1, 1, 1};
    DYOP(conv3_3_o, Conv2dWithActivationOp, conv3_3_kernels, conv3_3_strides, conv3_3_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv3_3_o, conv3_2, conv3_3_w, conv3_3_b);
    TENSOR(conv3_3, 0);
    LINKUPPER(conv3_3, conv3_3_o);

    TENSOR(conv3_4_w, 256, 3, 3, 256);
    TENSOR(conv3_4_b, 256);
    INIT(conv3_4_w, TensorInitType::XAVIER, 3*3*256); // fanIn
    INIT(conv3_4_b, TensorInitType::CONSTANT, 0);
    conv3_4_w->setTraining(1);
    conv3_4_b->setTraining(1);
    vector<size_t> conv3_4_kernels{3, 3};
    vector<size_t> conv3_4_strides{1, 1};
    vector<size_t> conv3_4_pads{1, 1, 1, 1};
    DYOP(conv3_4_o, Conv2dWithActivationOp, conv3_4_kernels, conv3_4_strides, conv3_4_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv3_4_o, conv3_3, conv3_4_w, conv3_4_b);
    TENSOR(conv3_4, 0);
    LINKUPPER(conv3_4, conv3_4_o);

    vector<size_t> pool3_kernels{2, 2};
    vector<size_t> pool3_strides{2, 2};
    vector<size_t> pool3_pads{0, 0, 0, 0};
    DYOP(pool3_o, MaxPoolOp, pool3_kernels, pool3_strides, pool3_pads);
    LINKUPPER(pool3_o, conv3_4);
    TENSOR(pool3, 0);
    LINKUPPER(pool3, pool3_o);


    TENSOR(conv4_1_w, 512, 3, 3, 256);
    TENSOR(conv4_1_b, 512);
    INIT(conv4_1_w, TensorInitType::XAVIER, 3*3*256); // fanIn
    INIT(conv4_1_b, TensorInitType::CONSTANT, 0);
    conv4_1_w->setTraining(1);
    conv4_1_b->setTraining(1);
    vector<size_t> conv4_1_kernels{3, 3};
    vector<size_t> conv4_1_strides{1, 1};
    vector<size_t> conv4_1_pads{1, 1, 1, 1};
    DYOP(conv4_1_o, Conv2dWithActivationOp, conv4_1_kernels, conv4_1_strides, conv4_1_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv4_1_o, pool3, conv4_1_w, conv4_1_b);
    TENSOR(conv4_1, 0);
    LINKUPPER(conv4_1, conv4_1_o);


    TENSOR(conv4_2_w, 512, 3, 3, 512);
    TENSOR(conv4_2_b, 512);
    INIT(conv4_2_w, TensorInitType::XAVIER, 3*3*512); // fanIn
    INIT(conv4_2_b, TensorInitType::CONSTANT, 0);
    conv4_2_w->setTraining(1);
    conv4_2_b->setTraining(1);
    vector<size_t> conv4_2_kernels{3, 3};
    vector<size_t> conv4_2_strides{1, 1};
    vector<size_t> conv4_2_pads{1, 1, 1, 1};
    DYOP(conv4_2_o, Conv2dWithActivationOp, conv4_2_kernels, conv4_2_strides, conv4_2_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv4_2_o, conv4_1, conv4_2_w, conv4_2_b);
    TENSOR(conv4_2, 0);
    LINKUPPER(conv4_2, conv4_2_o);

    TENSOR(conv4_3_w, 512, 3, 3, 512);
    TENSOR(conv4_3_b, 512);
    INIT(conv4_3_w, TensorInitType::XAVIER, 3*3*512); // fanIn
    INIT(conv4_3_b, TensorInitType::CONSTANT, 0);
    conv4_3_w->setTraining(1);
    conv4_3_b->setTraining(1);
    vector<size_t> conv4_3_kernels{3, 3};
    vector<size_t> conv4_3_strides{1, 1};
    vector<size_t> conv4_3_pads{1, 1, 1, 1};
    DYOP(conv4_3_o, Conv2dWithActivationOp, conv4_3_kernels, conv4_3_strides, conv4_3_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv4_3_o, conv4_2, conv4_3_w, conv4_3_b);
    TENSOR(conv4_3, 0);
    LINKUPPER(conv4_3, conv4_3_o);

    TENSOR(conv4_4_w, 512, 3, 3, 512);
    TENSOR(conv4_4_b, 512);
    INIT(conv4_4_w, TensorInitType::XAVIER, 3*3*512); // fanIn
    INIT(conv4_4_b, TensorInitType::CONSTANT, 0);
    conv4_4_w->setTraining(1);
    conv4_4_b->setTraining(1);
    vector<size_t> conv4_4_kernels{3, 3};
    vector<size_t> conv4_4_strides{1, 1};
    vector<size_t> conv4_4_pads{1, 1, 1, 1};
    DYOP(conv4_4_o, Conv2dWithActivationOp, conv4_4_kernels, conv4_4_strides, conv4_4_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv4_4_o, conv4_3, conv4_4_w, conv4_4_b);
    TENSOR(conv4_4, 0);
    LINKUPPER(conv4_4, conv4_4_o);

    vector<size_t> pool4_kernels{2, 2};
    vector<size_t> pool4_strides{2, 2};
    vector<size_t> pool4_pads{0, 0, 0, 0};
    DYOP(pool4_o, MaxPoolOp, pool4_kernels, pool4_strides, pool4_pads);
    LINKUPPER(pool4_o, conv4_4);
    TENSOR(pool4, 0);
    LINKUPPER(pool4, pool4_o);


    TENSOR(conv5_1_w, 512, 3, 3, 512);
    TENSOR(conv5_1_b, 512);
    INIT(conv5_1_w, TensorInitType::XAVIER, 3*3*512); // fanIn
    INIT(conv5_1_b, TensorInitType::CONSTANT, 0);
    conv5_1_w->setTraining(1);
    conv5_1_b->setTraining(1);
    vector<size_t> conv5_1_kernels{3, 3};
    vector<size_t> conv5_1_strides{1, 1};
    vector<size_t> conv5_1_pads{1, 1, 1, 1};
    DYOP(conv5_1_o, Conv2dWithActivationOp, conv5_1_kernels, conv5_1_strides, conv5_1_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv5_1_o, pool4, conv5_1_w, conv5_1_b);
    TENSOR(conv5_1, 0);
    LINKUPPER(conv5_1, conv5_1_o);


    TENSOR(conv5_2_w, 512, 3, 3, 512);
    TENSOR(conv5_2_b, 512);
    INIT(conv5_2_w, TensorInitType::XAVIER, 3*3*512); // fanIn
    INIT(conv5_2_b, TensorInitType::CONSTANT, 0);
    conv5_2_w->setTraining(1);
    conv5_2_b->setTraining(1);
    vector<size_t> conv5_2_kernels{3, 3};
    vector<size_t> conv5_2_strides{1, 1};
    vector<size_t> conv5_2_pads{1, 1, 1, 1};
    DYOP(conv5_2_o, Conv2dWithActivationOp, conv5_2_kernels, conv5_2_strides, conv5_2_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv5_2_o, conv5_1, conv5_2_w, conv5_2_b);
    TENSOR(conv5_2, 0);
    LINKUPPER(conv5_2, conv5_2_o);

    TENSOR(conv5_3_w, 512, 3, 3, 512);
    TENSOR(conv5_3_b, 512);
    INIT(conv5_3_w, TensorInitType::XAVIER, 3*3*512); // fanIn
    INIT(conv5_3_b, TensorInitType::CONSTANT, 0);
    conv5_3_w->setTraining(1);
    conv5_3_b->setTraining(1);
    vector<size_t> conv5_3_kernels{3, 3};
    vector<size_t> conv5_3_strides{1, 1};
    vector<size_t> conv5_3_pads{1, 1, 1, 1};
    DYOP(conv5_3_o, Conv2dWithActivationOp, conv5_3_kernels, conv5_3_strides, conv5_3_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv5_3_o, conv5_2, conv5_3_w, conv5_3_b);
    TENSOR(conv5_3, 0);
    LINKUPPER(conv5_3, conv5_3_o);

    TENSOR(conv5_4_w, 512, 3, 3, 512);
    TENSOR(conv5_4_b, 512);
    INIT(conv5_4_w, TensorInitType::XAVIER, 3*3*512); // fanIn
    INIT(conv5_4_b, TensorInitType::CONSTANT, 0);
    conv5_4_w->setTraining(1);
    conv5_4_b->setTraining(1);
    vector<size_t> conv5_4_kernels{3, 3};
    vector<size_t> conv5_4_strides{1, 1};
    vector<size_t> conv5_4_pads{1, 1, 1, 1};
    DYOP(conv5_4_o, Conv2dWithActivationOp, conv5_4_kernels, conv5_4_strides, conv5_4_pads, SWC_ACTIVATION_RELU);
    LINKUPPER(conv5_4_o, conv5_3, conv5_4_w, conv5_4_b);
    TENSOR(conv5_4, 0);
    LINKUPPER(conv5_4, conv5_4_o);

    vector<size_t> pool5_kernels{2, 2};
    vector<size_t> pool5_strides{2, 2};
    vector<size_t> pool5_pads{0, 0, 0, 0};
    DYOP(pool5_o, MaxPoolOp, pool5_kernels, pool5_strides, pool5_pads);
    LINKUPPER(pool5_o, conv5_4);
    TENSOR(pool5, 0);
    LINKUPPER(pool5, pool5_o);


    TENSOR(fc6_w, 0, 4096);
    TENSOR(fc6_b, 4096);
    INIT(fc6_w, TensorInitType::XAVIER, 25088); // fanIn
    INIT(fc6_b, TensorInitType::CONSTANT, 0);
    fc6_w->setTraining(1);
    fc6_b->setTraining(1);
    OP(fc6_o, MatrixMatrixFCBiasOp);
    LINKUPPER(fc6_o, pool5, fc6_w, fc6_b);
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

    TENSOR(fc7_w, 0, 4096);
    TENSOR(fc7_b, 4096);
    INIT(fc7_w, TensorInitType::XAVIER, 4096); // fanIn
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


    TENSOR(fc8_w, 0, 1000);
    TENSOR(fc8_b, 1000);
    INIT(fc8_w, TensorInitType::XAVIER, 4096); // fanIn
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

    G(vgg19);
    GpT(vgg19,
            data, conv1_1_w, conv1_1_b, conv1_1,
            conv1_2_w, conv1_2_b, conv1_2,
            pool1,
            conv2_1_w, conv2_1_b, conv2_1,
            conv2_2_w, conv2_2_b, conv2_2,
            pool2,
            conv3_1_w, conv3_1_b, conv3_1,
            conv3_2_w, conv3_2_b, conv3_2,
            conv3_3_w, conv3_3_b, conv3_3,
            conv3_4_w, conv3_4_b, conv3_4,
            pool3,
            conv4_1_w, conv4_1_b, conv4_1,
            conv4_2_w, conv4_2_b, conv4_2,
            conv4_3_w, conv4_3_b, conv4_3,
            conv4_4_w, conv4_4_b, conv4_4,
            pool4,
            conv5_1_w, conv5_1_b, conv5_1,
            conv5_2_w, conv5_2_b, conv5_2,
            conv5_3_w, conv5_3_b, conv5_3,
            conv5_4_w, conv5_4_b, conv5_4,
            pool5,
            fc6, fc6_w, fc6_b, relu6, dropout6, dropout6_mask,
            fc7, fc7_w, fc7_b, relu7, dropout7, dropout7_mask,
            fc8, fc8_w, fc8_b,
    		label, prob, loss);
    GpO(vgg19,
            conv1_1_o,
            conv1_2_o,
            pool1_o,
            conv2_1_o,
            conv2_2_o,
            pool2_o,
            conv3_1_o,
            conv3_2_o,
            conv3_3_o,
            conv3_4_o,
            pool3_o,
            conv4_1_o,
            conv4_2_o,
            conv4_3_o,
            conv4_4_o,
            pool4_o,
            conv5_1_o,
            conv5_2_o,
            conv5_3_o,
            conv5_4_o,
            pool5_o,
            fc6_o, relu6_o, dropout6_o,
            fc7_o, relu7_o, dropout7_o,
            fc8_o,
            softmax);

    vgg19->findInOut();
    vgg19->updateTopology();

    vgg19->initTensorNodes();


    vgg19->setTrainDataNodes(label, data);
    vgg19->addDisplayTensorNodes(loss);

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
     
    /*when benchmark enabled, disable emit some code*/
    config.benchmark = true;
    /* not do lowering for node liek FC, FCGrad etc.*/
    config.enable_lowering = false;

    /* about parallel strategy*/
    config.force_data_parallel = true;
    // config.geneticalgo_opt_parallel = true;
    // config.handcraft_parallel = true;

    vgg19->setConfig(config);

    dotGen(vgg19, "vgg19_infer.dot");

    Engine engine(vgg19);
    engine.compile();

    //dotGen(vgg19, "vgg19_train.dot");
    cout << vgg19->getCommTrace() << "\n";
    cout << vgg19->getCommCost() << "\n";

    string code = engine.genCode();
    // cout << code << "\n";

    return 0;
}
