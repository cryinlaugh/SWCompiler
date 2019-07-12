/*************************************************************************
    > File Name: testConvnet.cpp
    > Author: wayne
    > Mail:
    > Created Time: 二  7/ 9 14:00:53 2019
 ************************************************************************/

#include "SWC.h"
#include <iostream>

using namespace swc;
using namespace swc::op;
using namespace swc::pass;
using namespace std;

int main()
{
    TENSOR(data0, 8, 28, 28, 1);

    TENSOR(conv0_w, 16, 5, 5, 1);
    TENSOR(conv0_b, 16);
    INIT(conv0_w, TensorInitType::XAVIER, 784); // fanIn
    INIT(conv0_b, TensorInitType::CONSTANT, 0);
    conv0_w->setTraining(1);
    conv0_b->setTraining(1);
    vector<size_t> conv0_kernels{5, 5};
    vector<size_t> conv0_strides{1, 1};
    vector<size_t> conv0_pads{2, 2, 2, 2};
    DYOP(conv0, Conv2dOp, conv0_kernels, conv0_strides, conv0_pads);
    LINKUPPER(conv0, data0, conv0_w, conv0_b);
    TENSOR(data1, 0);
    LINKUPPER(data1, conv0);

    vector<size_t> pool0_kernels{3, 3};
    vector<size_t> pool0_strides{3, 3};
    vector<size_t> pool0_pads{0, 0, 0, 0};
    DYOP(pool0, MaxPoolOp, pool0_kernels, pool0_strides, pool0_pads);
    LINKUPPER(pool0, data1);
    TENSOR(data2, 0);
    LINKUPPER(data2, pool0);

    OP(relu0, ReluOp);
    LINKUPPER(relu0, data2);
    TENSOR(data3, 0);
    LINKUPPER(data3, relu0);

    TENSOR(conv1_w, 16, 5, 5, 16);
    TENSOR(conv1_b, 16);
    INIT(conv1_w, TensorInitType::XAVIER, 5*5*16); // fanIn
    INIT(conv1_b, TensorInitType::CONSTANT, 0);
    conv1_w->setTraining(1);
    conv1_b->setTraining(1);
    vector<size_t> conv1_kernels{5, 5};
    vector<size_t> conv1_strides{1, 1};
    vector<size_t> conv1_pads{2, 2, 2, 2};
    DYOP(conv1, Conv2dOp, conv1_kernels, conv1_strides, conv1_pads);
    LINKUPPER(conv1, data3, conv1_w, conv1_b);
    TENSOR(data4, 0);
    LINKUPPER(data4, conv1);

    vector<size_t> pool1_kernels{3, 3};
    vector<size_t> pool1_strides{3, 3};
    vector<size_t> pool1_pads{0, 0, 0, 0};
    DYOP(pool1, MaxPoolOp, pool1_kernels, pool1_strides, pool1_pads);
    LINKUPPER(pool1, data4);
    TENSOR(data5, 0);
    LINKUPPER(data5, pool1);

    OP(relu1, ReluOp);
    LINKUPPER(relu1, data5);
    TENSOR(data6, 0);
    LINKUPPER(data6, relu1);

    TENSOR(fc0_w, 144, 10);
    TENSOR(fc0_b, 10);
    INIT(fc0_w, TensorInitType::XAVIER, 144); // fanIn
    INIT(fc0_b, TensorInitType::CONSTANT, 0);
    fc0_w->setTraining(1);
    fc0_b->setTraining(1);
    OP(fc0, MatrixMatrixFCBiasOp);
    LINKUPPER(fc0, data6, fc0_w, fc0_b);
    TENSOR(data7, 0);
    LINKUPPER(data7, fc0);


    Tensor *label_t = new Tensor({8}, DataType::Int32_t);
    TensorNode *label = new TensorNode("selected", label_t);

    // OP(softmax, MatrixSoftmaxOp);
    OP(softmax, MatrixSoftmaxWithLossOp);
    LINKUPPER(softmax, data7, label);
    TENSOR(prob, 0);
    LINKUPPER(prob, softmax);
    TENSOR(loss, 1);
    LINKUPPER(loss, softmax);

    G(lenet);
    GpT(lenet, data0, conv0_w, conv0_b,
    		data1, data2,
    		data3, conv1_w, conv1_b,
    		data4, data5,
    		data6, fc0_w, fc0_b,
    		data7, label, prob, loss);
    GpO(lenet, conv0, pool0, relu0,
    	conv1, pool1, relu1,
    	fc0, softmax);

    lenet->initTensorNodes();

    lenet->updateTopology();

    TRAIN(lenet, "sgd", 0.001, 0.001, 0.9, 8);

    // TrainingConfig profile;
    // profile.batch = data0->getDims()[0];
    // IRGraph *lenet_train = getTrainNet(lenet, profile);


    TensorNode *data_input = (TensorNode *)lenet_train->getNodeByName("data0");
    TensorNode *label_input = (TensorNode *)lenet_train->getNodeByName("selected");

    lenet_train->setTrainDataNodes(label_input, data_input);
    lenet_train->updateTopology();

    PassManager passManager;
    RenamingNodePass renamingpass(lenet_train);
    LabelingPass labelingpass(lenet_train);
    LoweringPass loweringpass(lenet_train);
    /*
    * puzzle:
    * 1. Labelingpass must before loweringpass (setLowerMark)
    * 2. ? renamingpass must before labeling,  or setNodeNameLabel, labels may duplicate
    * 3. renamingpass after loweringpass. because lowering may take dum names. "momentum", "xx_reshape"
    * 4. labeling pass must after loweringpass. because new op need to be labeled.
    * -----
    * ???
    * break the dependency between labeling and lowering.
    * lowering->renaming->labeling?
    */

    passManager.add((OptimizePass *)&labelingpass);
    passManager.add((OptimizePass *)&loweringpass);
    passManager.add((OptimizePass *)&renamingpass);
    passManager.add((OptimizePass *)&labelingpass);
    passManager.run();


    dotGen(lenet_train);



    CodegenConfig config;
    config.train_mode = true;
    config.train_config.train_data_file = "mnist_labels_images.bin";
    config.train_config.label_bytes = BytesProto::ONE_BYTE_AS_INT;
    config.train_config.data_bytes = BytesProto::FOUR_BYTES_AS_FLOAT;
    config.train_config.train_data_samples = 60000;

    codegen::Codegen *cg = new codegen::Codegen(lenet_train, config);

    string code = cg->generate();

    cout << code << "\n";

    return 0;
}
