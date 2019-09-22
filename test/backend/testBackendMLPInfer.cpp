/*************************************************************************
	> File Name: test/testBackend.cpp
	> Author: wayne
	> Mail:  
	> Created Time: Sat 14 Sep 2019 11:08:12 AM UTC
 ************************************************************************/

#include <iostream>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace std;

int main() {

    TENSOR(data0, 8, 784);
    TENSOR(weight0, 784, 512);
    TENSOR(bias0, 512);
    data0_Tensor->setTensorInit(TensorInitType::FILE,
                                "input/mnist_images_8.bin");
    weight0_Tensor->setTensorInit(TensorInitType::FILE,
                                  "mlp_weight0.bin");
    bias0_Tensor->setTensorInit(TensorInitType::FILE, "mlp_bias0.bin");

    OP(fc0, MatrixMatrixFCBiasOp);
    LINKUPPER(fc0, data0, weight0, bias0);
    TENSOR(data1, 8, 512);
    LINKUPPER(data1, fc0);

    OP(tanh0, MatrixTanhOp);
    LINKUPPER(tanh0, data1);

    TENSOR(data2, 8, 512);
    LINKUPPER(data2, tanh0);

    TENSOR(weight1, 512, 10);
    TENSOR(bias1, 10);
    weight1_Tensor->setTensorInit(TensorInitType::FILE,
                                  "mlp_weight1.bin");
    bias1_Tensor->setTensorInit(TensorInitType::FILE, "mlp_bias1.bin");

    OP(fc1, MatrixMatrixFCBiasOp);
    LINKUPPER(fc1, data2, weight1, bias1);

    TENSOR(data3, 8, 10);
    LINKUPPER(data3, fc1);

    Tensor *labelt = new Tensor({8}, DataType::Int32_t);
    TensorNode *label = new TensorNode("selected", labelt);

    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data3);

    TENSOR(data4, 8, 10);
    LINKUPPER(data4, softmax);

    auto *argmax_o = new OpNode("argmax", new ArgMaxOp(1));
    argmax_o->exlinkUpperNode(data4);
    auto *top1_t =
        new TensorNode("top1", new Tensor({8, 1}, DataType::Int32_t), argmax_o);

    OP(accuracy_o, AccuracyOp);
    LINKUPPER(accuracy_o, top1_t, label); 
    auto *accuracy_t =
        new TensorNode("accur", new Tensor({1, 2}, DataType::Int32_t), accuracy_o);
    accuracy_t->getTensor()->setTensorInit(TensorInitType::CONSTANT, 0);

    // define IR graph
    G(mlp);
    GpT(mlp, data0, weight0, bias0, data1, data2, weight1, bias1, data3, data4, label, top1_t, accuracy_t);
    GpO(mlp, fc0, tanh0, fc1, softmax, argmax_o, accuracy_o);

    //====================================================
    mlp->findInOut();
    mlp->updateTopology();

    mlp->setInferDataNodes(label, data0);

    mlp->addDisplayTensorNodes(accuracy_t);
    //====================================================
    dotGen(mlp, "mlp_def.dot");

    //====================================================
    Config config;
    // config.mkldnn = true;
    config.use_dataloader = true;
    config.dataloader_src = "mnist_labels_images_10k.bin";  
    config.label_bytes = BytesProto::ONE_BYTE_AS_INT;
    config.data_bytes = BytesProto::FOUR_BYTES_AS_FLOAT;
    config.dataloader_samples= 10000;
    config.display = 10000 / 8;

    mlp->setConfig(config);
	
    Backend backend(mlp); 
    backend.compile();

    //dotGen(mlp, "mlp_compiled.dot");

    string code = backend.genCode();
    cout << code;

    return 0;
}
