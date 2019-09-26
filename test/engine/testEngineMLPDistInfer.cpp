/*************************************************************************
	> File Name: test/testEngine.cpp
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
                                  "input/mlp_weight0.bin");
    bias0_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias0.bin");

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
                                  "input/mlp_weight1.bin");
    bias1_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias1.bin");

    OP(fc1, MatrixMatrixFCBiasOp);
    LINKUPPER(fc1, data2, weight1, bias1);

    TENSOR(data3, 8, 10);
    LINKUPPER(data3, fc1);


    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data3);

    TENSOR(data4, 8, 10);
    LINKUPPER(data4, softmax);

    auto *argmax_o = new OpNode("argmax", new ArgMaxOp(3));
    argmax_o->exlinkUpperNode(data4);
    auto *top3_t =
        new TensorNode("top3", new Tensor({8, 3}, DataType::Int32_t), argmax_o);
    auto *print_o = new OpNode("print", new DebugOp());
    print_o->exlinkUpperNode(top3_t);
    auto *placeholder = new TensorNode("null", {0}, print_o);

    // define IR graph
    G(mlp);
    GpT(mlp, data0, weight0, bias0, data1, data2, weight1, bias1, data3, data4, top3_t, placeholder);
    GpO(mlp, fc0, tanh0, fc1, softmax, argmax_o, print_o);

    //====================================================
    mlp->findInOut();
    mlp->updateTopology();

    //====================================================
    dotGen(mlp, "mlp_def.dot");

    //====================================================
    Config config;
    config.mpi = true;
    config.mpi_size = 4;
    // config.mkldnn = true;
    mlp->setConfig(config);

	
    Engine engine(mlp); 
    engine.compile();

    dotGen(mlp, "mlp_compiled_parallel.dot");

    string code = engine.genCode();
    cout << code;

    return 0;
}
