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
using namespace std;

int main()
{
    TENSOR(data0, 8, 28, 28, 1);
    INIT(data0, TensorInitType::FILE, "mnist_images_8.bin");

    TENSOR(conv0_w, 16, 5, 5, 1);
    INIT(conv0_w, TensorInitType::XAVIER, 784); // fanIn
    TENSOR(conv0_b, 16);
    INIT(conv0_b, TensorInitType::CONSTANT, 0);
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
    INIT(conv1_w, TensorInitType::XAVIER, 5*5*16); // fanIn
    TENSOR(conv1_b, 16);
    INIT(conv1_b, TensorInitType::CONSTANT, 0);
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
    OP(fc0, MatrixMatrixFCBiasOp);
    LINKUPPER(fc0, data6, fc0_w, fc0_b);
    TENSOR(data7, 0);
    LINKUPPER(data7, fc0);

    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data7);
    TENSOR(prob, 0);
    LINKUPPER(prob, softmax);

    G(lenet);
    GpT(lenet, data0, conv0_w, conv0_b,
    		data1, data2,
    		data3, conv1_w, conv1_b,
    		data4, data5, 
    		data6, fc0_w, fc0_b,
    		data7, prob);
    GpO(lenet, conv0, pool0, relu0,
    	conv1, pool1, relu1,
    	fc0, softmax);

    lenet->initTensorNodes();

    lenet->updateTopology();
    dotGen(lenet);

	return 0;
}
