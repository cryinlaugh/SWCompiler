/*
 * testResnet.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-08-05
 */


#include <iostream>

#include "SWC.h"

#include <iostream>
#include "SWC.h"


using namespace std;
using namespace swc;
using namespace swc::op;

#define convbn_block(net, id, idc, upperNode,                                           \
        kernel_size, stride_size, pad_size, outChannel, inChannel,                      \
        bn_eps, bn_momentum)                                                            \
    TENSOR(conv_##id##idc##_w, outChannel,                                              \
            kernel_size, kernel_size, inChannel);                                       \
    TENSOR(conv_##id##idc##_b, outChannel);                                             \
    vector<size_t> conv_##id##idc##_kernels{kernel_size, kernel_size};                  \
    vector<size_t> conv_##id##idc##_strides{stride_size, stride_size};                  \
    vector<size_t> conv_##id##idc##_pads{pad_size, pad_size,                            \
            pad_size, pad_size};                                                        \
    DYOP(conv_##id##idc, Conv2dOp, conv_##id##idc##_kernels,                            \
            conv_##id##idc##_strides, conv_##id##idc##_pads);                           \
    LINKUPPER(conv_##id##idc, upperNode,                                                \
            conv_##id##idc##_w, conv_##id##idc##_b);                                    \
    TENSOR(conv_##id##idc##_out, 0);                                                    \
    LINKUPPER(conv_##id##idc##_out, conv_##id##idc);                                    \
    TENSOR(bn_##id##idc##_scale, 0);                                                    \
    TENSOR(bn_##id##idc##_shift, 0);                                                    \
    TENSOR(bn_##id##idc##_running_mean, 0);                                             \
    TENSOR(bn_##id##idc##_running_var, 0);                                              \
    DYOP(bn_##id##idc, BatchNormalizationOp, bn_eps, bn_momentum);                      \
    LINKUPPER(bn_##id##idc, conv_##id##idc##_out,                                       \
            bn_##id##idc##_scale, bn_##id##idc##_shift,                                 \
            bn_##id##idc##_running_mean, bn_##id##idc##_running_var);                   \
    TENSOR(bn_##id##idc##_out, 0);                                                      \
    LINKUPPER(bn_##id##idc##_out, bn_##id##idc);                                        \
    GpT(net, conv_##id##idc##_w, conv_##id##idc##_b, conv_##id##idc##_out,              \
            bn_##id##idc##_scale, bn_##id##idc##_shift,                                 \
            bn_##id##idc##_running_mean, bn_##id##idc##_running_var,                    \
            bn_##id##idc##_out);                                                        \
    GpO(net, conv_##id##idc, bn_##id##idc)


//This resblock for shortcut
#define resblock18_shortcut(net, id, upperNode,                                         \
        kernel_size_s, stride_size_s, pad_size_s, outChannel_s, inChannel_s,            \
        bn_eps_s, bn_momentum_s)                                                        \
    convbn_block(net, id, s, upperNode,                                                 \
            kernel_size_s, stride_size_s, pad_size_s, outChannel_s, inChannel_s,        \
            bn_eps_s, bn_momentum_s)


//This resblock for long road (with relu)
#define resblock18_left(net, id, upperNode,                                             \
        kernel_size_a, stride_size_a, pad_size_a, outChannel_a, inChannel_a,            \
        bn_eps_a, bn_momentum_a,                                                        \
        kernel_size_b, stride_size_b, pad_size_b, outChannel_b, inChannel_b,            \
        bn_eps_b, bn_momentum_b)                                                        \
    convbn_block(net, id, a, upperNode,                                                 \
            kernel_size_a, stride_size_a, pad_size_a, outChannel_a, inChannel_a,        \
            bn_eps_a, bn_momentum_a);                                                   \
    DYOP(relu_##id##a, ReluOp);                                                         \
    LINKUPPER(relu_##id##a, bn_##id##a_out);                                            \
    TENSOR(relu_##id##a_out, 0);                                                        \
    LINKUPPER(relu_##id##a_out, relu_##id##a);                                          \
    GpT(net, relu_##id##a_out);                                                         \
    GpO(net, relu_##id##a);                                                             \
    convbn_block(net, id, b, relu_##id##a_out,                                          \
            kernel_size_b, stride_size_b, pad_size_b, outChannel_b, inChannel_b,        \
            bn_eps_b, bn_momentum_b)



#define ResidualBlock18(net, id, upperNode, outChannel, inChannel, stride, bclass)      \
    resblock18_left(net, id, upperNode,                                                 \
        3, stride, 1, outChannel, inChannel, 0.0001, 0.9,                               \
        3, 1, 1, outChannel, outChannel,0.0001, 0.9);                                   \
    DYOP(res##id##bclass, ElementAddOp);                                                \
    if ((stride != 1) || (inChannel != outChannel)) {                                   \
        resblock18_shortcut(net, id, upperNode,                                         \
            1, stride, 0, outChannel, inChannel, 0.0001, 0.9);                          \
        LINKUPPER(res##id##bclass, bn_##id##s_out, bn_##id##b_out);                   \
    } else {                                                                            \
        LINKUPPER(res##id##bclass, upperNode, bn_##id##b_out);                         \
    }                                                                                   \
    TENSOR(res##id##bclass##_out, 0);                                                   \
    LINKUPPER(res##id##bclass##_out, res##id##bclass);                                  \
    DYOP(res##id##bclass##_relu, ReluOp);                                               \
    LINKUPPER(res##id##bclass##_relu, res##id##bclass##_out);                           \
    TENSOR(res##id##bclass##_relu_out, 0);                                              \
    LINKUPPER(res##id##bclass##_relu_out, res##id##bclass##_relu);                      \
    GpT(net, res##id##bclass##_out, res##id##bclass##_relu_out);                        \
    GpO(net, res##id##bclass, res##id##bclass##_relu)


int main() {

    //==========================
    // Example of Resnet-18 

    SWLOG_INFO << "Start generating graph..." << endl;

    G(resnet18);
    
    // data
    TENSOR(data, 128, 32, 32, 3);
    INIT(data, TensorInitType::FILE, "data_batch_1.bin");
    
    //conv1
    TENSOR(conv1_w, 64, 3, 3, 3);
    TENSOR(conv1_b, 64);
    vector<size_t> conv1_kernels{3, 3};
    vector<size_t> conv1_strides{1, 1};
    vector<size_t> conv1_pads{1, 1, 1, 1};
    DYOP(conv1, Conv2dOp, conv1_kernels, conv1_strides, conv1_pads);
    LINKUPPER(conv1, data, conv1_w,  conv1_b);
    TENSOR(conv1_out, 0);
    LINKUPPER(conv1_out, conv1);
    TENSOR(bn1_scale, 0);
    TENSOR(bn1_shift, 0);
    TENSOR(bn1_running_mean, 0);
    TENSOR(bn1_running_var, 0);
    DYOP(bn1, BatchNormalizationOp, 0.0001, 0.9);
    LINKUPPER(bn1, conv1_out, bn1_scale, bn1_shift, bn1_running_mean, bn1_running_var);
    TENSOR(bn1_out, 0);
    LINKUPPER(bn1_out, bn1);
    DYOP(relu1, ReluOp);
    LINKUPPER(relu1, bn1_out);
    TENSOR(relu1_out, 0);
    LINKUPPER(relu1_out, relu1);
    
    GpT(resnet18, data, conv1_w, conv1_b,
        conv1_out, bn1_out, 
        bn1_scale, bn1_shift, bn1_running_var, bn1_running_mean,
        relu1_out);
    GpO(resnet18, conv1, bn1, relu1);
    
    //resblock18_2a 
    ResidualBlock18(resnet18, 2, relu1_out, 64, 64, 1, a);
    //resblock18_3a 
    ResidualBlock18(resnet18, 3, res2a_relu_out, 64, 64, 1, a);

    //resblock18_4b 
    ResidualBlock18(resnet18, 4, res3a_relu_out, 128, 64, 2, b);
    //resblock18_5b 
    ResidualBlock18(resnet18, 5, res4b_relu_out, 128, 128, 1, b);
    
    //resblock18_6c
    ResidualBlock18(resnet18, 6, res5b_relu_out, 256, 128, 2, c);
    //resblock18_7c 
    ResidualBlock18(resnet18, 7, res6c_relu_out, 256, 256, 1, c);
    
    
    //resblock18_8d
    ResidualBlock18(resnet18, 8, res7c_relu_out, 512, 256, 2, d);
    //resblock18_9d 
    ResidualBlock18(resnet18, 9, res8d_relu_out, 512, 512, 1, d);
   

    
    vector<size_t> maxpool_kernels{4, 4};
    vector<size_t> maxpool_strides{1, 1};
    vector<size_t> maxpool_pads{0, 0, 0, 0};
    DYOP(maxpool, MaxPoolOp, maxpool_kernels, maxpool_strides, maxpool_pads);
    LINKUPPER(maxpool, res9d_relu_out);
    TENSOR(maxpool_out, 0);
    LINKUPPER(maxpool_out, maxpool);


    TENSOR(fc_weight, 512, 10);
    TENSOR(fc_bias, 10);
    OP(fc, MatrixMatrixFCBiasOp);
    LINKUPPER(fc, maxpool_out, fc_weight, fc_bias);
    TENSOR(fc_out, 0);
    LINKUPPER(fc_out, fc);


    GpT(resnet18, maxpool_out, fc_weight, fc_bias, fc_out); 
    GpO(resnet18, maxpool, fc);

    resnet18->initTensorNodes();

    resnet18->findInOut();
    resnet18->updateTopology();

    TRAIN(resnet18, "sgd", 0.001, 0.001, 0.9, 8);


    dotGen(resnet18_train);

}
