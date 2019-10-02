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

//This resblock for shortcut
#define resblock_shortcut(id, upperNode,                                                \
        kernel_size_s, stride_size_s, pad_size_s, outChannel_s, inChannel_s,            \
        bn_eps_s, bn_momentum_s)                                                        \
    TENSOR(conv_s_##id##_w, outChannel_s, kernel_size_s, kernel_size_s, inChannel_s);   \
    TENSOR(conv_s_##id##_b, outChannel_s);                                              \
    vector<size_t> conv_s_##id##_kernels{kernel_size_s, kernel_size_s};                 \
    vector<size_t> conv_s_##id##_strides{stride_size_s, stride_size_s};                 \
    vector<size_t> conv_s_##id##_pads{pad_size_s, pad_size_s, pad_size_s, pad_size_s};  \
    DYOP(conv_s_##id, Conv2dOp,                                                         \
            conv_s_##id##_kernels, conv_s_##id##_strides, conv_s_##id##_pads);          \
    LINKUPPER(conv_s_##id, upperNode, conv_s_##id##_w, conv_s_##id##_b);                \
    TENSOR(conv_s_##id##_out, 0);                                                       \
    LINKUPPER(conv_s_##id##_out, conv_s_##id);                                          \
    TENSOR(bn_s_##id##_scale, 0);                                                       \
    TENSOR(bn_s_##id##_shift, 0);                                                       \
    TENSOR(bn_s_##id##_running_mean, 0);                                                \
    TENSOR(bn_s_##id##_running_var, 0);                                                 \
    DYOP(bn_s_##id, BatchNormalizationOp, bn_eps_s, bn_momentum_s);                     \
    LINKUPPER(bn_s_##id, conv_s_##id##_out, bn_s_##id##_scale, bn_s_##id##_shift,       \
            bn_s_##id##_running_mean, bn_s_##id##_running_var);                         \
    TENSOR(bn_s_##id##_out, 0);                                                         \
    LINKUPPER(bn_s_##id##_out, bn_s_##id);                                              \


//This resblock for long road (with relu)
#define resblock(id, upperNode,                                                         \
        kernel_size_a, stride_size_a, pad_size_a, outChannel_a, inChannel_a,            \
        bn_eps_a, bn_momentum_a,                                                        \
        kernel_size_b, stride_size_b, pad_size_b, outChannel_b, inChannel_b,            \
        bn_eps_b, bn_momentum_b,                                                        \
        kernel_size_c, stride_size_c, pad_size_c, outChannel_c, inChannel_c,            \
        bn_eps_c, bn_momentum_c)                                                        \
    TENSOR(conv_a_##id##_w, outChannel_a, kernel_size_a, kernel_size_a, inChannel_a);   \
    TENSOR(conv_a_##id##_b, outChannel_a);                                              \
    vector<size_t> conv_a_##id##_kernels{kernel_size_a, kernel_size_a};                 \
    vector<size_t> conv_a_##id##_strides{stride_size_a, stride_size_a};                 \
    vector<size_t> conv_a_##id##_pads{pad_size_a, pad_size_a, pad_size_a, pad_size_a};  \
    DYOP(conv_a_##id, Conv2dOp,                                                         \
            conv_a_##id##_kernels, conv_a_##id##_strides, conv_a_##id##_pads);          \
    LINKUPPER(conv_a_##id, upperNode, conv_a_##id##_w, conv_a_##id##_b);                \
    TENSOR(conv_a_##id##_out, 0);                                                       \
    LINKUPPER(conv_a_##id##_out, conv_a_##id);                                          \
    TENSOR(bn_a_##id##_scale, 0);                                                       \
    TENSOR(bn_a_##id##_shift, 0);                                                       \
    TENSOR(bn_a_##id##_running_mean, 0);                                                \
    TENSOR(bn_a_##id##_running_var, 0);                                                 \
    DYOP(bn_a_##id, BatchNormalizationOp, bn_eps_a, bn_momentum_a);                     \
    LINKUPPER(bn_a_##id, conv_a_##id##_out, bn_a_##id##_scale, bn_a_##id##_shift,       \
            bn_a_##id##_running_mean, bn_a_##id##_running_var);                         \
    TENSOR(bn_a_##id##_out, 0);                                                         \
    LINKUPPER(bn_a_##id##_out, bn_a_##id);                                              \
    DYOP(relu_a_##id, ReluOp);                                                          \
    LINKUPPER(relu_a_##id, bn_a_##id##_out);                                            \
    TENSOR(relu_a_##id##_out, 0);                                                       \
    LINKUPPER(relu_a_##id##_out, relu_a_##id);                                          \
    TENSOR(conv_b_##id##_w, outChannel_b, kernel_size_b, kernel_size_b, inChannel_b);   \
    TENSOR(conv_b_##id##_b, outChannel_b);                                              \
    vector<size_t> conv_b_##id##_kernels{kernel_size_b, kernel_size_b};                 \
    vector<size_t> conv_b_##id##_strides{stride_size_b, stride_size_b};                 \
    vector<size_t> conv_b_##id##_pads{pad_size_b, pad_size_b, pad_size_b, pad_size_b};  \
    DYOP(conv_b_##id, Conv2dOp,                                                         \
            conv_b_##id##_kernels, conv_b_##id##_strides, conv_b_##id##_pads);          \
    LINKUPPER(conv_b_##id, relu_a_##id##_out, conv_b_##id##_w, conv_b_##id##_b);        \
    TENSOR(conv_b_##id##_out, 0);                                                       \
    LINKUPPER(conv_b_##id##_out, conv_b_##id);                                          \
    TENSOR(bn_b_##id##_scale, 0);                                                       \
    TENSOR(bn_b_##id##_shift, 0);                                                       \
    TENSOR(bn_b_##id##_running_mean, 0);                                                \
    TENSOR(bn_b_##id##_running_var, 0);                                                 \
    DYOP(bn_b_##id, BatchNormalizationOp, bn_eps_b, bn_momentum_b);                     \
    LINKUPPER(bn_b_##id, conv_b_##id##_out, bn_b_##id##_scale, bn_b_##id##_shift,       \
            bn_b_##id##_running_mean, bn_b_##id##_running_var);                         \
    TENSOR(bn_b_##id##_out, 0);                                                         \
    LINKUPPER(bn_b_##id##_out, bn_b_##id);                                              \
    DYOP(relu_b_##id, ReluOp);                                                          \
    LINKUPPER(relu_b_##id, bn_b_##id##_out);                                            \
    TENSOR(relu_b_##id##_out, 0);                                                       \
    LINKUPPER(relu_b_##id##_out, relu_b_##id);                                          \
    TENSOR(conv_c_##id##_w, outChannel_c, kernel_size_c, kernel_size_c, inChannel_c);   \
    TENSOR(conv_c_##id##_b, outChannel_c);                                              \
    vector<size_t> conv_c_##id##_kernels{kernel_size_c, kernel_size_c};                 \
    vector<size_t> conv_c_##id##_strides{stride_size_c, stride_size_c};                 \
    vector<size_t> conv_c_##id##_pads{pad_size_c, pad_size_c, pad_size_c, pad_size_c};  \
    DYOP(conv_c_##id, Conv2dOp,                                                         \
            conv_c_##id##_kernels, conv_c_##id##_strides, conv_c_##id##_pads);          \
    LINKUPPER(conv_c_##id, relu_b_##id##_out, conv_c_##id##_w, conv_c_##id##_b);        \
    TENSOR(conv_c_##id##_out, 0);                                                       \
    LINKUPPER(conv_c_##id##_out, conv_c_##id);                                          \
    TENSOR(bn_c_##id##_scale, 0);                                                       \
    TENSOR(bn_c_##id##_shift, 0);                                                       \
    TENSOR(bn_c_##id##_running_mean, 0);                                                \
    TENSOR(bn_c_##id##_running_var, 0);                                                 \
    DYOP(bn_c_##id, BatchNormalizationOp, bn_eps_c, bn_momentum_c);                     \
    LINKUPPER(bn_c_##id, conv_c_##id##_out, bn_c_##id##_scale, bn_c_##id##_shift,       \
            bn_c_##id##_running_mean, bn_c_##id##_running_var);                         \
    TENSOR(bn_c_##id##_out, 0);                                                         \
    LINKUPPER(bn_c_##id##_out, bn_c_##id);                                              \
    

#define GpB(resnet, id)                                                                 \
    GpT(resnet, conv_a_##id##_w, conv_a_##id##_b, conv_a_##id##_out,                    \
            bn_a_##id##_scale, bn_a_##id##_shift,                                       \
            bn_a_##id##_running_mean, bn_a_##id##_running_var,                          \
            bn_a_##id##_out, relu_a_##id##_out,                                         \
            conv_b_##id##_w, conv_b_##id##_b, conv_b_##id##_out,                        \
            bn_b_##id##_scale, bn_b_##id##_shift,                                       \
            bn_b_##id##_running_mean, bn_b_##id##_running_var,                          \
            bn_b_##id##_out, relu_b_##id##_out,                                         \
            conv_c_##id##_w, conv_c_##id##_b, conv_c_##id##_out,                        \
            bn_c_##id##_scale, bn_c_##id##_shift,                                       \
            bn_c_##id##_running_mean, bn_c_##id##_running_var,                          \
            bn_c_##id##_out,                                                            \
            conv_s_##id##_w, conv_s_##id##_b, conv_s_##id##_out,                        \
            bn_s_##id##_scale, bn_s_##id##_shift,                                       \
            bn_s_##id##_running_mean, bn_s_##id##_running_var,                          \
            bn_s_##id##_out);                                                           \
    GpO(resnet, conv_a_##id, bn_a_##id, relu_a_##id,                                    \
            conv_b_##id, bn_b_##id, relu_b_##id,                                        \
            conv_c_##id, bn_c_##id,                                                     \
            conv_s_##id, bn_s_##id);                                                    \


int main() {

    //==========================
    // Example of Resnet-50 

    SWLOG_INFO << "Start generating graph..." << endl;

    // data
    TENSOR(data, 1, 224, 224, 3);
    INIT(data, TensorInitType::FILE, "mnist_images_8.bin");
    
    //conv1
    TENSOR(conv1_w, 64, 7, 7, 3);
    TENSOR(conv1_b, 64);
    vector<size_t> conv1_kernels{7, 7};
    vector<size_t> conv1_strides{2, 2};
    vector<size_t> conv1_pads{3, 3, 3, 3};
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
    //pool1
    vector<size_t> pool1_kernels{3, 3};
    vector<size_t> pool1_strides{2, 2};
    vector<size_t> pool1_pads{0, 0, 0, 0};
    DYOP(pool1, MaxPoolOp, pool1_kernels, pool1_strides, pool1_pads);
    LINKUPPER(pool1, relu1_out);
    TENSOR(pool1_out, 0);
    LINKUPPER(pool1_out, pool1);

    //resblock2a 
    resblock(2, pool1_out, 
            1, 1, 0, 64, 3, 0.0001, 0.9, 
            3, 1, 1, 64, 64, 0.0001, 0.9,
            1, 1, 0, 256, 64, 0.0001, 0.9);
    resblock_shortcut(2, pool1_out,
            1, 1, 0, 256, 64, 0.0001, 0.9);

    DYOP(res2a, ElementAddOp);
    LINKUPPER(res2a, bn_s_2_out, bn_c_2_out);
    TENSOR(res2a_out, 0);
    LINKUPPER(res2a_out, res2a);
    DYOP(res2a_relu, ReluOp);
    LINKUPPER(res2a_relu, res2a_out);
    TENSOR(res2a_relu_out, 0);
    LINKUPPER(res2a_relu_out, res2a_relu);



    G(resnet50);
    GpT(resnet50, data, conv1_w, conv1_b,
        conv1_out, bn1_out, 
        bn1_scale, bn1_shift, bn1_running_var, bn1_running_mean,
        relu1_out, pool1_out);
    GpO(resnet50, conv1, bn1, relu1, pool1);

    GpB(resnet50, 2);

    GpT(resnet50, res2a_out, res2a_relu_out);
    GpO(resnet50, res2a, res2a_relu);



    resnet50->initTensorNodes();

    resnet50->findInOut();
    resnet50->updateTopology();

    TRAIN(resnet50, "sgd", 0.001, 0.001, 0.9, 8);

    dotGen(resnet50_train);

}
