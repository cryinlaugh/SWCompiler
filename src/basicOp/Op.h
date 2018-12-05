/*************************************************************************
	> File Name: Op.h
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:08 2018
 ************************************************************************/

#ifndef _OP_H
#define _OP_H

#include "../common.h"
#include "../tensor/tensor.h"

namespace swc{

template <typename Dtype>
class Op{
public: 
    //The following variables are constant values in a specific Op Class
    //indicating what kind of input/output tensors it should keep.
    const int _nInput;
    const int _nOutput;
    std::vector<int> _inputNDims;
    std::vector<int> _outputNDims;

    //The following variables indicating the real input/output tensors 
    //that the Op really have.
    int _nInputTensor;
    int _nOutputTensor;
    std::vector<Tensor<Dtype>* > _inputTensors;
    std::vector<Tensor<Dtype>* > _outputTensors;

    Op(int nInput = 0, int nOutput = 0);
    ~Op(){};
    void addInputTensor(Tensor<Dtype>* inputTensor);
    void addOutputTensor(Tensor<Dtype>* outputTensor);
    bool check();
};

}

#endif
