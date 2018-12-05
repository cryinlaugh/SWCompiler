/*************************************************************************
	> File Name: Op.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:16 2018
 ************************************************************************/

#include "Op.h"

namespace swc{

template<typename Dtype>
Op<Dtype>::Op(int nInput, int nOutput){
    _nInput = nInput;
    _nOutput = nOutput;
    _nInputTensor = 0;
    _nOutputTensor = 0;
}

template<typename Dtype>
void Op<Dtype>::addInputTensor(Tensor<Dtype>* inputTensor){
    _inputTensors.push_back(inputTensor);
    _nInputTensor++;
}

template<typename Dtype>
void Op<Dtype>::addOutputTensor(Tensor<Dtype>* outputTensor){
    _outputTensors.push_back(outputTensor);
    _nOutputTensor++;
}

template<typename Dtype>
bool Op<Dtype>::check(){
    if(_nInputTensor != _nInput) return false;
    if(_nOutputTensor != _nOutput) return false;
    for(int i=0; i<_nInput; i++){
        if(_inputTensors[i]->getNDim() != _inputNDims[i]) return false;
    }
    for(int i=0; i<_nOutput; i++){
        if(_outputTensors[i]->getNDim() != _inputNDims[i]) return false;
    }
    return true;
}
}
