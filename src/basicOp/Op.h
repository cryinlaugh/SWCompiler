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

namespace swc {

template <typename Dtype>
class Op {

protected: 

    /* The following variables are constant values in a specific Op Class
       indicating what kind of input/output tensors it should keep.            */

    const OpType      _opType;          // enum var, define the type of operation
    const int         _nInput;          // nums of input  tensor
    const int         _nOutput;         // nums of output tensor
    std::vector<int>  _inputNDims;      // input  tensors
    std::vector<int>  _outputNDims;     // output tensors

    /* The following variables indicating the real input/output tensors 
       that the Op really have, its useful in analyses or ref-code-generation. */

    int                           _nInputTensor;
    int                           _nOutputTensor;
    std::vector<Tensor<Dtype>* >  _inputTensors;
    std::vector<Tensor<Dtype>* >  _outputTensors;

public:

    Op(OpType opType = BASIC_OP, int nInput = 0, int nOutput = 0) :
       _opType(opType), _nInput(nInput), _nOutput(nOutput) { 

        _nInputTensor  = 0;
        _nOutputTensor = 0;
    };

    ~Op(){};

    void addInputTensor(Tensor<Dtype>* inputTensor) {
        _inputTensors.push_back(inputTensor);
        _nInputTensor++;
    }

    void addOutputTensor(Tensor<Dtype>* outputTensor) { 
        _outputTensors.push_back(outputTensor);
        _nOutputTensor++;
    }

    bool check() {

        if (_nInputTensor  != _nInput)  return false;
        if (_nOutputTensor != _nOutput) return false;

        for (int i=0; i<_nInput; i++) {
            if (_inputTensors[i]->getNDim()  != _inputNDims[i]) 
                return false;
        }

        for (int i=0; i<_nOutput; i++) {
            if (_outputTensors[i]->getNDim() != _inputNDims[i]) 
                return false;
        }

        return true;
    };

    OpType getOpType() { return _opType; }

    inline const int getnInput()  { return _nInput;  }
    inline const int getnOutput() { return _nOutput; }
};

}

#endif
