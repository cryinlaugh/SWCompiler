/*************************************************************************
	> File Name: Op.h
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:08 2018
 ************************************************************************/

#ifndef _OP_H
#define _OP_H

#include <string>

#include "common.h"
#include "SWLOG.h"

namespace swc {

//Forward declarations
template<typename Dtype> class Tensor;
template<typename Dtype> class IRGraph;
class IRNode;

template<typename Dtype>
class Op {
  public:

    Op(OpType opType = BASIC_OP, 
       int nInput = 0, 
       int nOutput = 0, 
       std::string opClassName = NULL)
       : _opType(opType), 
       _nInput(nInput), 
       _nOutput(nOutput), 
       _opClassName(opClassName) { 

        _nInputTensor  = 0;
        _nOutputTensor = 0;
    };

    ~Op(){};

    virtual void destroy(){};

    void addInputTensor(Tensor<Dtype>* inputTensor) {
        _inputTensors.push_back(inputTensor);
        _nInputTensor++;
    }

    void addOutputTensor(Tensor<Dtype>* outputTensor) { 
        _outputTensors.push_back(outputTensor);
        _nOutputTensor++;
    }
    bool check(); 

    OpType getOpType() { return _opType; }

    const std::string getOpName() { return _opClassName; }

    inline const int getnInput()  { return _nInput;  }
    inline const int getnOutput() { return _nOutput; }

    //for lowering
    virtual void lowering(IRGraph<Dtype>* graph, IRNode* node){
        SWLOG_INFO<< "Unimplemented in base Op class" << std::endl;
    }
  
  protected: 

    /* The following variables are constant values in a specific Op Class
       indicating what kind of input/output tensors it should keep.            */

    const OpType      _opType;          // enum var, define the type of operation
    const int         _nInput;          // nums of input  tensor
    const int         _nOutput;         // nums of output tensor
    std::vector<int>  _inputNDims;      // input  tensors
    std::vector<int>  _outputNDims;     // output tensors

    const std::string _opClassName;

    /* The following variables indicating the real input/output tensors 
       that the Op really have, its useful in analyses or ref-code-generation. */

    int                           _nInputTensor;
    int                           _nOutputTensor;
    std::vector<Tensor<Dtype>* >  _inputTensors;
    std::vector<Tensor<Dtype>* >  _outputTensors;

};

}

#endif
