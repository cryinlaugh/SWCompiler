/*************************************************************************
	> File Name: Op.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:16 2018
 ************************************************************************/

#include "Op.h"

#include "tensor/tensor.h"

namespace swc{
bool Op::check(){
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
