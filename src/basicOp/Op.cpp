/*************************************************************************
	> File Name: Op.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:16 2018
 ************************************************************************/

#include "Op.h"

namespace swc{

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

INSTANTIATE_CLASS(Op);

}
