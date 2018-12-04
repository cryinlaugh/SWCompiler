/*************************************************************************
	> File Name: Op.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:16 2018
 ************************************************************************/

#include "Op.h"

namespace swc{

template<typename Dtype>
Op<Dtype>::Op(){
    _nInputTensor = 0;
    _nOutputTensor = 0;
    _inputTensors = NULL;
    _outputTensors = NULL;
}

template<typename Dtype>
Op<Dtype>::Op(
        std::shared_ptr<std::vector<std::shared_ptr<Tensor<Dtype> > > > inputTensors, 
        std::shared_ptr<std::vector<std::shared_ptr<Tensor<Dtype> > > > outputTensors){
    _inputTensors = inputTensors;
    _outputTensors = outputTensors;
    _nInputTensor = (*inputTensors).size();
    _nOutputTensor = (*outputTensors).size();
}

}
