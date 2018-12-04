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
private:
    int _nInputTensor;
    int _nOutputTensor;
    std::shared_ptr<std::vector<std::shared_ptr<Tensor<Dtype> > > > _inputTensors;
    std::shared_ptr<std::vector<std::shared_ptr<Tensor<Dtype> > > > _outputTensors;

public:
    Op();
    Op(std::shared_ptr<std::vector<std::shared_ptr<Tensor<Dtype> > > > inputTensors,
            std::shared_ptr<std::vector<std::shared_ptr<Tensor<Dtype> > > > outputTensors);
    ~Op(){};

};

}

#endif
