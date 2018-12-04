/*************************************************************************
	> File Name: tensor.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:56:42 2018
 ************************************************************************/

#include "tensor.h"

namespace swc{

TensorShape::TensorShape(unsigned ndim, std::shared_ptr<std::vector<unsigned long> > shape){
    _ndim = ndim;
    _shape = shape;
}

const int TensorShape::getNDim() const{
    return _ndim; 
}

const int unsigned long TensorShape::getDim(int idx) const{
    return (*_shape)[idx];
}

template <typename Dtype>
Tensor<Dtype>::Tensor(){
    _type = UNKNOWN;
    _shape = NULL;
    _data = NULL;
}

template <typename Dtype>
Tensor<Dtype>::Tensor(TensorType t, std::shared_ptr<TensorShape> shape, std::shared_ptr<SWMem<Dtype> > tdata){
    _type = t;
    _shape = shape;
    _data = tdata; 
}


template <typename Dtype>
const int Tensor<Dtype>::getNDim() const{
    return _shape->getNDim();
}

template <typename Dtype>
const unsigned long Tensor<Dtype>::getDim(int dim) const{
    return _shape->getDim(dim);
}

}

