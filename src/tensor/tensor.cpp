/*************************************************************************
	> File Name: tensor.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:56:42 2018
 ************************************************************************/

#include "tensor.h"

namespace swc{

TensorShape::TensorShape(std::vector<unsigned long>* shape){
    _ndim = shape->size();
    _shape = shape;
}

const int TensorShape::getNDim() const{
    return _ndim; 
}

const int unsigned long TensorShape::getDim(int idx) const{
    return (*_shape)[idx];
}

}

