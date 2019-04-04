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

const unsigned long TensorShape::getDim(int idx) const{
    return (*_shape)[idx];
}

const unsigned long TensorShape::size() const {
    unsigned long size = 1;
    for(auto dim : *_shape)
        size *= dim;
    return size;
}

template<typename Dtype>
void Tensor<Dtype>::setTensorInit(TensorInitType type, Dtype value) { 
    initType_ = type;
    switch(type){
        case TensorInitType::CONSTANT: { 
            initInfo_.setConstant(value);
            break;
        }
        case TensorInitType::XAVIER: { 
            initInfo_.setFilterSize(value);
            break;
        }
        default:
            break;
    }
}

template<typename Dtype>
void Tensor<Dtype>::setTensorInit(TensorInitType type, std::string file) { 
	initType_  = type; 
	initInfo_.setFilePath(file); 
}

template<>
size_t Tensor<float>::getSizeInBytes() const {
    return _shape->size() * sizeof(float);
}

template<>
size_t Tensor<double>::getSizeInBytes() const {
    return _shape->size() * sizeof(double);
}

template<>
size_t Tensor<int>::getSizeInBytes() const {
    return _shape->size() * sizeof(int);
}

 

INSTANTIATE_CLASS(Tensor);

}

