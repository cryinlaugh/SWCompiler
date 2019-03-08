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

INSTANTIATE_CLASS(Tensor);

}

