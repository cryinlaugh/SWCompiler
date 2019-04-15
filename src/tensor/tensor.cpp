/*************************************************************************
	> File Name: tensor.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:56:42 2018
 ************************************************************************/
#include "tensor.h"
#include "SWLOG.h"

namespace swc {

TensorShape::TensorShape(std::vector<size_t> *shape) {
    _ndim = shape->size();
    shape_ = shape;
}

int TensorShape::getNDim() const { return _ndim; }

size_t TensorShape::getDim(int idx) const { return (*shape_)[idx]; }

size_t TensorShape::size() const {
    size_t size = 1;
    for (auto dim : *shape_)
        size *= dim;
    return size;
}

void Tensor::setTensorInit(TensorInitType type, float value) {
    initType_ = type;
    switch (type) {
    case TensorInitType::CONSTANT: {
        initInfo_.setConstant(value);
        break;
    }
    case TensorInitType::ZERO: {
        initInfo_.setConstant(0);
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

void Tensor::setTensorInit(TensorInitType type, std::string file, size_t offset) {
    assert((type==TensorInitType::FILE) && "init type does not match value");
    initType_ = type;
    initInfo_.setFilePath(file);
    initInfo_.setOffset(offset);
}

size_t Tensor::getSizeInBytes() const {
    switch (dataType_) {
    case DataType::Float_t:
        return shape_->size() * sizeof(float);
    case DataType::Double_t:
        return shape_->size() * sizeof(double);
    case DataType::Int8_t:
        return shape_->size() * sizeof(int8_t);
    case DataType::Int32_t:
        return shape_->size() * sizeof(int32_t);
    default:
        SWLOG_ERROR << "UNKNOWN DataType\n";
    }
}

TensorShape* Tensor::getShuffledTensorShape(const std::vector<size_t> &shuffle) const{
    std::vector<unsigned long> *shape = new std::vector<unsigned long>();
    for (auto idx : shuffle) {
        if ((int)idx < shape_->getNDim())
            shape->push_back(shape_->getDim(idx));
    }

    return new TensorShape(shape);
}
/*

template<>
size_t Tensor<double>::getSizeInBytes() const {
    return shape_->size() * sizeof(double);
}

template<>
size_t Tensor<int>::getSizeInBytes() const {
    return shape_->size() * sizeof(int);
}
*/
} // namespace swc
