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

TensorShape *
TensorShape::getShuffledTensorShape(const std::vector<size_t> &shuffle) const {
    std::vector<size_t> *shape = new std::vector<size_t>();
    for (auto idx : shuffle) {
        if ((int)idx < shape_->size())
            shape->push_back(shape_->at(idx));
    }

    return new TensorShape(shape);
}


Tensor *Tensor::clone() const {
    Tensor *t = new Tensor(shape_, dataType_);
    t->setTraining(train_);
    t->setTensorInit(initType_, initInfo_);
    return t;
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

void Tensor::setTensorInit(TensorInitType type, std::string file,
                           size_t offset) {
    assert((type == TensorInitType::FILE) && "init type does not match value");
    initType_ = type;
    initInfo_.setFilePath(file);
    initInfo_.setOffset(offset);
}

void Tensor::setTensorInit(TensorInitType type, TensorInitInfo info) {
    initType_ = type;
    initInfo_ = info;
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
        return shape_->size() * sizeof(float);
    }
}

TensorShape *
Tensor::getShuffledTensorShape(const std::vector<size_t> &shuffle) const {
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
