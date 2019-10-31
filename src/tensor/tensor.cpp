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
        if (idx < shape_->size())
            shape->push_back(shape_->at(idx));
    }

    return new TensorShape(shape);
}

TensorShape* TensorShape::getTiledShape(int index, int n) {
    int ndim = getNDim();
    std::vector<size_t>* tileVector = new std::vector<size_t>();
    for(int i = 0; i < ndim; i++) {
        if(i == index)
            tileVector->push_back(getDim(i)/n);
        else
            tileVector->push_back(getDim(i));
    }
    TensorShape* result= new TensorShape(tileVector);
    return result;
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
    std::vector<size_t> *shape = new std::vector<unsigned long>();
    for (auto idx : shuffle) {
        if ((int)idx < shape_->getNDim())
            shape->push_back(shape_->getDim(idx));
    }

    return new TensorShape(shape);
}
std::vector<size_t> 
    Tensor::getShuffledDims(const std::vector<size_t> &shuffle) const {
    std::vector<size_t> dims;
    for (auto idx : shuffle) {
        if ((int)idx < shape_->getNDim())
            dims.push_back(shape_->getDim(idx));
    }
    return dims;
}
std::pair<size_t, size_t> Tensor::viewAs2D(int n) {
    assert(n>=1 && n<=getNDim() && "illegal n");
    size_t dim0 = getDim(0);
    int i;
    for(i=1; i<n; i++)
        dim0 *= getDim(i);
    size_t dim1 = 1;
    for(; i<getNDim(); i++)
        dim1 *= getDim(i);
    return {dim0, dim1};
}

std::string Tensor::getMemLayoutTag() const{

    int ndim = getNDim();
    switch(mem_layout_){
    case layout_default:
        if(ndim == 1)
            return "x"; 
        if(ndim == 2)
            return "nc"; 
        break;
    case layout_nhwc:
        return "nhwc";
        break;
    case layout_nchw:
        return "nchw";
        break;
    case layout_nc:
        return "nc";
        break;
    case layout_cn:
        return "cn";
        break;
    default:
        return "any"; 
    }
    return "any"; 
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
