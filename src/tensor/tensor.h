/*************************************************************************
	> File Name: tensor.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tues. 12/ 4 15:53:19 2018
 ************************************************************************/

#ifndef _TENSOR_H
#define _TENSOR_H

#include "../common.h"
#include <string>

namespace swc{

class TensorShape{
private:
    int _ndim;
    std::vector<unsigned long>* _shape;
public:
    TensorShape(std::vector<unsigned long>* shape);
    ~TensorShape(){};
    const int getNDim() const;
    const unsigned long getDim(int idx) const;
    const unsigned long size() const;
};

template<typename Dtype>
class TensorInitInfo{
    std::string file_; // FILE initialization
    Dtype constant_; //constant initialization
    Dtype filterSize_; // xavier initialization
public:
    TensorInitInfo() : file_(""), constant_(0), filterSize_(3){}

    std::string getFilePath() const  { return file_; }
    Dtype getConstant() const  { return constant_; }
    Dtype getFilterSize() const  { return filterSize_; }

    void setFilePath(std::string f){ file_ = f; }
    void setConstant(Dtype c) { constant_ = c; }
    void setFilterSize(Dtype fsize) { filterSize_ = fsize; }
};

template <typename Dtype>
class Tensor{
private:
    TensorType _type;
    TensorShape* _shape;
    std::shared_ptr<SWMem<Dtype> > _data;

    TensorInitType initType_;
    TensorInitInfo<Dtype>  initInfo_;

public:
    Tensor(){
        _type = UNKNOWN;
        _shape = NULL;
        _data = NULL;

        initType_ = TensorInitType::NONE;
    }
    Tensor(TensorShape* shape){
        _type = TensorType(shape->getNDim());
        _shape = shape;

        initType_ = TensorInitType::NONE;
    }
    Tensor(const std::initializer_list<int> &shape){
        std::vector<unsigned long> *vec = new std::vector<unsigned long>();
        for(auto i : shape){
            int v = i;
            vec->push_back(v);
        }
        _shape = new TensorShape(vec);
        initType_ = TensorInitType::NONE;
    }
    ~Tensor(){};

    const int getNDim() const{
        return _shape->getNDim();
    };
    const unsigned long getDim(int dim) const{
        return _shape->getDim(dim);
    };
    const std::vector<unsigned long> getDims() const{
        std::vector<unsigned long> dims;
        for(int i=0; i<getNDim(); i++)
            dims.push_back(getDim(i));
        return dims;
    }

    TensorInitType getTensorInitType() { return initType_; }
    TensorInitInfo<Dtype> getTensorInitInfo() const { return initInfo_; }
    void reset(TensorShape* shape){ _shape = shape; }
    TensorShape* getTensorShape() const{ return _shape; }
    size_t size() { return _shape->size(); }
    size_t getSizeInBytes() const ;

    void setTensorInit(TensorInitType type, Dtype value);
    void setTensorInit(TensorInitType type, std::string file);
};

}

#endif
