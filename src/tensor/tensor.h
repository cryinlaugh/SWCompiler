/*************************************************************************
	> File Name: tensor.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tues. 12/ 4 15:53:19 2018
 ************************************************************************/

#ifndef _TENSOR_H
#define _TENSOR_H

#include "../common.h"

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
};


template <typename Dtype>
class Tensor{
private:
    TensorType _type;
    TensorShape* _shape;
    std::shared_ptr<SWMem<Dtype> > _data;

public:
    Tensor(){ 
        _type = UNKNOWN;
        _shape = NULL;
        _data = NULL;
    }
    Tensor(TensorShape* shape){
        _type = TensorType(shape->getNDim());
        _shape = shape;
    }
    ~Tensor(){}; 

    const int getNDim() const{
        return _shape->getNDim();
    };
    const unsigned long getDim(int dim) const{
        return _shape->getDim(dim);
    };

    TensorShape* getTensorShape() const{
        return _shape;

    }
};

}

#endif
