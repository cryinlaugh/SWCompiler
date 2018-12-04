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

enum TensorType{
    2D,
    1D,
    0D,
    UNKNOWN
};

class TensorShape{
private:
    int _ndim;
    shared_ptr<std::vector<unsigned long> > _shape;
public:
    TensorShape(unsigned ndim, shared_ptr<std::vector<unsigned> > shape);
    ~TensorShape(){};
    const int getNDim() const;
    const unsigned long getDim(int idx) const;
};


template <typename Dtype>
class Tensor{
private:
    TensorType _type;
    shared_ptr<TensorShape> _shape;
    shared_ptr<SWMem<Dtype> > _data;

public:
    Tensor();
    Tensor(TensorType t, shared_ptr<TensorShape> shape, shared_ptr<SWMem<Dtype> > tdata);
    ~Tensor(){}; 

    const int getNDim() const;
    const unsigned long getDim(int dim) const;
};

}

#endif
