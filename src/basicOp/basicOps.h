/*************************************************************************
	> File Name: basicOps.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 04 Dec 2018 12:05:58 PM UTC
 ************************************************************************/

#ifndef _BASICOPS_H
#define _BASICOPS_H

#include "Op.h"
namespace swc{

template <typename Dtype>
class MatrixMatrixFCOp : public Op<Dtype>{
public:
    MatrixMatrixFCOp():Op<Dtype>(2, 1){
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    }
    ~MatrixMatrixFCOp(){}

};

template <typename Dtype>
class MatrixTanhOp : public Op<Dtype>{
public:
    MatrixTanhOp():Op<Dtype>(1,1) {
        this->inputNDims.push_back(2);
        this->outputNDims.push_back(2);
    };
    ~MatrixTanhOp();
};

template <typename Dtype>
class MatrixSoftmaxOp : public Op<Dtype>{
public:
    MatrixSoftmaxOp(): Op<Dtype>(1,1) {
        this->inputNDims.push_back(2);
        this->outputNDims.push_back(2);
    };
    ~MatrixSoftmaxOp();

};

template <typename Dtype>
class MatrixLogNegLossOp : public Op<Dtype>{
public:
    MatrixLogNegLossOp():Op<Dtype>(1,1) {
        this->inputNDims.push_back(2);
        this->outputNDims.push_back(0);
    };
    ~MatrixLogNegLossOp();
};


}

#endif

