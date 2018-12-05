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



//=====================================================
//Definition of 2-D basic operations.
//Version v0.1: basic ops for simple-MLP-nobias-fw listed below
//--Tensor operations:
//----Not implemented.
//--Math Operations:
//----MatrixMatrixMul 
//----VectorMatrixMul
//----MatrixVectorMul
//=====================================================

template <typename Dtype>
class MatrixMatrixMulOp : public Op<Dtype>{
public:
    MatrixMatrixMulOp():Op<Dtype>(2, 1){
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    }
    ~MatrixMatrixMulOp(){}
};

template <typename Dtype>
class VectorMatrixMulOp : public Op<Dtype>{
public:
    VectorMatrixMulOp():Op<Dtype>(2, 1){
        this->_inputNDims.push_back(1);
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(1);
    }
    ~VectorMatrixMulOp(){}
};

template <typename Dtype>
class MatrixVectorMulOp : public Op<Dtype>{
public:
    MatrixVectorMulOp():Op<Dtype>(2, 1){
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(1);
    }
    ~MatrixVectorMulOp(){}
};

//=====================================================
//Definition of 1-D basic operations.
//Version v0.1: basic ops for simple-MLP listed below
//--Tensor operations:
//----Not implemented.
//--Math Operations:
//----VectorVectorInnerProduct 
//=====================================================

template <typename Dtype>
class VectorVectorInnerProductOp : public Op<Dtype>{
public:
    VectorVectorInnerProductOp():Op<Dtype>(2, 1){
        this->_inputNDims.push_back(1);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(0);
    }
    ~VectorVectorInnerProductOp(){}
};

//=====================================================
//Definition of 0-D basic operations.
//Version v0.1: basic ops for simple-MLP listed below
//--Tensor operations:
//----Not implemented.
//--Math Operations:
//----ScalarMul
//----ScalarAdd
//----ScalarExp
//----ScalarNeg
//----ScalarDiv
//----ScalarLog
//=====================================================


template <typename Dtype>
class ScalarMulOp : public Op<Dtype>{
public:
    ScalarMulOp():Op<Dtype>(2, 1){
        this->_inputNDims.push_back(0);
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarMulOp(){}
};

template <typename Dtype>
class ScalarAddOp : public Op<Dtype>{
public:
    ScalarAddOp():Op<Dtype>(2, 1){
        this->_inputNDims.push_back(0);
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarAddOp(){}
};

template <typename Dtype>
class ScalarMaxOp : public Op<Dtype>{
public:
    ScalarMaxOp():Op<Dtype>(2, 1){
        this->_inputNDims.push_back(0);
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarMaxOp(){}
};

template <typename Dtype>
class ScalarExpOp : public Op<Dtype>{
public:
    ScalarExpOp():Op<Dtype>(1, 1){
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarExpOp(){}
};

template <typename Dtype>
class ScalarNegOp : public Op<Dtype>{
public:
    ScalarNegOp():Op<Dtype>(1, 1){
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarNegOp(){}
};

template <typename Dtype>
class ScalarDivOp : public Op<Dtype>{
public:
    ScalarDivOp():Op<Dtype>(1, 1){
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarDivOp(){}
};

template <typename Dtype>
class ScalarLogOp : public Op<Dtype>{
public:
    ScalarLogOp():Op<Dtype>(1, 1){
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarLogOp(){}
};
}

#endif

