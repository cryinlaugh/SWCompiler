/*************************************************************************
	> File Name: basicOps.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 04 Dec 2018 12:05:58 PM UTC
 ************************************************************************/

#ifndef _BASICOPS_H
#define _BASICOPS_H

#include "op/Op.h"
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
    MatrixMatrixMulOp():Op<Dtype>(BASIC_OP, 2, 1, std::string("MatrixMatrixMul")){
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    }
    ~MatrixMatrixMulOp(){}
    void destroy(){};
};

template <typename Dtype>
class VectorMatrixMulOp : public Op<Dtype>{
public:
    VectorMatrixMulOp():Op<Dtype>(BASIC_OP, 2, 1, std::string("VectorMatrixMul")){
        this->_inputNDims.push_back(1);
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(1);
    }
    ~VectorMatrixMulOp(){}
    void destroy(){};
};

template <typename Dtype>
class MatrixVectorMulOp : public Op<Dtype>{
public:
    MatrixVectorMulOp():Op<Dtype>(BASIC_OP, 2, 1, std::string("MatrixVectorMul")){
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(1);
    }
    ~MatrixVectorMulOp(){}
    void destroy(){};
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
    VectorVectorInnerProductOp():Op<Dtype>(BASIC_OP, 2, 1, std::string("VectorVectorInnerProduct")){
        this->_inputNDims.push_back(1);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(0);
    }
    ~VectorVectorInnerProductOp(){}
    void destroy(){};
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
    ScalarMulOp():Op<Dtype>(BASIC_OP, 2, 1, std::string("ScalarMul")){
        this->_inputNDims.push_back(0);
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarMulOp(){}
    void destroy(){};
};

template <typename Dtype>
class ScalarAddOp : public Op<Dtype>{
public:
    ScalarAddOp():Op<Dtype>(BASIC_OP, 2, 1, std::string("ScalarAdd")){
        this->_inputNDims.push_back(0);
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarAddOp(){}
    void destroy(){};
};

template <typename Dtype>
class ScalarMaxOp : public Op<Dtype>{
public:
    ScalarMaxOp():Op<Dtype>(BASIC_OP, 2, 1, std::string("ScalarMax")){
        this->_inputNDims.push_back(0);
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarMaxOp(){}
    void destroy(){};
};

template <typename Dtype>
class ScalarExpOp : public Op<Dtype>{
public:
    ScalarExpOp():Op<Dtype>(BASIC_OP, 1, 1, std::string("ScalarExp")){
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarExpOp(){}
    void destroy(){};
};

template <typename Dtype>
class ScalarNegOp : public Op<Dtype>{
public:
    ScalarNegOp():Op<Dtype>(BASIC_OP, 1, 1, std::string("ScalarNeg")){
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarNegOp(){}
    void destroy(){};
};

template <typename Dtype>
class ScalarDivOp : public Op<Dtype>{
public:
    ScalarDivOp():Op<Dtype>(BASIC_OP, 1, 1, std::string("ScalarDiv")){
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarDivOp(){}
    void destroy(){};
};

template <typename Dtype>
class ScalarLogOp : public Op<Dtype>{
public:
    ScalarLogOp():Op<Dtype>(BASIC_OP, 1, 1, std::string("ScalarLog")){
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    }
    ~ScalarLogOp(){}
    void destroy(){};
};
}

#endif

