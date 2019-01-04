/*************************************************************************
	> File Name: tensorOps.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Fri 04 Jan 2019 04:09:06 AM UTC
 ************************************************************************/

#ifndef _TENSOROPS_H
#define _TENSOROPS_H

#include "op/Op.h"
namespace swc{


//=====================================================
//Definition of 2-D tensor operations.
//Version v0.1: basic ops for tensors listed below
//-- 1 to N Tensor operations:
//----Duplicate
//----Split
//-- N to 1 Tensor operations:
//----Concat
//-- 1 to 1 Tensor operations:
//----Transpose
//----Descend
//=====================================================

template <typename Dtype>
class MatrixDuplicateOp : public Op<Dtype>{
public:
    MatrixDuplicateOp(int nOutput):Op<Dtype>(TENSOR_OP, 1, nOutput, std::string("MatrixDuplicate")){
        this->_inputNDims.push_back(2);
        for(int i=0; i<nOutput; i++){
            this->_outputNDims.push_back(2);
        }
    }
    ~MatrixDuplicateOp(){}
    void destroy(){};
};

template <typename Dtype>
class MatrixSplitOp : public Op<Dtype>{
public:
    MatrixSplitOp(int nOutput):Op<Dtype>(TENSOR_OP, 1, nOutput, std::string("MatrixSplit")){
        this->_inputNDims.push_back(2);
        for(int i=0; i<nOutput; i++){
            this->_outputNDims.push_back(2);
        }
    }
    ~MatrixSplitOp(){}
    void destroy(){};
};

template <typename Dtype>
class MatrixConcatOp : public Op<Dtype>{
public:
    MatrixConcatOp(int nInput):Op<Dtype>(TENSOR_OP, nInput, 1, std::string("MatrixConcat")){
        for(int i=0; i<nInput; i++){
            this->_inputNDims.push_back(2);
        }
        this->_outputNDims.push_back(2);
    }
    ~MatrixConcatOp(){}
    void destroy(){};
};

template <typename Dtype>
class MatrixTransposeOp : public Op<Dtype>{
public:
    MatrixTransposeOp():Op<Dtype>(TENSOR_OP, 1, 1, std::string("MatrixTranspose")){
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    }
    ~MatrixTransposeOp(){}
    void destroy(){};
};

template <typename Dtype>
class MatrixDescendOp : public Op<Dtype>{
public:
    MatrixDescendOp():Op<Dtype>(TENSOR_OP, 1, 1, std::string("MatrixDescend")){
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(1);
    }
    ~MatrixDescendOp(){}
    void destroy(){};
};

//=====================================================
//Definition of 1-D tensor operations.
//Version v0.1: basic ops for tensors listed below
//-- 1 to N Tensor operations:
//----Duplicate
//----Split
//-- N to 1 Tensor operations:
//----Concat
//-- 1 to 1 Tensor operations:
//----Ascend
//----Descend
//=====================================================
template <typename Dtype>
class VectorDuplicateOp : public Op<Dtype>{
public:
    VectorDuplicateOp(int nOutput):Op<Dtype>(TENSOR_OP, 1, nOutput, std::string("VectorDuplicate")){
        this->_inputNDims.push_back(1);
        for(int i=0; i<nOutput; i++){
            this->_outputNDims.push_back(1);
        }
    }
    ~VectorDuplicateOp(){}
    void destroy(){};
};

template <typename Dtype>
class VectorSplitOp : public Op<Dtype>{
public:
    VectorSplitOp(int nOutput):Op<Dtype>(TENSOR_OP, 1, nOutput, std::string("VectorSplit")){
        this->_inputNDims.push_back(1);
        for(int i=0; i<nOutput; i++){
            this->_outputNDims.push_back(1);
        }
    }
    ~VectorSplitOp(){}
    void destroy(){};
};

template <typename Dtype>
class VectorConcatOp : public Op<Dtype>{
public:
    VectorConcatOp(int nInput):Op<Dtype>(TENSOR_OP, nInput, 1, std::string("VectorConcat")){
        for(int i=0; i<nInput; i++){
            this->_inputNDims.push_back(1);
        }
        this->_outputNDims.push_back(1);
    }
    ~VectorConcatOp(){}
    void destroy(){};
};

template <typename Dtype>
class VectorAscendOp : public Op<Dtype>{
public:
    VectorAscendOp():Op<Dtype>(TENSOR_OP, 1, 1, std::string("VectorAscend")){
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(2);
    }
    ~VectorAscendOp(){}
    void destroy(){};
};

template <typename Dtype>
class VectorDescendOp : public Op<Dtype>{
public:
    VectorDescendOp():Op<Dtype>(TENSOR_OP, 1, 1, std::string("VectorDescend")){
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(0);
    }
    ~VectorDescendOp(){}
    void destroy(){};
};


//=====================================================
//Definition of 0-D tensor operations.
//Version v0.1: basic ops for tensors listed below
//-- 1 to N Tensor operations:
//----Duplicate
//-- 1 to 1 Tensor operations:
//----Ascend
//=====================================================

template <typename Dtype>
class ScalarDuplicateOp : public Op<Dtype>{
public:
    ScalarDuplicateOp(int nOutput):Op<Dtype>(TENSOR_OP, 1, nOutput, std::string("ScalarDuplicate")){
        this->_inputNDims.push_back(0);
        for(int i=0; i<nOutput; i++){
            this->_outputNDims.push_back(0);
        }
    }
    ~ScalarDuplicateOp(){}
    void destroy(){};
};

template <typename Dtype>
class ScalarAscendOp : public Op<Dtype>{
public:
    ScalarAscendOp():Op<Dtype>(TENSOR_OP, 1, 1, std::string("ScalarAscend")){
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(1);
    }
    ~ScalarAscendOp(){}
    void destroy(){};
};
}


#endif
