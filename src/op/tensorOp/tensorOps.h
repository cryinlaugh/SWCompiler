/*************************************************************************
	> File Name: tensorOps.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Fri 04 Jan 2019 04:09:06 AM UTC
 ************************************************************************/

#ifndef _TENSOROPS_H
#define _TENSOROPS_H

#include <cassert>

#include "op/Op.h"

namespace swc {
namespace op {

class TensorDescendOp : public Op {
  public:
    TensorDescendOp(int nDim, int start, int end) : 
        Op(TENSOR_OP, 1, 1, std::string("TensorDescend")) {
        assert((start <= end) && "start can not be larger than end");
        _start = start;
        _end = end;
        _nDim = nDim;
        this->_inputNDims.push_back(nDim);
        this->_outputNDims.push_back(nDim-(end-start));
        this->_einOp = 1;
    }
    ~TensorDescendOp() {}

    void autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap);


    void destroy(){};

  private:
    int _nDim;
    int _start;
    int _end;
};

class TensorAscendOp : public Op {
  public:
    TensorAscendOp(int nDim, int start, int end) : 
        Op(TENSOR_OP, 1, 1, std::string("TensorAscend")) {
        assert((start <= end) && "start can not be larger than end");
        _start = start;
        _end = end;
        _nDim = nDim;
        this->_inputNDims.push_back(nDim-(end-start));
        this->_outputNDims.push_back(nDim);
        this->_einOp = 1;
    }
    ~TensorAscendOp() {}

    void autoDiff(IRGraph* graph, 
        IRNode* opNode,
        std::unordered_map<IRNode*, IRNode*>&gradNodeMap);
    
    void destroy(){};
  
  private:
    int _nDim;
    int _start;
    int _end;
};

//=====================================================
// Definition of 2-D tensor operations.
// Version v0.1: basic ops for tensors listed below
//-- 1 to N Tensor operations:
//----Duplicate
//----Split
//-- N to 1 Tensor operations:
//----Concat
//-- 1 to 1 Tensor operations:
//----Transpose
//----Descend
//=====================================================

class MatrixDuplicateOp : public Op {
  public:
    MatrixDuplicateOp(int nOutput)
        : Op(TENSOR_OP, 1, nOutput, std::string("MatrixDuplicate")) {
        this->_inputNDims.push_back(2);
        for (int i = 0; i < nOutput; i++) {
            this->_outputNDims.push_back(2);
        }
    }
    ~MatrixDuplicateOp() {}
    void destroy(){};
};

class MatrixSplitOp : public Op {
  public:
    MatrixSplitOp(int nOutput)
        : Op(TENSOR_OP, 1, nOutput, std::string("MatrixSplit")) {
        this->_inputNDims.push_back(2);
        for (int i = 0; i < nOutput; i++) {
            this->_outputNDims.push_back(2);
        }
    }
    ~MatrixSplitOp() {}
    void destroy(){};
};

class MatrixConcatOp : public Op {
  public:
    MatrixConcatOp(int nInput)
        : Op(TENSOR_OP, nInput, 1, std::string("MatrixConcat")) {
        for (int i = 0; i < nInput; i++) {
            this->_inputNDims.push_back(2);
        }
        this->_outputNDims.push_back(2);
    }
    ~MatrixConcatOp() {}
    void destroy(){};
};

class MatrixTransposeOp : public Op {
  public:
    MatrixTransposeOp() : Op(TENSOR_OP, 1, 1, std::string("MatrixTranspose")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);

        this->_einOp = 1;
        this->_einRep.push_back("ij");
        this->_einRep.push_back("ji");
    }
    ~MatrixTransposeOp() {}
    void destroy(){};
};


class MatrixDescendOp : public Op {
  public:
    MatrixDescendOp() : Op(TENSOR_OP, 1, 1, std::string("MatrixDescend")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(1);
    }
    ~MatrixDescendOp() {}
    void destroy(){};
};

//=====================================================
// Definition of 1-D tensor operations.
// Version v0.1: basic ops for tensors listed below
//-- 1 to N Tensor operations:
//----Duplicate
//----Split
//-- N to 1 Tensor operations:
//----Concat
//-- 1 to 1 Tensor operations:
//----Ascend
//----Descend
//=====================================================

class VectorDuplicateOp : public Op {
  public:
    VectorDuplicateOp(int nOutput)
        : Op(TENSOR_OP, 1, nOutput, std::string("VectorDuplicate")) {
        this->_inputNDims.push_back(1);
        for (int i = 0; i < nOutput; i++) {
            this->_outputNDims.push_back(1);
        }
    }
    ~VectorDuplicateOp() {}
    void destroy(){};
};

class VectorSplitOp : public Op {
  public:
    VectorSplitOp(int nOutput)
        : Op(TENSOR_OP, 1, nOutput, std::string("VectorSplit")) {
        this->_inputNDims.push_back(1);
        for (int i = 0; i < nOutput; i++) {
            this->_outputNDims.push_back(1);
        }
    }
    ~VectorSplitOp() {}
    void destroy(){};
};

class VectorConcatOp : public Op {
  public:
    VectorConcatOp(int nInput)
        : Op(TENSOR_OP, nInput, 1, std::string("VectorConcat")) {
        for (int i = 0; i < nInput; i++) {
            this->_inputNDims.push_back(1);
        }
        this->_outputNDims.push_back(1);
    }
    ~VectorConcatOp() {}
    void destroy(){};
};

class VectorAscendOp : public Op {
  public:
    VectorAscendOp() : Op(TENSOR_OP, 1, 1, std::string("VectorAscend")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(2);
    }
    ~VectorAscendOp() {}
    void destroy(){};
};

class VectorDescendOp : public Op {
  public:
    VectorDescendOp() : Op(TENSOR_OP, 1, 1, std::string("VectorDescend")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(0);
    }
    ~VectorDescendOp() {}
    void destroy(){};
};

//=====================================================
// Definition of 0-D tensor operations.
// Version v0.1: basic ops for tensors listed below
//-- 1 to N Tensor operations:
//----Duplicate
//-- 1 to 1 Tensor operations:
//----Ascend
//=====================================================

class ScalarDuplicateOp : public Op {
  public:
    ScalarDuplicateOp(int nOutput)
        : Op(TENSOR_OP, 1, nOutput, std::string("ScalarDuplicate")) {
        this->_inputNDims.push_back(0);
        for (int i = 0; i < nOutput; i++) {
            this->_outputNDims.push_back(0);
        }
    }
    ~ScalarDuplicateOp() {}
    void destroy(){};
};

class ScalarAscendOp : public Op {
  public:
    ScalarAscendOp() : Op(TENSOR_OP, 1, 1, std::string("ScalarAscend")) {
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(1);
    }
    ~ScalarAscendOp() {}
    void destroy(){};
};
} // namespace op
} // namespace swc

#endif
