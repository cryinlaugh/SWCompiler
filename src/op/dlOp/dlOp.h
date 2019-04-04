/*************************************************************************
	> File Name: dlOp.h
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:29 2018
 ************************************************************************/

#ifndef _DLOP_H
#define _DLOP_H

#include "op/basicOp/basicOps.h"

namespace swc{

//=====================================================
//Definition of 2-D deep learning specific operations.
//Version v0.1: ops for simple-MLP listed below
//--MatrixMatrixFC
//--Activation:
//----MatrixTanh
//--MatrixSoftmax
//--Loss:
//----MatrixLogNegLoss 
//=====================================================


template <typename Dtype>
class MatrixMatrixFCOp : public Op<Dtype>{
public:
    MatrixMatrixFCOp():Op<Dtype>(DL_OP, 2, 1, std::string("MatrixMatrixFC")){
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    }
    ~MatrixMatrixFCOp(){}
    void destroy(){};

    //for lowering
    void lowering(IRGraph<Dtype>* graph, IRNode* node);

};

template <typename Dtype>
class MatrixTanhOp : public Op<Dtype>{
public:
    MatrixTanhOp():Op<Dtype>(DL_OP, 1,1, std::string("MatrixTanh")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixTanhOp();
    void destroy(){};
};

template <typename Dtype>
class MatrixSoftmaxOp : public Op<Dtype>{
public:
    MatrixSoftmaxOp(): Op<Dtype>(DL_OP, 1,1, std::string("MatrixSoftmax")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixSoftmaxOp();
    void destroy(){};

};

template <typename Dtype>
class MatrixLogNegLossOp : public Op<Dtype>{
public:
    MatrixLogNegLossOp():Op<Dtype>(DL_OP, 1,1, std::string("MatrixLogNegLoss")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(0);
    };
    ~MatrixLogNegLossOp();
    void destroy(){};
};

template <typename Dtype>
class MatrixTanhGradOp : public Op<Dtype>{
public:
    MatrixTanhGradOp():Op<Dtype>(DL_OP, 2,1, std::string("MatrixTanhGrad")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixTanhGradOp();
    void destroy(){};
};

template <typename Dtype>
class MatrixSoftmaxGradOp : public Op<Dtype>{
public:
    MatrixSoftmaxGradOp(): Op<Dtype>(DL_OP, 2,1, std::string("MatrixSoftmaxGrad")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixSoftmaxGradOp();
    void destroy(){};
};

template <typename Dtype>
class MatrixTransOp : public Op<Dtype>{
public:
    MatrixTransOp(): Op<Dtype>(DL_OP, 1,1, std::string("MatrixTrans")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixTransOp();
    void destroy(){};
};

template <typename Dtype>
class MatrixAddOp : public Op<Dtype>{
public:
    MatrixAddOp(): Op<Dtype>(DL_OP, 2,1, std::string("MatrixAdd")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixAddOp();
    void destroy(){};
};

template <typename Dtype>
class PrintMatrixOp : public Op<Dtype>{
    PrintStreamType type_;
    std::string outfile_;
public:
    PrintMatrixOp(): Op<Dtype>(DL_OP, 1,0, std::string("PrintMatrix")) {
        this->_inputNDims.push_back(2);
    };
    ~PrintMatrixOp();
    void destroy(){};
    void setPrintStream(PrintStreamType type, std::string file=""){
        type_ = type;
        outfile_ = file;
    }
    PrintStreamType getPrintStreamType() { return type_; }
    std::string getOutFile() { return outfile_; }
};

template <typename Dtype>
class ScatterOp: public Op<Dtype>{
    size_t offset_;
public:
    ScatterOp() : Op<Dtype>(DL_OP, 0, 0, "Scatter"), offset_(0){}
    ScatterOp(size_t offset) : Op<Dtype>(DL_OP, 0, 0, "Scatter"), offset_(offset){}
    ~ScatterOp();

    void setOffset(size_t offset) { offset_ = offset; }
    size_t getOffset() {return offset_; }
};

template <typename Dtype>
class GatherOp: public Op<Dtype>{
    size_t offset_;
public:
    GatherOp() : Op<Dtype>(DL_OP, 0, 0, "Gather"), offset_(0){}
    GatherOp(size_t offset) : Op<Dtype>(DL_OP, 0, 0, "Gather"), offset_(offset){}
    ~GatherOp();
    void setOffset(size_t offset) { offset_ = offset; }
    size_t getOffset() {return offset_; }
};

template <typename Dtype>
class SubGraphOp: public Op<Dtype>{
    IRGraph<Dtype>* graph_;
public:
    SubGraphOp() : Op<Dtype>(DL_OP, 0, 0, "SubGraph"){}
    ~SubGraphOp();
    void setGraph(IRGraph<Dtype>* graph) { graph_ = graph; }
    IRGraph<Dtype>* getGraph() {return graph_; }

};

template <typename Dtype>
class Conv2dOp : public Op<Dtype>{
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;
    int group_{1};
public:
    Conv2dOp(): Op<Dtype>(DL_OP, 3,1, std::string("Conv2d")) {
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(4);
    };
    Conv2dOp(std::vector<size_t> &kernels,
            std::vector<size_t> &strides,
            std::vector<size_t> &pads): Op<Dtype>(DL_OP, 3,1, std::string("Conv2d")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(4);
    }
    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    size_t getGroup() { return group_; }
    ~Conv2dOp();
    void destroy(){}
};

template <typename Dtype>
class ReluOp: public Op<Dtype>{
public:
    ReluOp(): Op<Dtype>(DL_OP, 1,1, std::string("Relu")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
    }
    ~ReluOp();
    void destroy(){}
};

template <typename Dtype>
class MaxPoolOp: public Op<Dtype>{
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;
public:
    MaxPoolOp(): Op<Dtype>(DL_OP, 1,1, std::string("MaxPool")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
    }
    MaxPoolOp(std::vector<size_t> &kernels,
            std::vector<size_t> &strides,
            std::vector<size_t> &pads): Op<Dtype>(DL_OP, 1,1, std::string("MaxPool")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
    }
    ~MaxPoolOp();
    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    void destroy(){}
};

template <typename Dtype>
class BatchedAddOp: public Op<Dtype>{
public:
    BatchedAddOp(): Op<Dtype>(DL_OP, 2,1, std::string("BatchedAdd")) {
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(2);
    }
    ~BatchedAddOp();
    void destroy(){}
};

template <typename Dtype>
class TranposeOp: public Op<Dtype>{
    std::vector<size_t> shuffle_;
public:
    TranposeOp(const std::initializer_list<size_t> &shuffle): Op<Dtype>(DL_OP, 1,1, std::string("Transpose")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
        for(auto i : shuffle)
            shuffle_.push_back(i);
    }
    ~TranposeOp();
    std::vector<size_t> getShuffle() { return shuffle_; }
    void destroy(){}
};
//=====================================================
//Definition of 1-D deep learning specific operations.
//Version v0.1: ops for simple-MLP-nobias-fw listed below
//--Activation:
//----VectorTanh
//--VectorSoftmax
//--Loss:
//----VectorLogNegLoss 
//=====================================================

template <typename Dtype>
class VectorTanhOp : public Op<Dtype>{
public:
    VectorTanhOp():Op<Dtype>(DL_OP, 1,1, std::string("VectorTanh")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(1);
    };
    ~VectorTanhOp();
    void destroy(){};
};

template <typename Dtype>
class VectorSoftmaxOp : public Op<Dtype>{
public:
    VectorSoftmaxOp(): Op<Dtype>(DL_OP, 1,1, std::string("VectorSoftmax")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(1);
    };
    ~VectorSoftmaxOp();
    void destroy(){};
};


template <typename Dtype>
class VectorLogNegLossOp : public Op<Dtype>{
public:
    VectorLogNegLossOp():Op<Dtype>(DL_OP, 1,1, std::string("VectorLogNegLoss")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(0);
    };
    ~VectorLogNegLossOp();
    void destroy(){};
};

//=====================================================
//Definition of 0-D deep learning specific operations.
//Version v0.1: ops for simple-MLP-nobias-fw listed below
//--Activation:
//----ScalarTanh
//=====================================================

template <typename Dtype>
class ScalarTanhOp : public Op<Dtype>{
public:
    ScalarTanhOp():Op<Dtype>(DL_OP, 1,1, std::string("ScalarTanh")) {
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    };
    ~ScalarTanhOp();
    void destroy(){};
};
}

#endif
