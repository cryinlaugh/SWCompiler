/*************************************************************************
	> File Name: dlOp.h
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:29 2018
 ************************************************************************/

#ifndef _DLOP_H
#define _DLOP_H

#include "op/basicOp/basicOps.h"

namespace swc {
namespace op {

//=====================================================
// Definition of 2-D deep learning specific operations.
// Version v0.1: ops for simple-MLP listed below
//--MatrixMatrixFC
//--Activation:
//----MatrixTanh
//--MatrixSoftmax
//--Loss:
//----MatrixLogNegLoss
//=====================================================

class MatrixMatrixFCOp : public Op {
    // input, weight, bias
    // output
  public:
    MatrixMatrixFCOp() : Op(DL_OP, 3, 1, std::string("MatrixMatrixFC")) {
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(2);
    }
    ~MatrixMatrixFCOp() {}
    void destroy(){};

    // for lowering
    void lowering(IRGraph *graph, IRNode *node);
};

class MatrixMatrixFCGradOp : public Op {
    // input, wieght, bias, orig_output, orig_output_grad
    // input_grad, weight_grad, bias_grad
  public:
    MatrixMatrixFCGradOp()
        : Op(DL_OP, 5, 3, std::string("MatrixMatrixFCGrad")) {}
    ~MatrixMatrixFCGradOp() {}
    void destroy() {}

    // for lowering
    void lowering(IRGraph *graph, IRNode *node);
};

class MatrixTanhOp : public Op {
  public:
    MatrixTanhOp() : Op(DL_OP, 1, 1, std::string("MatrixTanh")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixTanhOp();
    void destroy(){};
};

class MatrixTanhGradOp : public Op {
  public:
    MatrixTanhGradOp() : Op(DL_OP, 2, 1, std::string("MatrixTanhGrad")) {}
    ~MatrixTanhGradOp();
    void destroy() {}
};

class MatrixSoftmaxOp : public Op {
  public:
    MatrixSoftmaxOp() : Op(DL_OP, 2, 1, std::string("MatrixSoftmax")) {
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(2);
    };
    ~MatrixSoftmaxOp();
    void destroy(){};
};

class MatrixSoftmaxGradOp : public Op {
  public:
    MatrixSoftmaxGradOp() : Op(DL_OP, 3, 1, std::string("MatrixSoftmaxGrad")){};
    ~MatrixSoftmaxGradOp();
    void destroy() {}
};

class SGDOp : public Op {
    // weight weight_grad momentum
    // weight
    float lr_{0.001};
    float decay_{0.001};
    float momentum_{0.9};
    size_t batch_{1};

  public:
    SGDOp() : Op(DL_OP, 3, 1, std::string("SGD")) {}
    SGDOp(float lr, float decay, float momentum, size_t batch)
        : Op(DL_OP, 2, 1, std::string("SGD")), lr_(lr), decay_(decay),
          momentum_(momentum), batch_(batch) {}
    ~SGDOp();
    float getLR() { return lr_; }
    float getDecay() { return decay_; }
    float getMomentum() { return momentum_; }
    size_t getBatch() { return batch_; }
    void destroy() {}
};

class MatrixLogNegLossOp : public Op {
  public:
    MatrixLogNegLossOp() : Op(DL_OP, 1, 1, std::string("MatrixLogNegLoss")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(0);
    };
    ~MatrixLogNegLossOp();
    void destroy(){};
};

class MatrixTransOp : public Op {
  public:
    MatrixTransOp() : Op(DL_OP, 1, 1, std::string("MatrixTrans")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixTransOp();
    void destroy(){};
};

class MatrixAddOp : public Op {
  public:
    MatrixAddOp() : Op(DL_OP, 2, 1, std::string("MatrixAdd")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixAddOp();
    void destroy(){};
};

class ElementAddOp : public Op {
  public:
    ElementAddOp() : Op(DL_OP, 2, 1, std::string("ElementAdd")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~ElementAddOp();
    void destroy(){};
};

class ElementSubOp : public Op {
  public:
    ElementSubOp() : Op(DL_OP, 2, 1, std::string("ElementAdd")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~ElementSubOp();
    void destroy(){};
};

class ElementMulOp : public Op {
  public:
    ElementMulOp() : Op(DL_OP, 2, 1, std::string("ElementAdd")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~ElementMulOp();
    void destroy(){};
};

class ElementDivOp : public Op {
  public:
    ElementDivOp() : Op(DL_OP, 2, 1, std::string("ElementAdd")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~ElementDivOp();
    void destroy(){};
};

class PrintMatrixOp : public Op {
    PrintStreamType type_;
    std::string outfile_;

  public:
    PrintMatrixOp() : Op(DL_OP, 1, 0, std::string("PrintMatrix")) {
        this->_inputNDims.push_back(2);
    };
    ~PrintMatrixOp();
    void destroy(){};
    void setPrintStream(PrintStreamType type, std::string file = "") {
        type_ = type;
        outfile_ = file;
    }
    PrintStreamType getPrintStreamType() { return type_; }
    std::string getOutFile() { return outfile_; }
};

class ScatterOp : public Op {
    size_t offset_;

  public:
    ScatterOp() : Op(DL_OP, 0, 0, "Scatter"), offset_(0) {}
    ScatterOp(size_t offset) : Op(DL_OP, 0, 0, "Scatter"), offset_(offset) {}
    ~ScatterOp();

    void setOffset(size_t offset) { offset_ = offset; }
    size_t getOffset() { return offset_; }
};

class GatherOp : public Op {
    size_t offset_;

  public:
    GatherOp() : Op(DL_OP, 0, 0, "Gather"), offset_(0) {}
    GatherOp(size_t offset) : Op(DL_OP, 0, 0, "Gather"), offset_(offset) {}
    ~GatherOp();
    void setOffset(size_t offset) { offset_ = offset; }
    size_t getOffset() { return offset_; }
};

class SubGraphOp : public Op {
    IRGraph *graph_;

  public:
    SubGraphOp() : Op(DL_OP, 0, 0, "SubGraph") {}
    ~SubGraphOp();
    void setGraph(IRGraph *graph) { graph_ = graph; }
    IRGraph *getGraph() { return graph_; }
};

class Conv2dOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;
    int group_{1};

  public:
    Conv2dOp() : Op(DL_OP, 3, 1, std::string("Conv2d")) {
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(4);
    };
    Conv2dOp(std::vector<size_t> &kernels, std::vector<size_t> &strides,
             std::vector<size_t> &pads)
        : Op(DL_OP, 3, 1, std::string("Conv2d")) {
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
    void destroy() {}
};

class BatchNormalizationOp : public Op {
    float epsilon_;

  public:
    BatchNormalizationOp(float eps)
        : Op(DL_OP, 5, 1, std::string("BatchNormalization")) {
        epsilon_ = eps;
        // TODO : dims of input
    }
    float getEpsilon() { return epsilon_; }
    ~BatchNormalizationOp();
    void destroy() {}
};

class ReluOp : public Op {
  public:
    ReluOp() : Op(DL_OP, 1, 1, std::string("Relu")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
    }
    ~ReluOp();
    void destroy() {}
};

class MaxPoolOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;

  public:
    MaxPoolOp() : Op(DL_OP, 1, 1, std::string("MaxPool")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
    }
    MaxPoolOp(std::vector<size_t> &kernels, std::vector<size_t> &strides,
              std::vector<size_t> &pads)
        : Op(DL_OP, 1, 1, std::string("MaxPool")) {
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
    void destroy() {}
};

class AvgPoolOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;

  public:
    AvgPoolOp() : Op(DL_OP, 1, 1, std::string("AveragePool")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
    }
    AvgPoolOp(std::vector<size_t> &kernels, std::vector<size_t> &strides,
              std::vector<size_t> &pads)
        : Op(DL_OP, 1, 1, std::string("AveragePool")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
    }
    ~AvgPoolOp();
    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    void destroy() {}
};

class BatchedAddOp : public Op {
  public:
    BatchedAddOp() : Op(DL_OP, 2, 1, std::string("BatchedAdd")) {
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(2);
    }
    ~BatchedAddOp();
    void destroy() {}
};

class BatchedReduceAddOp : public Op {
  public:
    BatchedReduceAddOp() : Op(DL_OP, 1, 1, std::string("BatchedReduceAdd")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(1);
    }
    ~BatchedReduceAddOp();
    void destroy() {}
};

class TransposeOp : public Op {
    std::vector<size_t> shuffle_;

  public:
    TransposeOp(const std::initializer_list<size_t> &shuffle)
        : Op(DL_OP, 1, 1, std::string("Transpose")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
        for (auto i : shuffle)
            shuffle_.push_back(i);
    }
    ~TransposeOp();
    std::vector<size_t> getShuffle() { return shuffle_; }
    void destroy() {}
};
//=====================================================
// Definition of 1-D deep learning specific operations.
// Version v0.1: ops for simple-MLP-nobias-fw listed below
//--Activation:
//----VectorTanh
//--VectorSoftmax
//--Loss:
//----VectorLogNegLoss
//=====================================================

class VectorTanhOp : public Op {
  public:
    VectorTanhOp() : Op(DL_OP, 1, 1, std::string("VectorTanh")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(1);
    };
    ~VectorTanhOp();
    void destroy(){};
};

class VectorSoftmaxOp : public Op {
  public:
    VectorSoftmaxOp() : Op(DL_OP, 1, 1, std::string("VectorSoftmax")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(1);
    };
    ~VectorSoftmaxOp();
    void destroy(){};
};

class VectorLogNegLossOp : public Op {
  public:
    VectorLogNegLossOp() : Op(DL_OP, 1, 1, std::string("VectorLogNegLoss")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(0);
    };
    ~VectorLogNegLossOp();
    void destroy(){};
};

//=====================================================
// Definition of 0-D deep learning specific operations.
// Version v0.1: ops for simple-MLP-nobias-fw listed below
//--Activation:
//----ScalarTanh
//=====================================================

class ScalarTanhOp : public Op {
  public:
    ScalarTanhOp() : Op(DL_OP, 1, 1, std::string("ScalarTanh")) {
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    };
    ~ScalarTanhOp();
    void destroy(){};
};
} // namespace op
} // namespace swc

#endif
