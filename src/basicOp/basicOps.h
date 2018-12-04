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
class MatrixMatrixFCOp : public Op{
public:
    MatrixMatrixFCOp():nInput(2), nOutput(1) {};
    MatrixMatrixFCOp(std::shared_ptr<std::vector<std::shared_ptr<Tensor<Dtype> > > > inputTensors,
            std::shared_ptr<std::vector<std::shared_ptr<Tensor<Dtype> > > > outputTensors) 
        : Op(inputTensors, outputTensors),
            nInput(2), nOutput(1) {};
    ~MatrixMatrixFCOp();

private:
    const int nInput;
    const int nOutput;

};

}

#endif

