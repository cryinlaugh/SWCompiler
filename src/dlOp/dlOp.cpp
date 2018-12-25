/*************************************************************************
	> File Name: dlOp.cpp
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:35 2018
 ************************************************************************/

#include "dlOp.h"

namespace swc {



INSTANTIATE_CLASS(MatrixMatrixFCOp);
INSTANTIATE_CLASS(MatrixTanhOp);
INSTANTIATE_CLASS(MatrixSoftmaxOp);
INSTANTIATE_CLASS(MatrixLogNegLossOp);


INSTANTIATE_CLASS(VectorTanhOp);
INSTANTIATE_CLASS(VectorSoftmaxOp);
INSTANTIATE_CLASS(VectorLogNegLossOp);


INSTANTIATE_CLASS(ScalarTanhOp);

} //namespace swc
