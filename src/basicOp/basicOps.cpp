/*************************************************************************
	> File Name: basicOps.cpp
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 04 Dec 2018 12:06:08 PM UTC
 ************************************************************************/

#include "basicOps.h"

namespace swc{


INSTANTIATE_CLASS(MatrixMatrixMulOp);
INSTANTIATE_CLASS(VectorMatrixMulOp);
INSTANTIATE_CLASS(MatrixVectorMulOp);

INSTANTIATE_CLASS(VectorVectorInnerProductOp);

INSTANTIATE_CLASS(ScalarAddOp);
INSTANTIATE_CLASS(ScalarMaxOp);
INSTANTIATE_CLASS(ScalarExpOp);
INSTANTIATE_CLASS(ScalarNegOp);
INSTANTIATE_CLASS(ScalarDivOp);
INSTANTIATE_CLASS(ScalarLogOp);

}
