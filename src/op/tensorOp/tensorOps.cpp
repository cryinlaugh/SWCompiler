/*************************************************************************
	> File Name: tensorOps.cpp
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Fri 04 Jan 2019 06:22:33 AM UTC
 ************************************************************************/
#include "tensorOps.h"

namespace swc{


INSTANTIATE_CLASS(MatrixDuplicateOp);
INSTANTIATE_CLASS(MatrixSplitOp);
INSTANTIATE_CLASS(MatrixConcatOp);
INSTANTIATE_CLASS(MatrixTransposeOp);
INSTANTIATE_CLASS(MatrixDescendOp);

INSTANTIATE_CLASS(VectorDuplicateOp);
INSTANTIATE_CLASS(VectorSplitOp);
INSTANTIATE_CLASS(VectorConcatOp);
INSTANTIATE_CLASS(VectorAscendOp);
INSTANTIATE_CLASS(VectorDescendOp);

INSTANTIATE_CLASS(ScalarDuplicateOp);
INSTANTIATE_CLASS(ScalarAscendOp);
}
