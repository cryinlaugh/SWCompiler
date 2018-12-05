#include <iostream>

#include "SWDSL.h"

#define Dtype float

int main()
{
    //============================
    // Example of 1 FC layer:
    //  T:data_0   T:weight_0
    //     \       /
    //      \     /
    //        O:FC_0
    //         |
    //      T:data_1    
    //         |
    //      O:Tanh_0
    //         |
    //      T:data_2
    //=============================

  TENSOR(data, 0, Data_0, 1000 , 1000)
  CHECKT(data, 0)

  TENSOR(weight, 0, Weight_0, 1000, 1000)
  CHECKT(weight, 0)

  OP(fc, 0, FC_0, MatrixMatrixFCOp)
  CHECKO(fc, 0)

  TENSOR(data, 1, Data_1, 1000 , 1000)
  CHECKT(data, 1)
  
  OP(tanh, 0, Tanh_1, MatrixTanhOp)
  CHECKO(tanh, 0)
  
  TENSOR(data, 2, Data_2, 1000 , 1000)
  CHECKT(data, 2)

  LINKUPPER(O(fc, 0), T(data, 0), T(weight, 0))
  LINKUPPER(T(data, 1), O(fc, 0))
  LINKUPPER(O(tanh, 0), T(data, 1))
  LINKUPPER(T(data, 2), O(tanh, 0))

  //define IR graph
  G(MLPLayer)

  GpT(MLPLayer, T(data, 0), T(data, 1), T(data, 2), T(weight, 0))
  GpO(MLPLayer, O(fc, 0), O(tanh, 0))

  checkG(MLPLayer)

  return 0;
}
