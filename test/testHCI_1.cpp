#include <iostream>

#include "SWC.h"

#define Dtype float

using namespace swc;
using namespace std;

int main() {

    /* =========================== */
    /*   Example of  1 FC layer:   */
    /*    T:data_0   T:weight_0    */
    /*         \        /          */
    /*          \      /           */
    /*           O:FC_0            */
    /*             |               */
    /*          T:data_1           */
    /*             |               */
    /*          O:Tanh_0           */
    /*             |               */
    /*          T:data_2           */
    /* =========================== */

    // define IR graph
    G(MLPLayer);    

    Str2Graph<Dtype>(MLPLayer, "TENSOR(Weight_0, 1000, 1000, 1000)");
    Str2Graph<Dtype>(MLPLayer, "TENSOR(Data_0,   1000, 1000, 1000)");
    Str2Graph<Dtype>(MLPLayer, "TENSOR(Data_1,   1000, 1000      )");
    Str2Graph<Dtype>(MLPLayer, "TENSOR(Data_2,   1000, 1000      )");

    Str2Graph<Dtype>(MLPLayer, "OP(FC_0,   MatrixMatrixFCOp)"      );
    // Str2Graph<Dtype>(MLPLayer, "OP(FC_1,   MatrixMatrixFCOp)"      );
    Str2Graph<Dtype>(MLPLayer, "OP(Tanh_0, MatrixTanhOp)"          );

    Str2Graph<Dtype>(MLPLayer, "LINKUPPER(FC_0,    Data_0,  Weight_0)");
    Str2Graph<Dtype>(MLPLayer, "LINKUPPER(Data_1,  FC_0)  ");
    Str2Graph<Dtype>(MLPLayer, "LINKUPPER(Tanh_0,  Data_1)");
    Str2Graph<Dtype>(MLPLayer, "LINKUPPER(Data_2,  Tanh_0)");

    while(1) {

        string pattern;
        cout << "Please input pattern: " << endl;
        getline(cin, pattern);

        if (pattern == "exit") {
            return 0;
        } else if (pattern == "show") {
            dotGen(MLPLayer);
        } else 
            Str2Graph<Dtype>(MLPLayer, pattern);
    }

    return 0;
}