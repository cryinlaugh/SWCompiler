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

    // test add_IRNodes to Graph
    Str2Graph<Dtype>(MLPLayer, "TENSOR(Weight_0, 1000, 1000, 1000)");
    Str2Graph<Dtype>(MLPLayer, "TENSOR(Weight_1, 2333, 2333,  666)");
    Str2Graph<Dtype>(MLPLayer, "TENSOR(Weight_2, 666, 2333,  666)" );

    Str2Graph<Dtype>(MLPLayer, "OP(FC_0,   MatrixMatrixFCOp)"      );
    Str2Graph<Dtype>(MLPLayer, "OP(FC_1,   MatrixMatrixFCOp)"      );
    Str2Graph<Dtype>(MLPLayer, "OP(Tanh_0, MatrixTanhOp)"          );

    Str2Graph<Dtype>(MLPLayer, "TENSOR(Data_0,   1000, 1000, 1000)");
    Str2Graph<Dtype>(MLPLayer, "TENSOR(Data_1,   1000, 1000      )");
    Str2Graph<Dtype>(MLPLayer, "TENSOR(Data_2,   1000, 1000      )");
        // // Update graph topo info
        // MLPLayer->findInOut();
        // MLPLayer->updateTopoNodeList();
        // MLPLayer->updateTopology();

    // get Op/TensorNode from graph by nodeName
    OpNode<Dtype>*     op_fc1  =     (OpNode<Dtype>*)getIRNodeByName_Topo(MLPLayer, "FC_1");
    TensorNode<Dtype>* t_Data1 = (TensorNode<Dtype>*)getIRNodeByName_Topo(MLPLayer, "Data_1");

    CHECKO(op_fc1);
    CHECKT(t_Data1);

    // // define IRNode
    // TENSOR(Weight_1, 2333, 2333,  666);
    // TENSOR(Data_0,   1000, 1000, 1000);
    // TENSOR(Weight_0, 1000, 1000, 1000);
    // TENSOR(Data_1,   1000, 1000      );
    // TENSOR(Data_2,   1000, 1000      );

    // OP(FC_0,   MatrixMatrixFCOp);
    // OP(FC_1,   MatrixMatrixFCOp);
    // OP(Tanh_0, MatrixTanhOp);

    // // Add IRNodes into the Graph
    // GpT(MLPLayer, Data_0, Data_1, Data_2, Weight_0);
    // GpO(MLPLayer, FC_0,   Tanh_0);
    
    cout << endl;

    // LINKUPPER(FC_0, Data_0);
    Str2Graph<Dtype>(MLPLayer, "LINKUPPER(FC_0,   Data_0)");

    // LINKUPPER(Data_1, op_fc1);
    // Str2Graph<Dtype>(MLPLayer, "LINKUPPER(Data_1, FC_1, FC_0)");

    // LINKUPPER(Data_1, FC_0);
    // LINKUPPER(Tanh_0, Data_1);
    // LINKUPPER(Data_2, Tanh_0);

    // Str2Graph<Dtype>(MLPLayer, "LINKUPPER(Data_1, FC_0)");
    Str2Graph<Dtype>(MLPLayer, "LINKUPPER(Tanh_0, Data_1)");
    Str2Graph<Dtype>(MLPLayer, "LINKUPPER(Data_2, Tanh_0)");
    
    // // Update graph topo info
    // MLPLayer->findInOut();
    // MLPLayer->updateTopoNodeList();
    // MLPLayer->updateTopology();    

    // test link_upper_G 
    Str2Graph<Dtype>(MLPLayer, "LINKUPPER(Data_1, FC_1,     FC_0)");
    Str2Graph<Dtype>(MLPLayer, "LINKUPPER(FC_1,   Weight_0, Weight_1, Weight_2)");
        // string link_Info = "LINKUPPER(FC_1,  Weight_0,   Weight_1, Weight_2)";
        // drop_mark(link_Info, " ");  // drop " " is very very important
        // vector<string> inputInfo_test = str_split(link_Info, ",");
        // Link_Upper_G<Dtype>(inputInfo_test, MLPLayer);
        
    dotGen(MLPLayer);

    return 0;
}
