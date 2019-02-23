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

        // test add_IRNodes
        std::vector<IRNode*> IRNodeBuff;
        Str2Graph_IRbuff<Dtype>(IRNodeBuff, "TENSOR(Weight_1, 2333, 2333,  666)");
        Str2Graph_IRbuff<Dtype>(IRNodeBuff, "OP(FC_1, MatrixMatrixFCOp)"        );
        Str2Graph_IRbuff<Dtype>(IRNodeBuff, "TENSOR(Weight_2, 666, 2333,  666)" );

        Str2Graph<Dtype>(MLPLayer, "TENSOR(Weight_1, 2333, 2333,  666)");
        Str2Graph<Dtype>(MLPLayer, "OP(FC_1, MatrixMatrixFCOp)"        );
        Str2Graph<Dtype>(MLPLayer, "TENSOR(Weight_2, 666, 2333,  666)" );

        // Update graph topo info
        MLPLayer->findInOut();
        MLPLayer->updateTopoNodeList();
        MLPLayer->updateTopology();
        // IRNode* testIR = getIRNodeByName_Topo(MLPLayer, "FC_1");
        // cout << endl;

        // // test link_upper 
        // string link_Info = "LINKUPPER(FC_1, Weight_1, Weight_2)";
        // Str2Graph_IRbuff<Dtype>(IRNodeBuff, link_Info);

        // get Op/TensorNode* from IRNodeBuff<IRNode*>[]
        // TensorNode<Dtype>* t_w1   = (TensorNode<Dtype>*)IRNodeBuff[0];
        OpNode<Dtype>* op_fc1 = (OpNode<Dtype>*)getIRNodeByName_Topo(MLPLayer, "FC_1");
        // TensorNode<Dtype>* t_w2   = (TensorNode<Dtype>*)IRNodeBuff[2];

        // GpT(MLPLayer, t_w1);
        // GpO(MLPLayer, op_fc1);
        // GpT(MLPLayer, t_w2);

        // // Update graph topo info
        // MLPLayer->findInOut();
        // MLPLayer->updateTopoNodeList();
        // MLPLayer->updateTopology(); 

    // define IRNode
    TENSOR(Weight_1, 2333, 2333,  666);
    TENSOR(Data_0,   1000, 1000, 1000);
    TENSOR(Weight_0, 1000, 1000, 1000);
    TENSOR(Data_1,   1000, 1000      );
    TENSOR(Data_2,   1000, 1000      );

    OP(FC_0,   MatrixMatrixFCOp);
    OP(FC_1,   MatrixMatrixFCOp);
    OP(Tanh_0, MatrixTanhOp);

    // Add IRNodes into the Graph
    GpT(MLPLayer, Data_0, Data_1, Data_2, Weight_0);
    GpO(MLPLayer, FC_0,   Tanh_0);

    LINKUPPER(FC_0, Data_0/*, Weight_0*/);

        // LINKUPPER(op_fc1, Weight_0);
        LINKUPPER(Data_1, op_fc1);

    LINKUPPER(Data_1, FC_0);
    LINKUPPER(Tanh_0, Data_1);
    LINKUPPER(Data_2, Tanh_0);
    
    // Update graph topo info
    MLPLayer->findInOut();
    MLPLayer->updateTopoNodeList();
    MLPLayer->updateTopology();    

        // test link_upper_G 
        string link_Info = "LINKUPPER(FC_1,  Weight_0,   Weight_1, Weight_2)";
        drop_mark(link_Info, " ");  // drop " " is very very important
        vector<string> inputInfo_test = str_split(link_Info, ",");
        Link_Upper_G<Dtype>(inputInfo_test, MLPLayer);
    
        // // test for find upperNodes
        // for (int up_num = 2; up_num < (int)inputInfo_test.size(); ++up_num) {

        //     for (int i = 0; i < MLPLayer->topologyNum(); i++) {        
        //         for (int j = 0; j < MLPLayer->getNumInTopoLevel(i); j++) {
        //             if (inputInfo_test[up_num] == MLPLayer->getNodeInTopo(i, j)->name()) 
        //                 cout << "Find upperNode: " << MLPLayer->getNodeInTopo(i, j)->name() << endl;
        //         }
        //     }
        // }
        
    dotGen(MLPLayer);

    // cout << endl << "Please press enter to continue!" << endl;
    // while(1) {
    //     if (cin.get() == '\n') 
    //         break;
    // }

    /* ———————— add new nodes ———————— */

    TENSOR(Result, 1000);

    OP(Softmax, MatrixSoftmaxOp);

    LINKUPPER(Softmax, Data_2, Data_1);

        // // test link_upper 
        // string link_Info_1 = "LINKUPPER(Softmax, Data_2, Data_1)";
        // Str2Graph_IRbuff<Dtype>(IRNodeBuff, link_Info_1);

    LINKUPPER(Result, Softmax);

    // Update the IR graph
    GpT(MLPLayer, Result);
    GpO(MLPLayer, Softmax);

    // CHECKT(Data_0);
    // CHECKT(Weight_0);
    // CHECKO(FC_0);
    // CHECKT(Data_1);
    // CHECKO(Tanh_0);
    // CHECKT(Data_2);
    // CHECKG(MLPLayer);

    // // lowering
    // Optimizer<Dtype>* opt = new Optimizer<Dtype>(MLPLayer);
    // opt->runOptimizer();

    // Update graph topo info
    MLPLayer->findInOut();
    MLPLayer->updateTopoNodeList();
    MLPLayer->updateTopology();

    // dotGen(MLPLayer);

    // // Make Logs
    // SWLOG_INFO << "this is LOG" << endl;

    return 0;
}
