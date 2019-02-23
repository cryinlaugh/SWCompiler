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

        // t_node's info for test 
        // string W_1_Info = "TENSOR(Weight_1, 2333, 2333,  666)";
        // vector<string> w_1 = str_split(str_w1, ",");

        // w_1.push_back("TENSOR");
        // w_1.push_back("Weight_1");
        // w_1.push_back("666");
        // w_1.push_back("666");
        // w_1.push_back("666");

        // vector<unsigned long>* t_dim = new vector<unsigned long>({ 666, 666, 666 });
        // TensorNode<Dtype>*     Weight_1  = create_TensorNode<Dtype>(w_1, t_dim);
                                  
        // op_node's info for test 
        // string FC_1_Info = "OP(FC_1, MatrixMatrixFCOp)";
        // vector<string> op_1 = str_split(str_op1, ",");

        // op_1.push_back("OP");
        // op_1.push_back("FC_1");
        // op_1.push_back("MatrixMatrixFCOp");

        // OpNode<Dtype>* FC_1 = create_OpNode<Dtype>(op_1);

        // define IR graph
        G(MLPLayer);

        // IRNodeBuff[] for test 
        std::vector<IRNode*> IRNodeBuff;
        // IRNodeBuff.push_back(FC_1);
        // IRNodeBuff.push_back(Weight_1);
        // IRNodeBuff.push_back(create_OpNode<Dtype>(op_1));
        // IRNodeBuff.push_back(create_TensorNode<Dtype>(w_1, t_dim));
        Str2Graph<Dtype>(IRNodeBuff, "TENSOR(Weight_1, 2333, 2333,  666)");
        Str2Graph<Dtype>(IRNodeBuff, "OP(FC_1, MatrixMatrixFCOp)"        );
        Str2Graph<Dtype>(IRNodeBuff, "TENSOR(Weight_2, 666, 2333,  666)" );

        // // check the info of IRNodeBuff[]
        // cout << "Check for IRNodeBuff." << endl;
        // cout << "Node's Name: " << IRNodeBuff[0]->name() << "     | NodeType: " << IRNodeBuff[0]->nodeType() << endl;
        // cout << "Node's Name: " << IRNodeBuff[1]->name() <<     " | NodeType: " << IRNodeBuff[1]->nodeType() << endl << endl;

        // test link_upper 
        string link_Info = "LINKUPPER(FC_1,  Weight_0,   Weight_1, Weight_2)";
        Str2Graph<Dtype>(IRNodeBuff, link_Info);

        // get Op/TensorNode* from IRNodeBuff<IRNode*>[]
        TensorNode<Dtype>* t_w1   = (TensorNode<Dtype>*)IRNodeBuff[0];
        OpNode<Dtype>*     op_fc1 =     (OpNode<Dtype>*)IRNodeBuff[1];
        TensorNode<Dtype>* t_w2   = (TensorNode<Dtype>*)IRNodeBuff[2];

        GpT(MLPLayer, t_w1);
        GpO(MLPLayer, op_fc1);
        GpT(MLPLayer, t_w2);

        // CHECKT(t_w1);
        // CHECKO(op_fc1);

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

    LINKUPPER(FC_0, Data_0, Weight_0);

        // LINKUPPER(FC_1, Weight_1, Weight_0);

        LINKUPPER(op_fc1, Weight_0);
        LINKUPPER(Data_1, op_fc1);

    LINKUPPER(Data_1, FC_0);
    LINKUPPER(Tanh_0, Data_1);
    LINKUPPER(Data_2, Tanh_0);
    
    // Update graph topo info
    MLPLayer->findInOut();
    MLPLayer->updateTopoNodeList();
    MLPLayer->updateTopology();    

    dotGen(MLPLayer);
    // CHECKG(MLPLayer);

    cout << endl << "Please press enter to continue!" << endl;
    while(1) {
        if (cin.get() == '\n') 
            break;
    }

    /* ———————— add new nodes ———————— */

    TENSOR(Result, 1000);

    OP(Softmax, MatrixSoftmaxOp);

    LINKUPPER(Softmax, Data_2, Data_1);

        // // test link_upper 
        // string link_Info_1 = "LINKUPPER(Softmax, Data_2, Data_1)";
        // Str2Graph<Dtype>(IRNodeBuff, link_Info_1);

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

    dotGen(MLPLayer);

    // // Make Logs
    // SWLOG_INFO << "this is LOG" << endl;

    return 0;
}
