/*************************************************************************
	> File Name: testMLP.cpp
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Wed 05 Dec 2018 03:34:34 AM UTC
 ************************************************************************/

#include "SWC.h"
#include <iostream>

using namespace swc;
using namespace std;

#define Dtype float

int main() {
    cout << "In test MLP main" << endl;
    //============================
    // Example of 1 FC layer:
    //  T:data_0   T:weight_0
    //     \       /
    //      \     /
    //        O:FC_0 -- T: bias_0
    //         |
    //      T:data_1
    //         |
    //      O:Tanh_1
    //         |
    //      T:data_2
    //=============================

    // define tensor nodes
    TensorNode *dataTensorNode_0 = new TensorNode("Data_0");
    // Init tensor nodes as following:
    //--init TensorShape:
    TensorShape *dataTensorShape_0 =
        new TensorShape(new vector<size_t>({1000, 1000}));
    //--init Tensor
    Tensor *dataTensor_0 = new Tensor(dataTensorShape_0);
    //--set tensor in tensor node
    dataTensorNode_0->setTensor(dataTensor_0);

    TensorNode *weightTensorNode_0 = new TensorNode("Weight_0", {1000, 1000});
    weightTensorNode_0->getLabel()->setTensorInitTypeLabel(
        TensorInitType::CONSTANT);

    TensorNode *biasTensorNode_0 = new TensorNode("Bias_0", {1000});

    // define op nodes
    OpNode *fcOpNode_0 = new OpNode("FC_0");
    // Init op nodes as following:
    //--init Op:
    MatrixMatrixFCOp *fcOp_0 = new MatrixMatrixFCOp();
    //--set Op in Op node
    fcOpNode_0->setOp(fcOp_0);

    // link upperNode from current node(Parent)
    // Relink upperNode to current node(Child)
    fcOpNode_0->pushParentNode(dataTensorNode_0, weightTensorNode_0,
                               biasTensorNode_0);
    dataTensorNode_0->pushChildNode(fcOpNode_0);
    weightTensorNode_0->pushChildNode(fcOpNode_0);
    biasTensorNode_0->pushChildNode(fcOpNode_0);

    TensorNode *dataTensorNode_1 =
        new TensorNode("Data_1", {1000, 1000}, fcOpNode_0);

    OpNode *tanhOpNode_1 = new OpNode("Tanh_1");
    MatrixTanhOp *tanhOp_1 = new MatrixTanhOp();
    tanhOpNode_1->setOp(tanhOp_1);

    tanhOpNode_1->pushParentNode(dataTensorNode_1);
    dataTensorNode_1->pushChildNode(tanhOpNode_1);

    TensorNode *dataTensorNode_2 =
        new TensorNode("Data_2", {1000, 1000}, tanhOpNode_1);

    // define IR graph
    IRGraph *MLPLayer = new IRGraph();
    MLPLayer->pushTensorNode(dataTensorNode_0, weightTensorNode_0,
                             biasTensorNode_0, dataTensorNode_1,
                             dataTensorNode_2);
    MLPLayer->pushOpNode(fcOpNode_0, tanhOpNode_1);

    MLPLayer->findInOut();
    MLPLayer->updateTopology();
    MLPLayer->updateTopoNodeList();

    printf("Generate MLP layer done!\n");

    MLPLayer->updateTopoNodeList();
    Optimizer *opt = new Optimizer(MLPLayer);
    opt->runOptimizer();
    dotGen(MLPLayer);

    codegen::Codegen *cg = new codegen::Codegen(MLPLayer);
    string code = cg->generate();
    cout << code;

    return 0;
}
