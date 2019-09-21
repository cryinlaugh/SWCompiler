#include <iostream>

#include "SWC.h"
#include "string.h"
//#include "parallel/TilingLabel.h"
//#include "parallel/ParallelPattern.h"
//#include "pass/ParallelingPass.h"
using namespace swc;
using namespace swc::op;
using namespace std;

int main() {
    //============================
    // Example of 2-layer
    //
    //  T:data0   T:weight0
    //     \       /
    //      \     /
    //        O:fc0 -- T:bias0
    //         |
    //      T:data1
    //         |
    //      O:tanh0
    //         |
    //      T:data2
    //                  T:weight1
    //          \       /
    //           \     /
    //          O:fc1 -- T:bias1
    //              |
    //          T:data3
    //              |
    //          O: softmax
    //              |
    //          T:data4
    //=============================

    TENSOR(data0, 8, 784);
    TENSOR(weight0, 784, 512);
    TENSOR(bias0, 512);
    data0_Tensor->setTensorInit(TensorInitType::FILE,
                                "input/mnist_images_8.bin");
    weight0_Tensor->setTensorInit(TensorInitType::FILE,
                                  "input/mlp_weight0.bin");
    bias0_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias0.bin");

    OP(fc0, MatrixMatrixFCBiasOp);
    LINKUPPER(fc0, data0, weight0, bias0);

    TENSOR(data1, 8, 512);
    LINKUPPER(data1, fc0);

    OP(tanh0, MatrixTanhOp);
    LINKUPPER(tanh0, data1);

    TENSOR(data2, 8, 512);
    LINKUPPER(data2, tanh0);

    TENSOR(weight1, 512, 10);
    TENSOR(bias1, 10);
    weight1_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_weight1.bin");
    bias1_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias1.bin");

    OP(fc1, MatrixMatrixFCBiasOp);
    LINKUPPER(fc1, data2, weight1, bias1);

    TENSOR(data3, 8, 10);
    LINKUPPER(data3, fc1);

    OP(softmax0, MatrixSoftmaxOp);
    // OP(softmax0, MatrixSoftmaxWithLossOp);
    LINKUPPER(softmax0, data3);

    TENSOR(data4, 8, 10);
    LINKUPPER(data4, softmax0);

    G(mlp);
    GpT(mlp, data0, data1, data2, data3, data4, weight0, weight1, bias0, bias1);
    GpO(mlp, fc0, fc1, tanh0, softmax0);

    auto *argmax_o = new OpNode("argmax", new ArgMaxOp(3));
    argmax_o->exlinkUpperNode(data4);
    auto *top3_t =
        new TensorNode("top3", new Tensor({8, 3}, DataType::Int32_t), argmax_o);
    auto *print_o = new OpNode("print", new DebugOp());
    print_o->exlinkUpperNode(top3_t);

    TENSOR(placeholder, 0);
    LINKUPPER(placeholder, print_o);

    // define IR graph
    GpT(mlp, top3_t, placeholder);
    GpO(mlp, argmax_o, print_o);

    mlp->findInOut();
    mlp->updateTopology();

    SETOUT(mlp, data4);

    mlp->findInOut();
    mlp->updateTopology();

    //data0->pushParentNode();

    //StrategyLabel* slabel = new StrategyLabel();
    //slabel->setStrategy({0,-1,0});
    //fc0->setStrategyLabel(new StrategyLabel({0, -1, -1, 0}));
    //tanh0->setStrategyLabel(new StrategyLabel({0, 0}));
    //fc1->setStrategyLabel(new StrategyLabel({0, -1, -1, 0}));
    //softmax0->setStrategyLabel(new StrategyLabel({1, 1}));



    swc::pass::LabelingPass labelingpass(mlp);
    labelingpass.run();

    swc::pass::LoweringPass loweringpass(mlp);
    loweringpass.run();
    labelingpass.run();


    swc::pass::ParallelLabelingPass parallelLabelingpass(mlp);
    parallelLabelingpass.run();

   swc::pass::ParallelLoweringPass parallelLoweringpass(mlp);
   parallelLoweringpass.run();



    swc::pass::RenamingNodePass renamingpass(mlp);
    renamingpass.run();

    swc::pass::EliminationPass elim(mlp);
    elim.run();

    //std::vector<string>

    //pass::Optimizer *opt = new pass::Optimizer(mlp);

    //opt->runOptimizer();
    //mlp->updateTopology();
    /*
    auto *argmax_o = new OpNode("argmax", new ArgMaxOp(3));
    argmax_o->exlinkUpperNode(data4);
    auto *top3_t =
        new TensorNode("top3", new Tensor({8, 3}, DataType::Int32_t), argmax_o);
    auto *print_o = new OpNode("print", new DebugOp());
    print_o->exlinkUpperNode(top3_t);

    TENSOR(placeholder, 0);
    LINKUPPER(placeholder, print_o);

    // define IR graph
    GpT(mlp, top3_t, placeholder);
    GpO(mlp, argmax_o, print_o);

    mlp->findInOut();
    mlp->updateTopology();
    labelingpass.run();
    */

    dotGen(mlp);
    Config config;
    std::cout << "dotgen ok\n";
    codegen::ParallelCodegen *cg = new codegen::ParallelCodegen(mlp, config);
    string code = cg->generate();
    cout << code;


    return 0;
}
