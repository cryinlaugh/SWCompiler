/*************************************************************************
    > File Name: testPolyMPI.cpp
    > Author: wayne
    > Mail:
    > Created Time: 一  7/22 11:02:13 2019
 ************************************************************************/

 #include <iostream>

 #include "SWC.h"

 using namespace swc;
 using namespace swc::op;
 using namespace std;

 int main() {
     //============================
     // Example of 2-layer
     // Fully Connected network:
     // data parallel, fc0 and tanh0
     // run on different MPI processes.
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

     //====================================================
     OP(cpux, ParallelSubGraphOp);

     LINKUPPER(cpux, data0, weight0, bias0);

     TENSOR(data2, 8, 512);
     LINKUPPER(data2, cpux);

     // define IR graph
     G(mlp);
     GpT(mlp, data0, data2, weight0, bias0);
     GpO(mlp, cpux);

     TENSOR(weight1, 512, 10);
     TENSOR(bias1, 10);
     weight1_Tensor->setTensorInit(TensorInitType::FILE,
                                    "input/mlp_weight1.bin");
     bias1_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias1.bin");

     OP(fc1, MatrixMatrixFCBiasOp);
     LINKUPPER(fc1, data2, weight1, bias1);

     TENSOR(data3, 8, 10);
     LINKUPPER(data3, fc1);

     Tensor *labelt = new Tensor({8}, DataType::Int32_t);
     TensorNode *labeln = new TensorNode("selected", labelt);

     OP(softmax, MatrixSoftmaxOp);
     LINKUPPER(softmax, data3, labeln);

     TENSOR(data4, 8, 10);
     LINKUPPER(data4, softmax);

     OpNode *argmax = new OpNode("argmax", new ArgMaxOp(3));
     argmax->exlinkUpperNode(data4);

     TensorNode *top3_idx =
         new TensorNode("top3", new Tensor({8, 3}, DataType::Int32_t), argmax);

     OpNode *print_top3 = new OpNode("print_top3", new DebugOp());
     print_top3->exlinkUpperNode(top3_idx);

     GpT(mlp, data3, data4, weight1, bias1, labeln);
     GpO(mlp, fc1, softmax);

     mlp->pushOpNode(argmax, print_top3);
     mlp->pushTensorNode(top3_idx);

     //====================================================
     Device dev_cpu0;
     Device dev_cpux;
     dev_cpux.id = 999;

     //-----------CPU0-------------------------------------
     TensorNode *data0_rep0 = new TensorNode("data0");
     data0_rep0->setTensor(data0->getTensor());
     TensorNode *weight0_rep0 = new TensorNode("weight0");
     weight0_rep0->setTensor(weight0->getTensor());
     TensorNode *bias0_rep0 = new TensorNode("bias0");
     bias0_rep0->setTensor(bias0->getTensor());

     DYOP(scatter00, ScatterOp, 0, 2);
     DYOP(scatter01, ScatterOp, -1, 2);
     DYOP(scatter02, ScatterOp, -1, 2);
     scatter01->setRunOnce();
     scatter02->setRunOnce();
     LINKUPPER(scatter00, data0_rep0);
     LINKUPPER(scatter01, weight0_rep0);
     LINKUPPER(scatter02, bias0_rep0);

     TENSOR(data0_cpux, 4, 784);
     TENSOR(weight0_cpux, 784, 512);
     TENSOR(bias0_cpux, 512);
     weight0_cpux_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
     bias0_cpux_Tensor->setTensorInit(TensorInitType::PARENTOP, 0);
     LINKUPPER(data0_cpux, scatter00);
     LINKUPPER(weight0_cpux, scatter01);
     LINKUPPER(bias0_cpux, scatter02);

     OP(matmul0_cpux, MatrixMatrixFCBiasOp);
     LINKUPPER(matmul0_cpux, data0_cpux, weight0_cpux, bias0_cpux);
     TENSOR(data1_cpux, 4, 512);
     LINKUPPER(data1_cpux, matmul0_cpux);

     OP(tanh0_cpux, MatrixTanhOp);
     LINKUPPER(tanh0_cpux, data1_cpux);
     TENSOR(data2_cpux, 4, 512);
     LINKUPPER(data2_cpux, tanh0_cpux);

     DYOP(gather0, GatherOp, 0, 2);
     LINKUPPER(gather0, data2_cpux);

     TensorNode *data2_rep0 = new TensorNode("data2");
     data2_rep0->setTensor(data2->getTensor());
     LINKUPPER(data2_rep0, gather0);


	 scatter00->setPolymorphic(true);
	 scatter01->setPolymorphic(true);
	 scatter02->setPolymorphic(true);
	 gather0->setPolymorphic(true);



     IRGraph *subGraph0 = new IRGraph();
     subGraph0->pushTensorNode(data0_rep0, weight0_rep0, bias0_rep0, data0_cpux,
                               weight0_cpux, bias0_cpux, data1_cpux, data2_cpux,
                               data2_rep0);
     subGraph0->pushOpNode(scatter00, scatter01, scatter02, matmul0_cpux,
                           tanh0_cpux, gather0);

     data0_rep0->setExternal(true);
     weight0_rep0->setExternal(true);
     bias0_rep0->setExternal(true);
     data2_rep0->setExternal(true);
     subGraph0->setDeviceLabel(dev_cpux);
     //====================================================

     cpux_Op->setGraph(subGraph0);
	 mlp->setLevel(0);
	 subGraph0->setLevel(1);

     mlp->findInOut();
     mlp->updateTopology();
     pass::Optimizer *opt = new pass::Optimizer(mlp);
     opt->runOptimizer();

     subGraph0->findInOut();
     subGraph0->updateTopology();
     opt->setGraph(subGraph0);
     opt->runOptimizer();
     //====================================================

     dotGen(mlp);
     dotGen(subGraph0, "subGraph0.dot");
     //====================================================
     CodegenConfig config;
     config.mpi = true;
     codegen::Codegen *cg = new codegen::Codegen(mlp, config);
     string code = cg->generate();
     cout << code;

     return 0;
 }
