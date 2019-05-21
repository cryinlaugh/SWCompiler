# SWCompiler C++ API
The SWCompiler C++ API provides high level C++ interface for neural network specification, optimization pass, visualization and compilation. The API provides:
* C++ interface for specifying deep learning models.
* Datasets loading module. (TODO: parallel io on multiple nodes)
* Model serialization for model importing and exporting, as well as snapshot/checkpoints.
* Interfae for pre-defined computing kernels and libraries.
* Auto differentiation for explicitly building training network from model.
* PassManager for optimizaton and transformation passes on IR Graph.
* Model visualizaiton.
* Codegen interfaces for model-to-source transformation.

## Network specification
__TensorNode definition__

_code pieces 1_ define a TensorNode "weight0", whose dimension is {784, 512}, add this node to graph.
```c++
IRGraph graph = new IRGraph();
...
auto *weight0 = new TensorNode("weight0", {784, 512});
weight0->getTensor()->setTensorInit(TensorInitType::FILE, "mlp_weight_0.bin");

graph->pushTensorNode(weight0);
```

__OpNode definition__

_code pieces 2_ define an OpNode "fc0", a fully connect operator, link input Tensornode data0, weight0 and bias0, and then add it to graph.
```c++
OpNode *fc0 = new OpNode("fc0", new MatrixMatrixFCOp());    // define OpNode

fc0->exlinkUpper(data0, weight0, bias0);                    // link to input TensorNodes

graph->pushOpNode(fc0);                                     // add to graph
```

## Auto differntiation
TODO: optimizer specification
We need to define optimizer SGD/Adam/RMSprop and others, and then construct network(graph) for training. Currently, only SGD is supported.
```c++
TrainingProfile profile; // default profile {lr: 0.001, decay: 0.001, momentum: 0.9, batch: 1}
profile.batch = data_0->getDims()[0];
IRGraph *net = getTrainNet(mlp, profile);
```

## Run optimization passes

_code pieces_ run some mandatory passes for graph transformations.
```c++
PassManager passManager;
RenamingNodePass renamingpass(net);
LabelingPass labelingpass(net);
LoweringPass loweringpass(net);
passManager.add((OptimizePass *)&renamingpass);
passManager.add((OptimizePass *)&labelingpass);
passManager.add((OptimizePass *)&loweringpass);
passManager.add((OptimizePass *)&labelingpass);
passManager.run();imizePass *)&labelingpass;
passManager.run();
```

## Codegen
todo

## End-to-end example (Inference)
We use pre-trained weights in test/input.
In bianry build directory.
```shell
cmake -DCMAKE_BUILD_TYPE=Debug -DLEVELDEBUG=4 ..
make testInfer
./testInfer
```
Then, we get generated files: Graph.cpp and other dependencies.
```c++
g++ -O3 Graph.cpp
./a.out
```
You may get these results.
```
5 0 3
0 5 3
4 0 3
1 8 2
9 4 7
2 0 9
1 3 8
3 0 8
```

test/testInfer.cpp source code with comments.
```c++
#include <iostream>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace std;

int main() {
    // macros for defining TensorNodes and OpNodes
    TENSOR(data0, 8, 784);
    TENSOR(weight0, 784, 512);
    TENSOR(bias0, 512);

    // set tensor initialization
    data0_Tensor->setTensorInit(TensorInitType::FILE, "input/mnist_images_8.bin");
    weight0_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_weight0.bin");
    bias0_Tensor->setTensorInit(TensorInitType::FILE, "input/mlp_bias0.bin");

    OP(fc0, MatrixMatrixFCOp);
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

    OP(fc1, MatrixMatrixFCOp);
    LINKUPPER(fc1, data2, weight1, bias1);

    TENSOR(data3, 8, 10);
    LINKUPPER(data3, fc1);

    OP(softmax, MatrixSoftmaxOp);
    LINKUPPER(softmax, data3);

    TENSOR(data4, 8, 10);
    LINKUPPER(data4, softmax);

    // get the top3 prediction results of label ids.
    auto *argmax_o = new OpNode("argmax", new ArgMaxOp(3));
    argmax_o->exlinkUpperNode(data4);
    auto *top3_t =
        new TensorNode("top3", new Tensor({8, 3}, DataType::Int32_t), argmax_o);
    auto *print_o = new OpNode("print", new DebugOp());
    print_o->exlinkUpperNode(top3_t);

    // define IR graph
    G(mlp);
    GpT(mlp, data0, weight0, bias0, data1, data2, weight1, bias1, data3, data4,
        top3_t);
    GpO(mlp, fc0, tanh0, fc1, softmax, argmax_o, print_o);

    //====================================================
    // do topology sort and run passes.
    mlp->updateTopology();
    pass::Optimizer *opt = new pass::Optimizer(mlp);
    opt->runOptimizer();

    //====================================================
    // generate svg for IRGraph
    dotGen(mlp);

    //====================================================
    // C++ code generation for running
    CodegenConfig config;
    codegen::Codegen *cg = new codegen::Codegen(mlp, config);
    string code = cg->generate();
    
    cout << code;

    return 0;
}
```
