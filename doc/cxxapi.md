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

## End-to-end example
todo