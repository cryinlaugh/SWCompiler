/*************************************************************************
	> File Name: testResnet.cpp
	> Author: wayne
	> Mail:  
	> Created Time: 五  4/ 5 13:14:36 2019
 ************************************************************************/

#include "SWC.h"
#include "importer/Caffe2Importer.h"
#include <iostream>

using namespace swc;
using namespace std;

int main() {
    IRGraph *graph = new IRGraph();
    std::vector<TensorNode *> udef;
    auto *data = new TensorNode("gpu_0/data", {1, 224, 224, 3});
    data->getTensor()->setTensorInit(TensorInitType::FILE, "cat_285.bin");
    udef.push_back(data);
    Caffe2Importer loader(graph, "./resnet50/predict_net.pb",
                          "./resnet50/init_net.pb", udef);

    graph->findInOut();
    graph->updateTopology();
    graph->updateTopoNodeList();

    // Optimizer is a must because Codegen need label
    Optimizer *opt = new Optimizer(graph);
    opt->runOptimizer();

    dotGen(graph);

    codegen::Codegen *cg = new codegen::Codegen(graph);
    string code = cg->generate();
    cout << code;

    return 0;
}
