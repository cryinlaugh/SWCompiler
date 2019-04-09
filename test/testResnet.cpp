/*************************************************************************
	> File Name: testResnet.cpp
	> Author: wayne
	> Mail:  
	> Created Time: 五  4/ 5 13:14:36 2019
 ************************************************************************/

#include <iostream>
#include "SWC.h"
#include "importer/Caffe2Importer.h"

using namespace swc;
using namespace std;

int main() {
    IRGraph<float>* graph = new IRGraph<float>();
    std::vector<TensorNode<float>*> udef;
    auto *data = new TensorNode<float>("gpu_0/data", {1, 224, 224, 3});
    data->getTensor()->setTensorInit(TensorInitType::FILE, "cat_285.bin");     
    udef.push_back(data);
    Caffe2Importer loader(graph, "./resnet50/predict_net.pb", "./resnet50/init_net.pb",
                        udef);

    graph->findInOut();
    graph->updateTopology();
    graph->updateTopoNodeList();

    // Optimizer is a must because Codegen need label 
    Optimizer<float>* opt = new Optimizer<float>(graph);
    opt->runOptimizer();

    dotGen(graph);

    codegen::Codegen<float>* cg = new codegen::Codegen<float>(graph);
    string code = cg->generate();
    cout << code;

	return 0;
}
