/*************************************************************************
	> File Name: testImporter.cpp
	> Author: wayne
	> Mail:  
	> Created Time: 四  3/28 16:16:24 2019
 ************************************************************************/

#include <iostream>
#include "SWC.h"
#include "importer/Caffe2Importer.h"

using namespace swc;
using namespace std;

int main() {
    IRGraph<float>* graph = new IRGraph<float>();
    std::vector<TensorNode<float>*> udef;
    auto *data = new TensorNode<float>("data", {8, 28, 28, 1});
    data->getTensor()->setTensorInit(TensorInitType::FILE, "mnist_images_8.bin");     
    udef.push_back(data);
    Caffe2Importer importer(graph, "./lenet_mnist/predict_net.pb", "./lenet_mnist/init_net.pb",
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
