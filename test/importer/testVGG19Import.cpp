/*************************************************************************
	> File Name: test/importer/testVGG19Import.cpp
	> Author: wayne
	> Mail:  
	> Created Time: Mon 23 Sep 2019 02:54:46 AM UTC
 ************************************************************************/

#include "SWC.h"
#include "importer/Caffe2Importer.h"
#include <iostream>

using namespace swc;
using namespace swc::pass;
using namespace std;

int main() {
    IRGraph *graph = new IRGraph();
    std::vector<TensorNode *> udef;
    auto *data = new TensorNode("data", {1, 224, 224, 3});
    data->getTensor()->setTensorInit(TensorInitType::FILE, "input/cat_285.bin");
    udef.push_back(data);
    Caffe2Importer importer(graph, "./vgg19/predict_net.pb",
                            "./vgg19/init_net.pb", udef);

    auto *softmax_t = graph->getNodeByName("prob");
    assert(softmax_t && "softmax not found");

    auto *argmax_o = new OpNode("argmax", new ArgMaxOp(10));
    argmax_o->exlinkUpperNode(softmax_t);

    auto *top3_t =
        new TensorNode("top3", new Tensor({1, 10}, DataType::Int32_t), argmax_o);

    auto *print_o = new OpNode("print", new DebugOp());
    print_o->exlinkUpperNode(top3_t);

    graph->pushOpNode(argmax_o, print_o);
    graph->pushTensorNode(top3_t);

    auto *placeholder = new TensorNode("null", {0}, print_o);
    graph->pushTensorNode(placeholder);
    //---------------------------------------------------------------

    graph->findInOut();
    graph->updateTopology();

    dotGen(graph, "vgg19-import.dot");

    Config config;
    config.mkldnn = true;
    graph->setConfig(config);

    Backend backend(graph); 
    backend.compile();

    dotGen(graph, "vgg19-compiled.dot");
    string code = backend.genCode();

    return 0;
}
