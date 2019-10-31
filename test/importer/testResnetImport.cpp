/*************************************************************************
        > File Name: testResnetImport.cpp
        > Author: wayne
        > Mail:
        > Created Time: 五  4/ 5 13:14:36 2019
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
    auto *data = new TensorNode("gpu_0/data", {1, 224, 224, 3});
    data->getTensor()->setTensorInit(TensorInitType::FILE, "input/cat_285.bin");
    udef.push_back(data);
    Caffe2Importer importer(graph, "./resnet50/predict_net.pb",
                            "./resnet50/init_net.pb", udef);

    auto *softmax_t = graph->getNodeByName("gpu_0/softmax");
    assert(softmax_t && "gpu0_softmax not found");

    auto *argmax_o = new OpNode("argmax", new ArgMaxOp(3));
    argmax_o->exlinkUpperNode(softmax_t);

    auto *top3_t =
        new TensorNode("top3", new Tensor({1, 3}, DataType::Int32_t), argmax_o);

    auto *print_o = new OpNode("print", new DebugOp());
    print_o->exlinkUpperNode(top3_t);

    graph->pushOpNode(argmax_o, print_o);
    graph->pushTensorNode(top3_t);

    auto *placeholder = new TensorNode("null", {0}, print_o);
    graph->pushTensorNode(placeholder);
    //---------------------------------------------------------------

    graph->findInOut();
    graph->updateTopology();

    dotGen(graph, "resnet_import.dot");

    Config config;
    config.mkldnn = true;
    graph->setConfig(config);

    Engine engine(graph); 
    engine.compile();

    dotGen(graph);
    string code = engine.genCode();

    return 0;
}
