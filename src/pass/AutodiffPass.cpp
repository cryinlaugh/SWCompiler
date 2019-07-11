/*
 * AutodiffPass.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-05-23
 */

#include "AutodiffPass.h"

#include <stdlib.h>
#include <string>

#include "SWLOG.h"
#include "graphIR/IRGraph.h"
#include "graphIR/IRNode.h"

using namespace std;

namespace swc {
namespace pass {

METHOD_TYPE AutodiffPass::string2method(std::string& s)
{
    for (unsigned long i = 0; i < s.size(); i++)
       s[i] = tolower(s[i]);
    
    if (s == "sgd") return SGD_METHOD;
    if (s == "adam") return ADAM_METHOD;
    return -1;
}

void AutodiffPass::getMethods() 
{
    SWLOG_DEBUG(4)<<"No method determinated..."<<std::endl;
    SWLOG_DEBUG(4)<<"Please choose a solver: SGD, ADAM..."<<std::endl;
    abort();
}

void AutodiffPass::getSGDParameters()
{
    ((SGD_PARAMETERS*)_parameters)->lr = 0.001;
    ((SGD_PARAMETERS*)_parameters)->decay = 0.001;
    ((SGD_PARAMETERS*)_parameters)->momentum = 0.9;
    ((SGD_PARAMETERS*)_parameters)->batch = 1;
}
void AutodiffPass::getSGDParameters(float lr,
                                    float decay,
                                    float momentum,
                                    size_t batch)
{
    ((SGD_PARAMETERS*)_parameters)->lr = lr;
    ((SGD_PARAMETERS*)_parameters)->decay = decay;
    ((SGD_PARAMETERS*)_parameters)->momentum = momentum;
    ((SGD_PARAMETERS*)_parameters)->batch = batch;
}

void AutodiffPass::getADAMParameters()
{
    ((ADAM_PARAMETERS*)_parameters)->lr = 0.01;
}
void AutodiffPass::getADAMParameters(float lr)
{
    ((ADAM_PARAMETERS*)_parameters)->lr = lr;
}

void AutodiffPass::run(IRGraph* graph_train)
{
    SWLOG_DEBUG(4) << "AutodiffPass Run" << endl;
    _graph->copyTo(graph_train);
    //graph_train = _graph;

    std::vector<IRNode*> topo_nodes;
    std::unordered_map<IRNode*, IRNode*> gradNodeMap;

    //get each level nodes;
    for (int i = 0; i < graph_train->topologyNum(); i++) {
        for (int j = 0; j < graph_train->getNumInTopoLevel(i); j++) {
            auto node = graph_train->getNodeInTopo(i, j);
            SWLOG_DEBUG(4) << "TopoLevel.." << i << "\tType..." 
                << (node->nodeType() == TENSOR_NODE ? "TENSOR\t" : "OP\t")
                << (node->name()) << std::endl;
            topo_nodes.push_back(node);
        }
    }

    //node autodiff
    for (auto it = topo_nodes.rbegin(); it != topo_nodes.rend(); it++) {
        IRNode* irnode = *it;
       
        // if tensorNode build new grad tensorNode then add 
        // method(sgd .etc) opNode.
        if (irnode->nodeType() == TENSOR_NODE) {
            irnode->autoDiff(graph_train, gradNodeMap, _parameters, _method);
        }
        // if opNode builds opNode and tensorNode to calculate the grad.
        else if (irnode->nodeType() == OP_NODE) {
            irnode->autoDiff(graph_train, gradNodeMap);
        }
        else {
            std::cout << "illegal node type"<< std::endl;
            abort();
        }

        //for (auto it : gradNodeMap) {
        //    SWLOG_DEBUG(4) << "gradNode Map:"
        //        << "\t" << it.first->name() << "\t" << it.second->name() << "\n";
        //}
    }

}

void AutodiffPass::destroy()
{
    if (_parameters != NULL) free(_parameters);
    SWLOG_DEBUG(4) << "free AutodiffPass parameters" << endl;
}

void AutodiffPass::show()
{
    SWLOG_DEBUG(4) << "Show Methods:" << endl;
    switch(_method)
    {
        case SGD_METHOD:
            SWLOG_DEBUG(4) << "----SGD" << endl;
            SWLOG_DEBUG(4) << "----learning rate:" 
                << ((SGD_PARAMETERS*)_parameters)->lr << endl;
            SWLOG_DEBUG(4) << "----decay:"
                << ((SGD_PARAMETERS*)_parameters)->decay << endl;
            SWLOG_DEBUG(4) << "----momentum:"
                << ((SGD_PARAMETERS*)_parameters)->momentum << endl;
            SWLOG_DEBUG(4) << "----batch:"
                << ((SGD_PARAMETERS*)_parameters)->batch << endl;
            break;
        case ADAM_METHOD:
            SWLOG_DEBUG(4) << "----ADAM" << endl;
            SWLOG_DEBUG(4) << "----learning rate:" 
                << ((ADAM_PARAMETERS*)_parameters)->lr << endl;
            break;
        default:
            break;
    }
}


} //namespace pass
} //namespace swc

