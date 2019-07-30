/*
 * EliminationPass.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-07-29
 */


#include "EliminationPass.h"

#include <stdlib.h>
#include <string>

#include "SWLOG.h"
#include "graphIR/IRGraph.h"
#include "graphIR/IRNode.h"
#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"

using namespace std;

namespace swc {
namespace pass {

void EliminationPass::destroy() 
{
    SWLOG_DEBUG(4) << "EliminationPass Destroy" << endl;
    return;
}


void EliminationPass::run(IRGraph* graph)
{
    SWLOG_DEBUG(4) << "EliminationPass Run" << endl;
    //graph_train = _graph;

    std::vector<IRNode*> topo_nodes;

    //get each level nodes;
    for (int i = 0; i < graph->topologyNum(); i++) {
        for (int j = 0; j < graph->getNumInTopoLevel(i); j++) {
            auto node = graph->getNodeInTopo(i, j);
            SWLOG_DEBUG(4) << "TopoLevel.." << i << "\tType..." 
                << (node->nodeType() == TENSOR_NODE ? "TENSOR\t" : "OP\t")
                << (node->name()) << std::endl;
            topo_nodes.push_back(node);
        }
    }

    //node elimination
    int flag = 1;
    while(flag) {
        flag = 0;
        for (auto it = topo_nodes.rbegin(); it != topo_nodes.rend(); it++) {
            IRNode* irnode = *it;

            if ((irnode->getLabel()->getIsOut() == 0) &&
                    (irnode->childNum() == 0)) {

                SWLOG_DEBUG(4) << "Node " << irnode->name()
                    << " is not marked out node and the out rank is zero. "
                    << " Eliminate it!" << std::endl;

                flag = 1;
               
                // break link;
                while(irnode->parentNum()) {
                    irnode->destroyUpperNode(irnode->getParentNode(0));
                }
                // remove from graph
                if (irnode->nodeType() == TENSOR_NODE) {
                    TensorNode* tnode = (TensorNode*)irnode;
                    graph->delTensorNode(tnode);
                    tnode->destroy();
                } else {
                    OpNode* onode = (OpNode*)irnode;
                    graph->delOpNode(onode);
                    onode->destroy();
                }
                if (!delVecMember(topo_nodes, irnode)) {
                    std::cout << "Del irnode Failed" << irnode->name() << std::endl;
                }
                break;
            }
        }
    }
                
    graph->updateTopology();
}



} //namespace pass
} //namespace swc

