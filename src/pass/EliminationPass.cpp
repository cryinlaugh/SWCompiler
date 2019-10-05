/*
 * EliminationPass.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-07-29
 */


#include "EliminationPass.h"

#include <cstdlib>
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


void EliminationPass::run()
{
    SWLOG_DEBUG(4) << "EliminationPass Run" << endl;

    // set isOut mark for logical out nodes (you want to keep)
    // !!! (not topology order out nodes, e.g. we want to remove nodes for input_grad)
    _graph->setLogicalOutMark();

    std::vector<IRNode*> topo_nodes;

    //get each level nodes;
    for (int i = 0; i < _graph->topologyNum(); i++) {
        for (int j = 0; j < _graph->getNumInTopoLevel(i); j++) {
            auto node = _graph->getNodeInTopo(i, j);
            SWLOG_DEBUG(2) << "TopoLevel.." << i << "\tType..."
                << (node->nodeType() == TENSOR_NODE ? "TENSOR\t" : "OP\t")
                << (node->name()) << std::endl;
            topo_nodes.push_back(node);
        }
    }

    //node elimination
    for (auto it = topo_nodes.rbegin(); it != topo_nodes.rend(); it++) {
        IRNode* irnode = *it;

        SWLOG_DEBUG(2) << "[Node] " << irnode->name() << " " << irnode->getLabel()->getIsOut() 
            << " " << irnode->childNum() << "\n";

        if(irnode->getLabel()->getIsOut()
            || irnode->childNum()>0) {
            continue;
        }


        if (irnode->nodeType() == TENSOR_NODE) {
            // parentNum of TensorNode should <= 1
            auto parent_op = irnode->parentNum()>0 ? irnode->getParentNode(0) : nullptr;
            if(parent_op && parent_op->childNum() > 1)
                continue;

            SWLOG_DEBUG(6) << "TensorNode " << irnode->name()
                << " not marked out | out degree=0 | is only child of parent, "
                << " Eliminate it!" << std::endl;

            while(irnode->parentNum()) {
                irnode->destroyUpperNode(irnode->getParentNode(0));
            }

            TensorNode* tnode = (TensorNode*)irnode;
            _graph->delTensorNode(tnode);
            tnode->destroy();

            if (!delVecMember(topo_nodes, irnode)) {
                std::cout << "Del irnode Failed" << irnode->name() << std::endl;
                exit(0);
            }

            // break;

        } else if(irnode->nodeType() == OP_NODE) {
            // outdegree of Tensornode can be many, and order does not matter 
            SWLOG_DEBUG(6) << "OpNode " << irnode->name()
                << " not marked out | out degree=0,"
                << " Eliminate it!" << std::endl;

            while(irnode->parentNum()) {
                irnode->destroyUpperNode(irnode->getParentNode(0));
            }

            OpNode* onode = (OpNode*)irnode;
            _graph->delOpNode(onode);
            onode->destroy();

            if (!delVecMember(topo_nodes, irnode)) {
                std::cout << "Del irnode Failed" << irnode->name() << std::endl;
                exit(0);
            }

            // break;
        }
    }
    /*
    int flag = 1;
    while(flag) {
        flag = 0;
        for (auto it = topo_nodes.rbegin(); it != topo_nodes.rend(); it++) {
            IRNode* irnode = *it;

            if ((irnode->getLabel()->getIsOut() == 0) &&
                    (irnode->childNum() == 0)) {

                SWLOG_DEBUG(2) << "Node " << irnode->name()
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
                    _graph->delTensorNode(tnode);
                    tnode->destroy();
                } else {
                    OpNode* onode = (OpNode*)irnode;
                    _graph->delOpNode(onode);
                    onode->destroy();
                }
                if (!delVecMember(topo_nodes, irnode)) {
                    std::cout << "Del irnode Failed" << irnode->name() << std::endl;
                }
                break;
            }
        }
    }
    */
    _graph->updateTopology();
}



} //namespace pass
} //namespace swc
