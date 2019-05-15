/***********************************************
#
#      Filename: LoweringPass.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-01-23 10:57:27
# Last Modified: 2019-01-23 10:57:27
***********************************************/
#ifndef _LOWERINGPASS_H
#define _LOWERINGPASS_H
#include "OptimizePass.h"
#include "SWLOG.h"

#include "TileHint.h"
#include "TilingLabel.h"

#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"

namespace swc {
namespace pass {
class LoweringPass;
}
} // namespace swc

class swc::pass::LoweringPass : public OptimizePass {
    using OptimizePass::_graph;

  public:
    LoweringPass(IRGraph *graph) : OptimizePass(graph){};
    ~LoweringPass(){};

    void runTileLowering() {

        this->_graph->findInOut();
        this->_graph->updateTopology();
        this->_graph->updateTopoNodeList();

        printf("\nGraph Struct:\nIn:");
        for (int i = 0; i < this->_graph->inNodeNum(); i++)
            printf("%s  ", this->_graph->getInNode(i)->name().c_str());
        printf("\nOut:");
        for (int i = 0; i < this->_graph->outNodeNum(); i++)
            printf("%s  ", this->_graph->getOutNode(i)->name().c_str());
        printf("\n\nTopology List:\n");
        for (int i = 0; i < this->_graph->topologyNum(); i++) {
            printf("TopologyID: %d\t", i);
            for (int j = 0; j < this->_graph->getNumInTopoLevel(i); j++)
                printf("%s  ",
                       this->_graph->getNodeInTopo(i, j)->name().c_str());
            printf("\n");
        }

        printf("\nNode Info:\n");
        for (int i = 0; i < this->_graph->tensorNodeNum(); i++) {
            printf("ID:%d, ", i);
            printf("TopologyID:%d, ",
                   this->_graph->getTensorNode(i)->topologyId());
            printf("Name:%s, ", this->_graph->getTensorNode(i)->name().c_str());
            printf("in:%d, ", this->_graph->getTensorNode(i)->parentNum());
            printf("out:%d\n", this->_graph->getTensorNode(i)->childNum());
        }

        for (int i = 0; i < this->_graph->opNodeNum(); i++) {
            printf("ID:%d, ", i);
            printf("TopologyID:%d, ", this->_graph->getOpNode(i)->topologyId());
            printf("Name:%s, ", this->_graph->getOpNode(i)->name().c_str());
            printf("in:%d, ", this->_graph->getOpNode(i)->parentNum());
            printf("out:%d\n", this->_graph->getOpNode(i)->childNum());
        }

        printf("---------------------------------------------------------------"
               "-----------------------\n");

        int nOpNodes = this->_graph->opNodeNum();
        std::vector<OpNode *> toDeleteOps;
        // int
        for (int i = 0; i < nOpNodes; i++) {
            OpNode *opnode = this->_graph->getOpNode(i);

            // lowering opnode
            OpTilingLabel *optilinglabel =
                dynamic_cast<OpTilingLabel *>(opnode->getLabel());

            std::vector<OpNode *> tileOpNodes;

            int tilenum = optilinglabel->getReplicateNum();

            for (int j = 0; j < tilenum; ++j) {
                //(std::string)opnode->name()+std::to_string(j)
                OpNode *tileOpNode = new OpNode("replicateOp");
                tileOpNodes.push_back(tileOpNode);
                this->_graph->pushOpNode(tileOpNode);
            }

            // lowering input tensors
            for (int j = 0; j < opnode->parentNum(); ++j) {
                TensorNode *inputTensorNode =
                    dynamic_cast<TensorNode *>(opnode->getParentNode(j));
                TensorTilingLabel *tlabel = dynamic_cast<TensorTilingLabel *>(
                    inputTensorNode->getLabel());

                // inputTensorNode->name()+"-tile"
                OpNode *tileOpNode = new OpNode("spiltOp");
                // TBD: get label num to check it spilt  or replicate and define
                // opnode
                tileOpNode->exlinkUpperNode(inputTensorNode);
                this->_graph->pushOpNode(tileOpNode);
                int tilenum = tlabel->getTotalTileNum();
                for (int k = 0; k < tilenum; ++k) {
                    //(std::string)inputTensorNode->name()+std::to_string(k)
                    TensorNode *tileTensorNode = new TensorNode("spiltTensor");
                    tileTensorNode->exlinkUpperNode(tileOpNode);
                    tileOpNodes[k]->exlinkUpperNode(tileTensorNode);
                    this->_graph->pushTensorNode(tileTensorNode);
                }
            }

            // lowering output tensors
            for (int j = 0; j < opnode->childNum(); ++j) {
                TensorNode *outputTensorNode =
                    dynamic_cast<TensorNode *>(opnode->getChildNode(j));
                TensorTilingLabel *tlabel = dynamic_cast<TensorTilingLabel *>(
                    outputTensorNode->getLabel());

                // outputTensorNode->name()+"-tile"
                OpNode *tileOpNode = new OpNode("concactOp");
                this->_graph->pushOpNode(tileOpNode);
                // TBD: get label num to check it concact  or reduce and define
                // opnode
                outputTensorNode->exlinkUpperNode(tileOpNode);
                int tilenum = tlabel->getTotalTileNum();
                for (int k = 0; k < tilenum; ++k) {
                    //(std::string)outputTensorNode->name()+std::to_string(k)
                    TensorNode *tileTensorNode =
                        new TensorNode("concactTensor");
                    tileOpNode->exlinkUpperNode(tileTensorNode);
                    tileTensorNode->exlinkUpperNode(tileOpNodes[k]);
                    this->_graph->pushTensorNode(tileTensorNode);
                }
            }

            toDeleteOps.push_back(opnode);
        }
        for (size_t i = 0; i < toDeleteOps.size(); i++) {
            OpNode *opnode = toDeleteOps[i];
            int parentNum = opnode->parentNum();
            int childNum = opnode->childNum();
            for (int j = 0; j < parentNum; ++j) {
                opnode->destroyUpperNode(opnode->getParentNode(j));
            }
            for (int j = 0; j < childNum; ++j) {
                opnode->getChildNode(j)->destroyUpperNode(opnode);
            }
            this->_graph->delOpNode(opnode);
        }

        this->_graph->findInOut();
        this->_graph->updateTopology();
        this->_graph->updateTopoNodeList();

        printf("\nGraph Struct:\nIn:");
        for (int i = 0; i < this->_graph->inNodeNum(); i++)
            printf("%s  ", this->_graph->getInNode(i)->name().c_str());
        printf("\nOut:");
        for (int i = 0; i < this->_graph->outNodeNum(); i++)
            printf("%s  ", this->_graph->getOutNode(i)->name().c_str());
        printf("\n\nTopology List:\n");
        for (int i = 0; i < this->_graph->topologyNum(); i++) {
            printf("TopologyID: %d\t", i);
            for (int j = 0; j < this->_graph->getNumInTopoLevel(i); j++)
                printf("%s  ",
                       this->_graph->getNodeInTopo(i, j)->name().c_str());
            printf("\n");
        }

        printf("\nNode Info:\n");
        for (int i = 0; i < this->_graph->tensorNodeNum(); i++) {
            printf("ID:%d, ", i);
            printf("TopologyID:%d, ",
                   this->_graph->getTensorNode(i)->topologyId());
            printf("Name:%s, ", this->_graph->getTensorNode(i)->name().c_str());
            printf("in:%d, ", this->_graph->getTensorNode(i)->parentNum());
            printf("out:%d\n", this->_graph->getTensorNode(i)->childNum());
        }

        for (int i = 0; i < this->_graph->opNodeNum(); i++) {
            printf("ID:%d, ", i);
            printf("TopologyID:%d, ", this->_graph->getOpNode(i)->topologyId());
            printf("Name:%s, ", this->_graph->getOpNode(i)->name().c_str());
            printf("in:%d, ", this->_graph->getOpNode(i)->parentNum());
            printf("out:%d\n", this->_graph->getOpNode(i)->childNum());
        }
    }

    void runLowering() {

        int nTensorNodes = _graph->tensorNodeNum();
        int nOpNodes = _graph->opNodeNum();

        for (int i = 0; i < nTensorNodes; i++) {
            TensorNode *tnode = _graph->getTensorNode(i);
            Label *tlabel = tnode->getLabel();
            (void)tlabel;
        }

        for (int i = 0; i < nOpNodes; i++) {
            OpNode *tnode = _graph->getOpNode(i);
            Label *tlabel = tnode->getLabel();
            if (tlabel->getLowerMark()) {
                tnode->getOp()->lowering(_graph, tnode);
            } else {
            }
        }
    }

    void run() {
        SWLOG_DEBUG(4) << "Start Lowering Pass." << std::endl;
        runLowering();
        SWLOG_DEBUG(4) << "Finish lowering pass. " << std::endl;

        // //std::cout<<"test"<<std::endl;
        // runTileLowering();
        // SWLOG_INFO << "Finish Lowering Pass." << std::endl;
    }
};
#endif
