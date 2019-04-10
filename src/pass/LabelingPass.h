/***********************************************
#
#      Filename: LabelingPass.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-01-21 10:57:27
# Last Modified: 2019-01-21 10:57:27
***********************************************/
#ifndef _LABELINGPASS_H
#define _LABELINGPASS_H
#include "SWLOG.h"
#include "OptimizePass.h"

#include "TileHint.h"
#include "TilingLabel.h"

#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
namespace swc {

class LabelingPass: public OptimizePass {
//private:
    //IRGraph* _graph;
    using OptimizePass::_graph;
public:
    LabelingPass(IRGraph * graph): OptimizePass(graph) {};
    ~LabelingPass() {};

   void initLabelingPass(){
        //check and init Label in each node
        int nTensorNodes = _graph->tensorNodeNum();
        int nOpNodes = _graph->opNodeNum();

        for (int i = 0; i < nTensorNodes; i++) {
            TensorNode* node = _graph->getTensorNode(i);
            Label* label = node->getLabel();
            label->setNodeNameLabel(node->name());
            label->setTypeNameLabel(std::string("Tensor"));
        }

        for (int i = 0; i < nOpNodes; i++) {
            OpNode* node = _graph->getOpNode(i);
            Label* label = node->getLabel();
            label->setNodeNameLabel(node->name());
            label->setTypeNameLabel((node->getOp())->getOpName());
        }
    }
    void setLoweringMark() {
        int nTensorNodes = _graph->tensorNodeNum();
        int nOpNodes = _graph->opNodeNum();

        for (int i = 0; i < nTensorNodes; i++) {
            TensorNode* tnode = _graph->getTensorNode(i);
            Label* label = tnode->getLabel();
            //do nothing for tensor nodes
            SWLOG_INFO << "Do nothing for " << label->getTypeNameLabel() << " " << label->getNodeNameLabel()
                        << " ." << std::endl;
        }

        for (int i = 0; i < nOpNodes; i++) {
            OpNode* node = _graph->getOpNode(i);
            Label* label = node->getLabel();
            if ((label->getTypeNameLabel()).compare("MatrixMatrixFC") == 0) {
                SWLOG_INFO << label->getTypeNameLabel()
                            << " operator is marked to be lowered." << std::endl;
                label->setLowerMark();
            } else {
                SWLOG_INFO << "Do nothing for " << label->getTypeNameLabel()
                            << " operator " << node->name() << std::endl;
            }
        }
    }

    void run() {
        SWLOG_INFO << "Start Labeling Pass." << std::endl;
        initLabelingPass();
        setLoweringMark();
        SWLOG_INFO << "Finish Labeling Pass." << std::endl;
    }

};
}
#endif
