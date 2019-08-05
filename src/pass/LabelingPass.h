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
#include "OptimizePass.h"
#include "SWLOG.h"


#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"

namespace swc {
namespace pass {
class LabelingPass;
}
} // namespace swc

class swc::pass::LabelingPass : public swc::pass::OptimizePass {
    // private:
    // IRGraph* _graph;
    using OptimizePass::_graph;

  public:
    LabelingPass(IRGraph *graph) : OptimizePass(graph){};
    ~LabelingPass(){};

    void initLabelingPass() {
        // check and init Label in each node
        int nTensorNodes = _graph->tensorNodeNum();
        int nOpNodes = _graph->opNodeNum();

        for (int i = 0; i < nTensorNodes; i++) {
            TensorNode *node = _graph->getTensorNode(i);
            Label *label = node->getLabel();
            label->setNodeNameLabel(node->name());
            label->setTypeNameLabel(std::string("Tensor"));
        }

        for (int i = 0; i < nOpNodes; i++) {
            OpNode *node = _graph->getOpNode(i);
            Label *label = node->getLabel();
            label->setNodeNameLabel(node->name());
            label->setTypeNameLabel((node->getOp())->getOpName());
        }
    }
    void setLoweringMark() {
        int nTensorNodes = _graph->tensorNodeNum();
        int nOpNodes = _graph->opNodeNum();

        for (int i = 0; i < nTensorNodes; i++) {
            TensorNode *tnode = _graph->getTensorNode(i);
            Label *label = tnode->getLabel();
            (void)label;
            // do nothing for tensor nodes
        }

        for (int i = 0; i < nOpNodes; i++) {
            OpNode *node = _graph->getOpNode(i);
            Label *label = node->getLabel();
            if ((label->getTypeNameLabel()).compare("MatrixMatrixFCBias") == 0) {
                SWLOG_DEBUG(2)
                    << label->getTypeNameLabel() << " "
                    << label->getNodeNameLabel()
                    << " operator is marked to be lowered." << std::endl;
                label->setLowerMark();
            } else if ((label->getTypeNameLabel())
                           .compare("MatrixMatrixFCGrad") == 0) {
                SWLOG_DEBUG(2)
                    << label->getTypeNameLabel() << " "
                    << label->getNodeNameLabel()
                    << " operator is marked to be lowered." << std::endl;
                label->setLowerMark();
            } else if ((label->getTypeNameLabel())
                           .compare("MatrixMatrixFCBiasGrad") == 0) {
                SWLOG_DEBUG(2)
                    << label->getTypeNameLabel() << " "
                    << label->getNodeNameLabel()
                    << " operator is marked to be lowered." << std::endl;
                label->setLowerMark();
            } else {
            }
        }
    }

    void setTraining() {
        int nTensorNodes = _graph->tensorNodeNum();
        for (int i = 0; i < nTensorNodes; i++) {
            TensorNode *tnode = _graph->getTensorNode(i);
            Label *label = tnode->getLabel();
            label->setTraining(tnode->getTraining());

            SWLOG_DEBUG(2) << label->getTypeNameLabel() << " "
                           << label->getNodeNameLabel()
                           << " train: " << tnode->getTraining() << std::endl;
        }
    }

    void run() {
        SWLOG_DEBUG(4) << "Start Labeling Pass." << std::endl;
        initLabelingPass();
        setLoweringMark();
        setTraining();
        SWLOG_DEBUG(4) << "Finish Labeling Pass." << std::endl;
    }
};
#endif
