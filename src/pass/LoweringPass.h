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
#include "parallel/TilingLabel.h"

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
    LoweringPass(IRGraph *graph) : OptimizePass(graph){}
    ~LoweringPass(){}

    void runLowering() {

        int nTensorNodes = _graph->tensorNodeNum();
        int nOpNodes = _graph->opNodeNum();
        std::vector<OpNode*> opNodes;
        for(int i=0; i<nOpNodes; i++)
            opNodes.push_back(_graph->getOpNode(i));


        for (int i = 0; i < nTensorNodes; i++) {
            TensorNode *tnode = _graph->getTensorNode(i);
            Label *tlabel = tnode->getLabel();
            (void)tlabel;
        }

        for (auto opnode : opNodes) {
            Label *tlabel = opnode->getLabel();
            SWLOG_DEBUG(10) << opnode->name() << " of " << 
               nOpNodes  << " lowering mark "
                << tlabel->getLowerMark() << "\n";
            if (tlabel->getLowerMark()) {
                opnode->getOp()->lowering(_graph, opnode);
            } else {
            }
        }
    }

    void run() {
        SWLOG_DEBUG(4) << "Start Lowering Pass.\n";
        runLowering();
        SWLOG_DEBUG(4) << "Finish lowering pass.\n\n";
    }
};
#endif
