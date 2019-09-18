/***********************************************
#
#      Filename: src/pass/ParallelLoweringPass.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-09-18 16:07:46
# Last Modified: 2019-09-18 16:07:46
***********************************************/
#include "common.h"
#include "OptimizePass.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "parallel/TilingLabel.h"
#include "parallel/ParallelPattern.h"
namespace swc {
namespace pass {
class ParallelLoweringPass;
}

class swc::pass::ParallelLoweringPass: public swc::pass::OptimizePass {
    using OptimizePass::_graph;
public:
    ParallelLoweringPass(IRGraph *graph): OptimizePass(graph) {
    };
    ~ParallelLoweringPass() {};
    void runLowering(int parallelnum) {

        std::vector<TensorNode * > topoTensorNodes;
        std::vector<OpNode *> topoOpNodes;

        for (int i = 0; i < _graph->topologyNum(); i++) {
            for (int j = 0; j < _graph->getNumInTopoLevel(i); j++) {
                IRNode * irnode = _graph->getNodeInTopo(i, j);
                if(irnode->nodeType() == TENSOR_NODE) {
                    SWLOG_DEBUG(4) << "tensornode :" << i
                        << "," << j << "-" << irnode->name() << std::endl;
                    topoTensorNodes.push_back(dynamic_cast<TensorNode *>(irnode));
                } else if(irnode->nodeType() == OP_NODE) {
                    SWLOG_DEBUG(4) << "opnode: " << i
                        << "," << j << "-" << irnode->name() << std::endl;
                    topoOpNodes.push_back(dynamic_cast<OpNode*>(irnode));
                }

            }
        }

        for(unsigned long i = 0; i < topoOpNodes.size(); i++) {
            OpNode* curOpNode = topoOpNodes[i];
            if(curOpNode->getStrategyLabel() == NULL) {

                SWLOG_DEBUG(4) << "WARNING: " << curOpNode->name()
                << " now no strategy" << std::endl;
                continue;
            }

            StrategyLabel* slabel = curOpNode->getStrategyLabel();
            std::vector<int> opstrategy = slabel->getStrategy();
            int strategyindex = 0;

            std::ostringstream ss_strategy;
            for(auto s : opstrategy) {
                ss_strategy << s << " ";
            }
            SWLOG_DEBUG(4) << "Dealing with OpNode " << curOpNode->name()
            << " strategy: " << ss_strategy.str() << "\n";

            std::vector<TensorNode*> tempinput;
            for(int j = 0; j < curOpNode->parentNum(); j++) {
                TensorNode * inputNode = dynamic_cast<TensorNode*>(curOpNode->getParentNode(j));
                tempinput.push_back(inputNode);
            }
            for(unsigned long j = 0; j < tempinput.size(); j++) {

                TensorNode * originNode = tempinput[j];
                TilingLabel * tlabel = originNode->getTilingLabel();
                int strategy =  opstrategy[strategyindex];
                strategyindex++;

                if(!tlabel->isApplied()) {
                    ForkPattern* forkpattern = new ForkPattern(originNode, parallelnum);
                    forkpattern->apply(strategy, _graph);
                    curOpNode ->destroyUpperNode(originNode);
                    curOpNode -> exlinkUpperNode(tlabel->getCurrentNode());

                } else if(strategy != tlabel->getCurrentStrategy()) {
                    if(tlabel->getCurrentStrategy() == -2) {
                        // joinpattern fllowed by forkpatter...
                        ForkPattern* forkpattern = new ForkPattern(originNode, parallelnum);
                        forkpattern->apply(strategy, _graph);
                        curOpNode ->destroyUpperNode(originNode);
                        curOpNode -> exlinkUpperNode(tlabel->getCurrentNode());
                        continue;
                    }
                    // transfrom pattern
                    TransformPattern * transformpattern = new TransformPattern(originNode, parallelnum);
                    transformpattern->apply(tlabel->getCurrentStrategy(), strategy, _graph);
                    curOpNode -> destroyUpperNode(originNode);
                    curOpNode ->exlinkUpperNode(tlabel->getCurrentNode());


                } else {

                    curOpNode -> destroyUpperNode(originNode);
                    curOpNode->exlinkUpperNode(tlabel->getCurrentNode());
                    //directly link with current tiling
                }

            }


            //for Output
            std::vector<TensorNode*> tempoutput;
            for(int j = 0; j < curOpNode->childNum(); j++) {
                TensorNode * outputNode = dynamic_cast<TensorNode*>(curOpNode->getChildNode(j));
                tempoutput.push_back(outputNode);
            }
            for(unsigned long  j = 0; j < tempoutput.size(); j++) {

                TensorNode * originNode = tempoutput[j];

                TilingLabel * tlabel = originNode->getTilingLabel();

                int strategy = opstrategy[strategyindex];
                strategyindex++;
                if(!tlabel->isApplied()) {
                    JoinPattern* joinpattern = new JoinPattern(originNode, parallelnum);
                    joinpattern->apply(strategy, _graph);
                    originNode->destroyUpperNode(curOpNode);
                    tlabel->getCurrentNode()->exlinkUpperNode(curOpNode);
                    //join pattern
                // beblow two conditions should not occur
                } else if(strategy != tlabel -> getCurrentStrategy()) {
                    TransformPattern * transformpattern = new TransformPattern(originNode, parallelnum);
                    transformpattern->apply(strategy, _graph);
                    originNode->destroyUpperNode(curOpNode);
                    tlabel->getCurrentNode()->exlinkUpperNode(curOpNode);
                    //transform pattern
                } else {
                    originNode->destroyUpperNode(curOpNode);
                    tlabel->getCurrentNode()->exlinkUpperNode(curOpNode);
                    //directly link with current tiling node
                }

            }

        }

    }


// construct parallel zone



    void run() {
        //SWLOG_DEBUG(4) << "Start Paralleling Pass." << std::endl;
        runLowering(4);
        //SWLOG_DEBUG(4) << "Finish Paralleling pass. " << std::endl;

        // //std::cout<<"test"<<std::endl;
        // runTileLowering();
        // SWLOG_INFO << "Finish Lowering Pass." << std::endl;
    }



};

}
