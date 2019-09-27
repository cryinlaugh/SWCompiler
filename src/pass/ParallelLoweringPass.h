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
    }
    ~ParallelLoweringPass() {};

    void runMemSavingLowering(int parallel_num) {

        // get tensor nodes and op nodes in topology order
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

                SWLOG_DEBUG(4) << "WARNING: " << curOpNode->name() << " now no strategy\n"; 
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
                    ForkPattern* forkpattern = new ForkPattern(originNode, parallel_num);
                    forkpattern->apply(strategy, _graph);
                    curOpNode ->destroyUpperNode(originNode);
                    curOpNode -> exlinkUpperNode(tlabel->getCurrentNode());

                } else if(strategy != tlabel->getCurrentStrategy()) {
                    SWLOG_DEBUG(4) << originNode->name() << " strategy " << tlabel->getCurrentStrategy()
                        << " -> " << strategy << "\n";
                    if(tlabel->getCurrentStrategy() == -2) {
                        // joinpattern fllowed by forkpatter...
                        ForkPattern* forkpattern = new ForkPattern(originNode, parallel_num);
                        forkpattern->apply(strategy, _graph);
                        curOpNode ->destroyUpperNode(originNode);
                        curOpNode -> exlinkUpperNode(tlabel->getCurrentNode());
                        continue;
                    }
                    if(tlabel->getCurrentStrategy()>=0 && strategy == -1) {
                        // joinpattern fllowed by forkpattern...
                        ForkPattern* forkpattern = new ForkPattern(originNode, parallel_num);
                        forkpattern->apply(strategy, _graph);
                        curOpNode ->destroyUpperNode(originNode);
                        curOpNode -> exlinkUpperNode(tlabel->getCurrentNode());
                        continue;
                    }
                    // transfrom pattern
                    TransformPattern * transformpattern = new TransformPattern(originNode, parallel_num);
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
                    JoinPattern* joinpattern = new JoinPattern(originNode, parallel_num);
                    joinpattern->apply(strategy, _graph);
                    originNode->destroyUpperNode(curOpNode);
                    tlabel->getCurrentNode()->exlinkUpperNode(curOpNode);
                    //join pattern
                // beblow two conditions should not occur
                } else if(strategy != tlabel -> getCurrentStrategy()) {
                    SWLOG_DEBUG(4) << originNode->name() << " strategy " << tlabel->getCurrentStrategy()
                        << " -> " << strategy << "\n";
                    TransformPattern * transformpattern = new TransformPattern(originNode, parallel_num);
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

    void runCommSavingLowering(int parallel_num) {
        // get tensor nodes and op nodes in topology order
        std::vector<TensorNode * > topoTensorNodes;
        std::vector<OpNode *> topoOpNodes;

        for (int i = 0; i < _graph->topologyNum(); i++) {
            for (int j = 0; j < _graph->getNumInTopoLevel(i); j++) {
                IRNode * irnode = _graph->getNodeInTopo(i, j);
                if(irnode->nodeType() == TENSOR_NODE) {
                    SWLOG_DEBUG(2) << "tensornode :" << i
                        << "," << j << "-" << irnode->name() << std::endl;
                    topoTensorNodes.push_back(dynamic_cast<TensorNode *>(irnode));
                } else if(irnode->nodeType() == OP_NODE) {
                    SWLOG_DEBUG(2) << "opnode: " << i
                        << "," << j << "-" << irnode->name() << std::endl;
                    topoOpNodes.push_back(dynamic_cast<OpNode*>(irnode));
                }

            }
        }

        for(auto *curOpNode : topoOpNodes){

            if(curOpNode->getStrategyLabel() == NULL) {
                SWLOG_DEBUG(4) << "WARNING: " << curOpNode->name() << " now no strategy\n"; 
                continue;
            }

            StrategyLabel* slabel = curOpNode->getStrategyLabel();
            std::vector<int> op_strategy = slabel->getStrategy();

            std::ostringstream ss_strategy;
            for(auto s : op_strategy) {
                ss_strategy << s << " ";
            }
            SWLOG_DEBUG(4) << "Dealing with OpNode " << curOpNode->name()
                << " strategy: " << ss_strategy.str() << "\n";

            int nInputs = curOpNode->parentNum();
            int nOutputs = curOpNode->childNum();

            assert((nInputs+nOutputs)==(int)op_strategy.size() && "op_strategy.size() != (op.parentnum+op.childnum)");
             
            for(int i=0; i<nInputs; i++) {
                auto *tnode = (TensorNode*) curOpNode->getParentNode(i);
                int strategy = op_strategy.at(i);
                TilingLabel *tlabel = tnode->getTilingLabel();

                if(tlabel->strategySize() == 0) {
                    // the same as tlabel->getAppiled() == false
                    ForkPattern* forkpattern = new ForkPattern(tnode, parallel_num);
                    forkpattern->apply(strategy, _graph);

                    // after apply, tlabel->getCurrentNode() points to new par_tnode
                    TensorNode *par_tnode = tlabel->getCurrentNode();
                    tnode->replaceUseKeepOrder(curOpNode, par_tnode); 

                    tlabel->insertStrategy(strategy, par_tnode);

                } else if(tlabel->strategyExist(strategy)) {
                    TensorNode * par_tnode = tlabel->getStrategyParNode(strategy);  

                    // curOpNode->destroyUpperNode(tnode);
                    // curOpNode->exlinkUpperNode(par_tnode);
                    //
                    // Need to Keep Order 
                    tnode->replaceUseKeepOrder(curOpNode, par_tnode); 
                } else {
                    // tnode has history parallel strategies, but not same
                    
                    // select which history strategy to transform to strategy  
                    // 1. simply choose the newest one
                    // 2. DONE: choose best history strategy for transform (e.g. less count of pieces
                    // current: 2
                    SWLOG_DEBUG(6) << tnode->name() << " select best transform source\n"; 
                    int pre_strategy = tlabel->selectTransPreStrategy(strategy); 
                     
                    if(pre_strategy == -2) {
                        ForkPattern* forkpattern = new ForkPattern(tnode, parallel_num);
                        forkpattern->apply(strategy, _graph);

                        TensorNode *par_tnode = tlabel->getCurrentNode();
                        tnode->replaceUseKeepOrder(curOpNode, par_tnode); 

                        tlabel->insertStrategy(strategy, par_tnode);
                        continue;
                    }
                    if(pre_strategy>=0 && strategy == -1) {
                        // joinpattern fllowed by forkpattern...
                        ForkPattern* forkpattern = new ForkPattern(tnode, parallel_num);
                        forkpattern->apply(strategy, _graph);

                        TensorNode *par_tnode = tlabel->getCurrentNode();
                        tnode->replaceUseKeepOrder(curOpNode, par_tnode); 

                        tlabel->insertStrategy(strategy, par_tnode);
                        continue;
                    }
                    // transfrom pattern
                    TransformPattern * transformpattern = new TransformPattern(tnode, parallel_num);
                    transformpattern->apply(pre_strategy, strategy, _graph);

                    TensorNode *par_tnode = tlabel->getCurrentNode();
                    tnode->replaceUseKeepOrder(curOpNode, par_tnode); 

                    tlabel->insertStrategy(strategy, par_tnode);
                } // has history strategies, but not same

            } // for nInputs

            for(int i=0; i<nOutputs; i++) {
                auto *tnode = (TensorNode*) curOpNode->getChildNode(i);
                int strategy = op_strategy.at(nInputs + i);
                TilingLabel *tlabel = tnode->getTilingLabel();
                
                // indegree of TensorNode === 1, but outdegree may >1 
                // which means that if out tnode already has parallelization strategy
                // it must be parallelized by tnode's children
                // However, we run on opnods in topoOrder, so children should not have run 
                assert(tlabel->strategySize()==0 && "out tensornodes shouldn't be parallelized before me"); 

                JoinPattern* joinpattern = new JoinPattern(tnode, parallel_num);
                joinpattern->apply(strategy, _graph);

                TensorNode *par_tnode = tlabel->getCurrentNode();
                // std::cout << "join patter par_tnode= " << par_tnode->name() << "\n";

                // this is a must e.g
                curOpNode->replaceOutKeepOrder(par_tnode, i);

                tlabel->insertStrategy(strategy, par_tnode);
            }
        }

    } // runCommSavingLowering


    void run() {
        SWLOG_DEBUG(6) << "Start Paralleling Pass.\n";

        auto parallel_degree = _graph->getConfig().mpi_size;
        assert(parallel_degree>1 && "error, degree of parallellism unset, please set config.mpi_size");

        auto config = _graph->getConfig();

        if(config.parallel_preference == ParallelStrategy::MEM_SAVING) {
            SWLOG_DEBUG(6) << "runMemSavingLowering\n";
            runMemSavingLowering(parallel_degree);
        }
        else if(config.parallel_preference == ParallelStrategy::COMM_SAVING) {
            SWLOG_DEBUG(6) << "runCommSavingLowering\n";
            runCommSavingLowering(parallel_degree);
        }


        SWLOG_DEBUG(4) << "Finish Paralleling pass.\n";
    }

};

}
