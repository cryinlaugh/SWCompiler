/*************************************************************************
	> File Name: CommSavingParaLoweringPass.cpp
	> Author: wayne
	> Mail:  
	> Created Time: Wed 09 Oct 2019 06:42:49 AM UTC
 ************************************************************************/
#include "ParallelLoweringPass.h"
#include <set>

namespace swc {
namespace pass {

void ParallelLoweringPass::runExpCommSavingLowering(int parallel_num) {
    auto config = _graph->getConfig();

    //------------------------step1---------------------------------
    // get tensor nodes and op nodes in topology order
    std::vector<TensorNode * > topoTensorNodes;
    std::vector<OpNode *> topoOpNodes;
    for (int i = 0; i < _graph->topologyNum(); i++) {
        for (int j = 0; j < _graph->getNumInTopoLevel(i); j++) {
            IRNode * irnode = _graph->getNodeInTopo(i, j);
            if(irnode->nodeType() == TENSOR_NODE) {
                topoTensorNodes.push_back((TensorNode*)irnode);

            } else if(irnode->nodeType() == OP_NODE) {
                topoOpNodes.push_back((OpNode*)(irnode));
            }

        }
    }

    //------------------------step2---------------------------------
    // collect split plans for trainable weights
    std::map<TensorNode*, std::set<int>> trainableWeights;
    for(auto opnode : topoOpNodes) {
        StrategyLabel* slabel = opnode->getStrategyLabel();

        if(slabel == NULL)
            continue;

        std::vector<int> op_strategy = slabel->getStrategy();

        int nInputs = opnode->parentNum();
        for(int i=0; i<nInputs; i++) {
            auto *tnode = (TensorNode*) opnode->getParentNode(i);
            int t_strategy = op_strategy.at(i);
            // collect trainable weight tensors
            if(tnode->getTraining()) {
                if(trainableWeights.count(tnode)) {
                    auto &t_plans = trainableWeights.at(tnode);
                    t_plans.insert(t_strategy);
                }else {
                    trainableWeights[tnode]  = {t_strategy};
                }
            } // trainable weights

        } // for inputs of opnode

    } // for all opnodes

    /*
    std::cout << "---------strategies for trainable weight tensors----------------\n";
    for(auto iter : trainableWeights) {
        std::cout << iter.first->name() << ": ";
        for(auto ts : iter.second) {
            std::cout << ts << " ";
        }
        std::cout << "\n";
    }
    */

    //------------------------step3---------------------------------
    // apply fork but do not link to operator
    for(auto iter : trainableWeights) {
        auto &tnode = iter.first;
        TilingLabel *tlabel = tnode->getTilingLabel();

        for(auto strategy : iter.second) {
            if(tlabel->strategySize() == 0) {
                // the same as tlabel->getAppiled() == false
                ForkPattern* forkpattern = new ForkPattern(tnode, parallel_num);
                forkpattern->apply(strategy, _graph);

                // after apply, tlabel->getCurrentNode() points to new par_tnode
                TensorNode *par_tnode = tlabel->getCurrentNode();

                tlabel->insertStrategy(strategy, par_tnode);

            } else if(tlabel->strategyExist(strategy)) {
                // will not happen for set
                std::cout << "error\n";
                exit(0);
            }else {
                // weight has parallel strategies, but not same (ascend order -1. 0, 1...)
                SWLOG_DEBUG(4) << tnode->name() << " select best transform source\n"; 
                int pre_strategy = tlabel->selectTransPreStrategy(strategy); 

                TransformPattern * transformpattern = new TransformPattern(tnode, parallel_num);
                transformpattern->apply(pre_strategy, strategy, _graph);

                TensorNode *par_tnode = tlabel->getCurrentNode();

                tlabel->insertStrategy(strategy, par_tnode);
            }

        }

    } // step3

    //------------------------step4---------------------------------
    // it's time to really transform the graph
    
    for(auto *curOpNode : topoOpNodes){

        /**
         * wayne 2019.10.10
         * 关于Optimzer的问题，今日和zwl兄讨论的结果
         * 1. Optimzer Op不加入策略空间
         * 2. 分PS架构，和分布式(如AllReduce)应该做为不同配置
         * 所作修改:
         * 1. PS架构时，假定SGD等Optimzer不可拆即可
         * 2. de-centarlized optimzer时，SGD应该严格按照w的scatter方案做并行
         * 3. 给SGD按照w的scatter方案打上label 注:目前会导致output的stgy也可能-1
         * 3a. SGD的输入正常处理（按照打的stgy看怎么变换)
         * 3b. w的scatter方案为-1， 对应SGD为AllReduce,输出的tensor*与w-1指向同一个即可
         * 3c. w的scatter方案为i, 对应SGD应当是local update, dw_j可能需要transform到dw_i,
         *   输出应当构造一个tilied_out与w_i指向同一tensor
         * 4. IRGraph::removeRedundantScatter时，-1也消除(只要scatter的父节点出度为1)
         * 其他：
         * 1.后续的对输出处理中还有 SGD/Optimizer Specific的代码，注意
         * 2.本代码通过过初步测试.
         *
         */
        if(curOpNode->getStrategyLabel() == NULL) {
            SWLOG_DEBUG(4) << "WARNING: " << curOpNode->name() << " now no strategy\n"; 

            // centralized like PS, op run serially
            if(config.decentralized_optimizer == false) {
                continue;
            }

            // remind ! non-optimizer operator continue too
            if(!dynamic_cast<SGDOp*>(curOpNode->getOp())) {
                continue;
            }

            // reach this, must be sgd | other optimizer 
            // w dw momentum -> w_cp 
            auto *weight = (TensorNode*) curOpNode->getParentNode(0);
            auto *weightGrad = (TensorNode*) curOpNode->getParentNode(1);
            //auto *momentum = (TensorNode*) curOpNode->getParentNode(2);
            //auto *out = (TensorNode*) curOpNode->getChildNode(0);
            TilingLabel *weightLabel = weight->getTilingLabel();
            TilingLabel *weightGradLabel = weightGrad->getTilingLabel();

            assert(weightLabel->strategySize() && " SGD.W no parallelization strategy");
            assert(weightGradLabel->strategySize() && " SGD.dW no parallelization strategy");

            // select smallest strategy
            auto w_s = weightLabel->selectTransPreStrategy();
            auto dw_s = weightGradLabel->selectTransPreStrategy();
            (void)dw_s;

            assert(dw_s!=-1 && "gradient is out, ParallelStrategy cannot be -2");

            /** Note: Optimizer strategy may be -1 -1 -1 -1, output stgy maybe not be -1
             *
             *
             */
            curOpNode->setStrategyLabel(new StrategyLabel({w_s, w_s, w_s, w_s}));
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
                SWLOG_DEBUG(4) << tnode->name() << " select best transform source\n"; 
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

        //------------------------special for SGD or other optimizer-----------------------
        if(dynamic_cast<SGDOp*>(curOpNode->getOp())) {
            // w dw momentum -> w_cp 
            // when reach this
            // SGD must be labeled {w_s, w_s, w_s, w_s}
            // e.g. {-1, -1, -1, -1}, {0, 0, 0, 0}
            auto *weight = (TensorNode*) curOpNode->getParentNode(0);
            auto *out = (TensorNode*) curOpNode->getChildNode(0);

            int out_strategy = op_strategy.at(nInputs + 0);

            if(out_strategy == -1) {
                out->setTensor(weight->getTensor());
                continue;
            }


            auto *out_tile = new TensorNode(out->name() + "_tile", weight->getTensor());
            curOpNode->replaceOutKeepOrder(out_tile, 0);
            // _graph->delTensorNode(out);
            _graph->pushTensorNode(out_tile);

            _graph->addLogicalOutNodes(out_tile);
            continue;

        }
        //------------------------special for SGD---------------------------------



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
    } // for opnodes


    //---------------

} // runLowering
} // namespace pass
} // namespace swc
