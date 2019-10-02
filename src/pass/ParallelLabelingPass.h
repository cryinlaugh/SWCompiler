/***********************************************
#
#      Filename: src/pass/ParallelLabelingPass.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-09-18 15:40:59
# Last Modified: 2019-09-18 15:40:59
***********************************************/
#ifndef _PARALLEL_LABELING_PASS_H
#define _PARALLEL_LABELING_PASS_H

#include "common.h"
#include "OptimizePass.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "parallel/TilingLabel.h"
#include "parallel/parallelGen.h"
#include "parallel/SearchSpace.h"
#include <random>
#include <algorithm>
#include <ctime>

namespace swc {
namespace pass {
class ParallelLabelingPass;
}

class swc::pass::ParallelLabelingPass: public swc::pass::OptimizePass {
    using OptimizePass::_graph;
    mutable std::mt19937_64 rng{std::random_device{}()};
public:
    ParallelLabelingPass(IRGraph *graph): OptimizePass(graph) {
        srand(time(NULL));
    };
    ~ParallelLabelingPass() {};
    void runLabeling(int p) {

        std::vector<TensorNode * > topoTensorNodes;
        std::vector<OpNode *> topoOpNodes;

        for (int i = 0; i < _graph->topologyNum(); i++) {
            for (int j = 0; j < _graph->getNumInTopoLevel(i); j++) {
                IRNode * irnode = _graph->getNodeInTopo(i, j);
                if(irnode->nodeType() == TENSOR_NODE) {
                    SWLOG_DEBUG(4) << "tensornode : (" << i
                        << "," << j << ") " << irnode->name() << std::endl;
                    topoTensorNodes.push_back(dynamic_cast<TensorNode *>(irnode));
                } else if(irnode->nodeType() == OP_NODE) {
                    SWLOG_DEBUG(4) << "opnode: (" << i
                        << "," << j << ") " << irnode->name() << std::endl;
                    topoOpNodes.push_back(dynamic_cast<OpNode*>(irnode));
                }

            }
        }
        //init tiling label
        for(unsigned long  i = 0; i < topoTensorNodes.size(); ++i) {
            TensorNode * originNode = topoTensorNodes[i];
            TilingLabel * tlabel =  new TilingLabel();
            originNode->setTilingLabel(tlabel);
        }

        std::cout<<topoOpNodes.size()<<std::endl;

        //get startegy
        for(unsigned long i =0; i<topoOpNodes.size(); ++i){
            OpNode * originNode = topoOpNodes[i];
            SWLOG_DEBUG(10) << i << " get Candidate strategies for "<<originNode->name()
                << " : " << originNode->getOp()->getOpName()<<"\n";

            if(originNode->getOp()->getEinOp()== 0){
                std::cout << originNode->name()<<" can not be parallelized (einOp=0)\n";
                continue;
            }

            std::vector<std::vector<int> > strategies = ParallelGen::generateStgy(originNode->getOp());
            int strategy_size = strategies.size();
            if(strategy_size == 0){
                std::cout << originNode->name()<<" get 0 strategies\n";
                continue;
            }

            for(auto sgy : strategies){
                for(auto dim: sgy)
                    std::cout<<dim<<" ";
                std::cout<<"\n";
            }

            std::vector<std::vector<int>> legal_strategies; // first available strategy

            // channel % degree may not be zero
            // finalstrategy = strategies[0];
            //int nInputs = originNode->getOp()->getnInput();
            //int nOutputs = originNode->getOp()->getnOutput();
            int nInputs = originNode->parentNum();
            int nOutputs = originNode->childNum();
            for(auto strategy : strategies) {
                bool legal = true;
                int idx = 0;
                for(auto tensor_dim: strategy) {
                    
                    Tensor* tensor;
                    if(idx < nInputs) {
                        tensor = ((TensorNode*)originNode->getParentNode(idx))->getTensor();
                    } else if(idx < (nInputs+nOutputs)) {
                        tensor = ((TensorNode*)originNode->getChildNode(idx-nInputs))->getTensor();
                    } else {
                        legal = false;
                        break;
                    }

                    if(tensor_dim >= 0) {
                        if(tensor->getDim(tensor_dim) % p) {
                            legal = false;
                            break;
                        }
                    }
                    idx++;
                } // for parallel dim in this strategy
                
                if(legal)
                    legal_strategies.push_back(strategy);
            }

            if(legal_strategies.size() > 0) {
                // int random_s_idx = rand() % legal_strategies.size();
                std::uniform_int_distribution<size_t> dist(0, legal_strategies.size()-1);
                int random_s_idx  = dist(rng);

                auto best = legal_strategies[random_s_idx];

                if(_graph->getConfig().force_data_parallel) {
                    best = ParallelGen::generateDataParStgy(originNode); 
                }


                std::cout << "-----legal strategies------\n";
                for(auto sgy : legal_strategies){
                    for(auto s: sgy)
                        std::cout << s <<" ";
                    std::cout<<"\n";
                }

                std::cout << "-----selected strategy no." << random_s_idx << "------\n";
                for(auto s : best)
                    std::cout << s << " ";
                std::cout << "\n";

                StrategyLabel* slabel =  new StrategyLabel(best);
                originNode->setStrategyLabel(slabel);
            } else {
                std::cout << "-----no legal strategy------\n";
            }

        }

    }
    void run() {
        SWLOG_DEBUG(4) << "Start Paralleling Pass." << std::endl;

        auto parallel_degree = _graph->getConfig().mpi_size;
        assert(parallel_degree>1 && "error, degree of parallellism unset, please set config.mpi_size");
        runLabeling(parallel_degree);
        //get startegy
        SWLOG_DEBUG(4) << "Finish Paralleling pass. " << std::endl;

    }



};

}
#endif
