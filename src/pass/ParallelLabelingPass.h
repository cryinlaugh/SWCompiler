/***********************************************
#
#      Filename: src/pass/ParallelLabelingPass.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-09-18 15:40:59
# Last Modified: 2019-09-18 15:40:59
***********************************************/
#include "common.h"
#include "OptimizePass.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "parallel/TilingLabel.h"
#include "parallel/parallelGen.h"
#include <random>

namespace swc {
namespace pass {
class ParallelLabelingPass;
}

class swc::pass::ParallelLabelingPass: public swc::pass::OptimizePass {
    using OptimizePass::_graph;
public:
    ParallelLabelingPass(IRGraph *graph): OptimizePass(graph) {
    };
    ~ParallelLabelingPass() {};
    void runLabeling(int p) {

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
        //init tiling label
        for(unsigned long  i = 0; i < topoTensorNodes.size(); ++i) {
            TensorNode * originNode = topoTensorNodes[i];
            TilingLabel * tlabel =  new TilingLabel();
            originNode->setTilingLabel(tlabel);
        }

        std::cout<<topoOpNodes.size()<<std::endl;
        //get startegy
        for(unsigned long i =0;i<topoOpNodes.size();++i){
            OpNode * originNode = topoOpNodes[i];
            SWLOG_DEBUG(10) <<"get Candidate strategies for "<<originNode->name()
                << " : " << originNode->getOp()->getOpName()<<"\n";

            if(originNode->getOp()->getEinOp()== 0){
                SWLOG_DEBUG(10) <<originNode->name()<<" can not be parallelized:\n";
                continue;
            }

            std::vector<std::vector<int> > strategies = ParallelGen::generateStgy(originNode->getOp());
            //default select 0

            for(auto sgy : strategies){
                for(auto dim: sgy)
                    std::cout<<dim<<" ";
                std::cout<<"\n";
            }

            int strategy_size = strategies.size();
            std::vector<std::vector<int>> legal_strategies; // first available strategy
            if(strategy_size == 0){
                return;
            }
            else{

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
                            tensor = ((TensorNode*)originNode->getParentNode(idx-nInputs))->getTensor();
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
                    int random_s_idx = rand() % legal_strategies.size();
                    auto best = legal_strategies[random_s_idx];

                    std::cout << "-----legal strategy------\n";
                    for(auto sgy : legal_strategies){
                        for(auto s: sgy)
                            std::cout << s <<" ";
                        std::cout<<"\n";
                    }

                    std::cout << "-----selected strategy------\n";
                    for(auto s : best)
                        std::cout << s << " ";
                    std::cout << "\n";

                    StrategyLabel* slabel =  new StrategyLabel(best);
                    originNode->setStrategyLabel(slabel);
                }
            }

        }

    }
    void run() {
        //SWLOG_DEBUG(4) << "Start Paralleling Pass." << std::endl;


        runLabeling(2);
        //get startegy
        //SWLOG_DEBUG(4) << "Finish Paralleling pass. " << std::endl;

        // //std::cout<<"test"<<std::endl;
        // runTileLowering();
        // SWLOG_INFO << "Finish Lowering Pass." << std::endl;
    }



};

}
