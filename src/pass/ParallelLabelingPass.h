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
    void runLabeling(int parallelnum) {

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
            std::vector<std::vector<int> > strategies = ParallelGen::generateStgy(originNode->getOp());
            //default select 0
            
            std::cout<<"Candidate strategies for "<<originNode->name()<<":\n";
            for(auto sgy : strategies){
                for(auto dim: sgy)
                    std::cout<<dim<<" ";
                std::cout<<"\n";
            }
            int strategy_size = strategies.size();
            std::vector<int> finalstrategy;
            if(strategy_size == 0){
            
                std::vector<int> test;
                for(int j=0;j<originNode->parentNum();++j){
                    test.push_back(0);
                 }
                 for(int j=0;j<originNode->childNum();++j){
                    test.push_back(0);
                }
                finalstrategy = test;
            }
            else{
                finalstrategy = strategies[0];
                StrategyLabel* slabel =  new StrategyLabel(finalstrategy);
                originNode->setStrategyLabel(slabel);
            }

        }

    }
    void run() {
        //SWLOG_DEBUG(4) << "Start Paralleling Pass." << std::endl;
        

        runLabeling(4);
        //get startegy 
        //SWLOG_DEBUG(4) << "Finish Paralleling pass. " << std::endl;

        // //std::cout<<"test"<<std::endl;
        // runTileLowering();
        // SWLOG_INFO << "Finish Lowering Pass." << std::endl;
    }



};

}
