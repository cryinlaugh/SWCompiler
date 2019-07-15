/***********************************************
#
#      Filename: src/pass/ParallelingPass.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-07-05 15:51:37
# Last Modified: 2019-07-05 15:51:37
***********************************************/
#include "OptimizePass.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "parallel/TilingLabel.h"
//#include "string"
namespace swc {
namespace pass {
class ParallelingPass;
}

class swc::pass::ParallelingPass: public swc::pass::OptimizePass {
    using OptimizePass::_graph;
public:
    ParallelingPass(IRGraph *graph): OptimizePass(graph) {};
    ~ParallelingPass() {};

    std::vector<int> getStrategy(OpNode * op, int num) {

        std::vector<int> test(num,1);
        return test;
        
        //return {0, -1, 0};//for test
    }

    void runParalleling(int parallelnum) {
        //int nOpNodes = _graph->opNodeNum();
        //int nTensorNodes =  _graph->tensorNodeNum();
        for (int i = 0; i < _graph->topologyNum(); i++) {
            for (int j = 0; j < _graph->getNumInTopoLevel(i); j++) {

                std::cout<<i<<","<<j<<std::endl;
                IRNode * node = _graph->getNodeInTopo(i, j);
                if(dynamic_cast<OpNode*>(node) != nullptr) {
                    std::vector<TensorNode* > tilingTensorNodes;

                    for (int k = 0; k < node->parentNum(); ++k) {
                        tilingTensorNodes.push_back((TensorNode*)node->getParentNode(k));
                    }
                    for (int k = 0; k < node->childNum(); ++k) {
                        tilingTensorNodes.push_back((TensorNode*)node->getChildNode(k));
                    }


                    int tilingTensorNum = tilingTensorNodes.size();
                    std::vector<int> strategy = getStrategy(dynamic_cast<OpNode*>(node),tilingTensorNum);
                    for(int k = 0; k < tilingTensorNum; k++) {
                        

                        // std::cout<<tilingTensorNum<<std::endl;
                        // std::cout<<strategy.size()<<std::endl;
                        // std::cout<<k<<std::endl;

                        std::cout<< tilingTensorNodes[k]->name() <<std::endl;
                        Label  * tlabel = tilingTensorNodes[k]->getLabel();



                        std::cout<<"test"<<std::endl;


                        if(!tlabel->isAssign()) {

                            //int ndim=tilingTensorNodes[k]->getTensor()->getNDim();
                            TensorTilingLabel * tensortilinglabel = new TensorTilingLabel(2);
                            if(strategy[k] >= 0)
                                tensortilinglabel->addTileBydim(strategy[k], parallelnum);
                            else if(strategy[k] == -1 )
                                tensortilinglabel->replicate(parallelnum);
                            else if(strategy[k] == -2)
                                tensortilinglabel->reduce(parallelnum);

                            tensortilinglabel->assign();
                            tlabel->assign();
                            tilingTensorNodes[k]->setLabel(tensortilinglabel);
                            std::cout<<"test2"<<std::endl;
                        }

                    }
                   
                    OpTilingLabel * opTilingLabel = new OpTilingLabel();
                    node->setLabel(opTilingLabel);

                   


                }

            }

        }
    }

    void run() {
        //SWLOG_DEBUG(4) << "Start Paralleling Pass." << std::endl;
        runParalleling(4);
        //SWLOG_DEBUG(4) << "Finish Paralleling pass. " << std::endl;

        // //std::cout<<"test"<<std::endl;
        // runTileLowering();
        // SWLOG_INFO << "Finish Lowering Pass." << std::endl;
    }



};

}
