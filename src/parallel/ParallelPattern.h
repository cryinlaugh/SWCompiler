/***********************************************
#
#      Filename: src/parallel/ParallelPattern.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-07-05 11:04:16
# Last Modified: 2019-07-05 11:04:16
***********************************************/
#include "op/Op.h"
#include "op/dlop/dlOp.h"
#include "graphIR/IRGraph.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "TilingLabel.h"
using namespace swc::op;

namespace swc {


// class Maper{

// private:
//     int _type;//0: replicate, 1:map


//     int _node;// which input
//     int _index;//




// };

// class Reducer{



//     int _type;//built in or custom
//     int _node;
//     int _index;//





// }







//make sure each share to have no-loose boundary sufficently
class ParallelPattern {
private:
    int _num;
    Op * _op;
public:
    ParallelPattern(Op * op) {
        _op = op;

    };
    ~ParallelPattern() {};
    


    

    void applyToGraph(IRGraph * graph, IRNode *node ) {
        //std::vector<int> strategy = {2, -1, 2}; //testhints :  2 means spilt at dim 2, -1 means replicate , -2 means repliate

        std::vector<OpNode *> parallelOpNodes;

        std::vector<IRGraph* > subGraphs;

        //paralleling op
        for(int k = 0; k < _num; k++) {
            OpNode * replicateOpNode = new OpNode(node->name() + "_rep");
            parallelOpNodes.push_back(replicateOpNode);
            
            IRGraph *subGraph = new IRGraph();
            subGraph->pushOpNode(replicateOpNode);
            subGraphs.push_back(subGraph);
        }
        //paralleling input
        for (int i = 0; i < node->parentNum(); ++i) {
            TensorNode* input = dynamic_cast<TensorNode*>(node->getParentNode(i));
            //TensorTilingLabel *tlabel = dynamic_cast<TensorTilingLabel *>(input->getLabel());//TBD

            //int tilenum = tlabel->getTotalTileNum();
            for (int k = 0; k < _num; ++k) {
                TensorNode * cloneInput = new TensorNode(input->name() + "_clone");
                cloneInput->setTensor(input->getTensor());

                OpNode * scatterOpNode = new OpNode(input->name() + "_scatter"); //TBD：scatter op node
                //auto *sop = (ScatterOp *)scatterOpNode->getOp();
                //sop->setOffset(4 * 784);

                TensorNode *tileInput = new TensorNode(input->name() + "_tile");
                //setTensor


                parallelOpNodes[k]->exlinkUpperNode(tileInput);
                tileInput->exlinkUpperNode(scatterOpNode);
                scatterOpNode->exlinkUpperNode(cloneInput);
                subGraphs[k]->pushTensorNode(cloneInput, tileInput);
                subGraphs[k]->pushOpNode(scatterOpNode);
            }


        }

        //paralleling output
        for (int i = 0; i < node->childNum(); ++i) {
            TensorNode* output = dynamic_cast<TensorNode *>(node->getChildNode(i));
            //TensorTilingLabel *tlabel = dynamic_cast<TensorTilingLabel *>(output->getLabel());

            //int tilenum = tlabel->getTotalTileNum();
            for(int k = 0; k < _num; ++k) {
                TensorNode * cloneOutput = new TensorNode(output->name() + "_clone");
                cloneOutput->setTensor(output->getTensor());

                OpNode * gatherOpNode =  new OpNode(output->name() + "_gather");
                // set op;
                //
                TensorNode * tileOutput = new TensorNode(output->name() + "_tile" );

                //set Tensor
                cloneOutput->exlinkUpperNode(gatherOpNode);
                gatherOpNode->exlinkUpperNode(tileOutput);
                tileOutput->exlinkUpperNode(parallelOpNodes[k]);
                subGraphs[k]->pushTensorNode(tileOutput,cloneOutput);
                subGraphs[k]->pushOpNode(gatherOpNode);
            }
        }


        for(int k=0;k<_num;k++){
            OpNode* subGraphNode = new OpNode(node->name()+"_subgraph"); 
            subGraphNode->setOp(new SubGraphOp());
        }




    }



}
















};
}
