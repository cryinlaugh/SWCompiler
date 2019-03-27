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
#include "SWLOG.h"
#include "TilingLabel.h"
#include "TileHint.h"
#include "../src/graphIR/IRGraph.h"
#include "OptimizePass.h"
namespace swc {



template<typename Dtype>
class LabelingPass: public OptimizePass<Dtype> {
//private:
    //IRGraph<Dtype>* _graph;
public:
    LabelingPass(IRGraph<Dtype> * graph): OptimizePass<Dtype>(graph) {};
    ~LabelingPass() {};

//    void initLabelingPass(){

//        // //check and init Label in each node
//    	int nTensorNodes =this-> _graph->tensorNodeNum();
//    	int nOpNodes = this->_graph->opNodeNum();

    //     for (int i = 0; i < nTensorNodes; i++) {
    //         TensorNode<Dtype>* tnode = this->_graph->getTensorNode(i);
    //         Label* tlabel = tnode->getLabel();
    //         tlabel->setNodeNameLabel(std::string("TensorNode"));
    //         tlabel->setTypeNameLabel(std::string("Tensor"));

    //         //SWLOG_INFO << tlabel->getNodeNameLabel() << std::endl;
    //         //SWLOG_INFO << tlabel->getTypeNameLabel() << std::endl;

    //     }

    //     for (int i = 0; i < nOpNodes; i++) {
    //         OpNode<Dtype>* tnode = this->_graph->getOpNode(i);
    //         Label* tlabel = tnode->getLabel();
    //         tlabel->setNodeNameLabel(std::string("OpNode"));
    //         tlabel->setTypeNameLabel((tnode->getOp())->getOpName());

    //         //SWLOG_INFO << tlabel->getNodeNameLabel() << std::endl;
    //         //SWLOG_INFO << tlabel->getTypeNameLabel() << std::endl;
    //     }
    // }

    void runTiling() {
        //get all tilehints from opnode and send it to related nodes


        int nOpNodes = this->_graph->opNodeNum();
        int nTensorNodes = this-> _graph->tensorNodeNum();



        for (int i = 0; i < nOpNodes; i++) {
            OpNode<Dtype>* opnode = this->_graph->getOpNode(i);
            std::vector<TensorNode<Dtype>*> tilingTensorNodes;
            for (int i = 0; i < opnode->parentNum(); ++i) {
                tilingTensorNodes.push_back((TensorNode<Dtype>*)opnode->getParentNode(i));
            }
            for (int i = 0; i < opnode->childNum(); ++i) {
                tilingTensorNodes.push_back((TensorNode<Dtype>*)opnode->getChildNode(i));

            }
            for(int i = 0; i < tilingTensorNodes.size(); i++) {
                // Label* tlabel = tilingTensorNodes[i]->getLabel();
                TensorTilingLabel* tensortilinglabel = new TensorTilingLabel(2);
                tilingTensorNodes[i]->setLabel(tensortilinglabel);
                //add tilehint to tensorTiling label
            }

            OpTilingLabel * optilinglabel = new OpTilingLabel();
            opnode->setLabel(optilinglabel);
        }

        //采用策略，遍历每个tensor的tilehint集合设置tilelabel
        //a  simplestategy, 所有的都在维度0上2分
        for(int i = 0; i <  nTensorNodes; i++) {
            TensorTilingLabel*  tlabel = dynamic_cast<TensorTilingLabel*>(this->_graph->getTensorNode(i)->getLabel());
            tlabel->addTileBydim(0, 2);
        }

        for(int i = 0; i <  nOpNodes; i++) {
            OpTilingLabel* olabel = dynamic_cast<OpTilingLabel*>(this->_graph->getOpNode(i)->getLabel());
            olabel->setReplicateNum(2);
        }

    }
    void run() {
        SWLOG_INFO << "Start Tile Labeling Pass." << std::endl;
        runTiling();
        SWLOG_INFO << "Finish Tile Labeling Pass." << std::endl;
    }

};
}
#endif
