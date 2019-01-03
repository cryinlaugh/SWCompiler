/*************************************************************************
	> File Name: optimizer.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 11 Dec 2018 07:31:15 AM UTC
 ************************************************************************/

#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include "IRGraph.h"
#include "SWLOG.h"
#include "Label.h"

namespace swc{

class Optimizer{
private:
    
public:
    Optimizer(){};
    ~Optimizer(){};

    template <typename Dtype>
    void runOptimize(IRGraph<Dtype>* graph){
        SWLOG_INFO << "Start doing optimization."<<std::endl;
        runLabelingPass(0, graph);
        runLoweringPass(graph);
        runLabelingPass(0, graph);
    }

    template <typename Dtype>
    void initLabelingPass(IRGraph<Dtype>* graph){
        //check and init Label in each node
        int nTensorNodes = graph->tensorNodeNum();
        int nOpNodes = graph->opNodeNum();

        for(int i=0; i<nTensorNodes; i++){
            TensorNode<Dtype>* tnode = graph->getTensorNode(i);
            Label* tlabel = tnode->getLabel();
            tlabel->setNodeNameLabel(std::string("TensorNode"));
            tlabel->setTypeNameLabel(std::string("Tensor"));

            SWLOG_INFO << tlabel->getNodeNameLabel() <<std::endl;
            SWLOG_INFO << tlabel->getTypeNameLabel() <<std::endl;
            
        }
        for(int i=0; i<nOpNodes; i++){
            OpNode<Dtype>* tnode = graph->getOpNode(i);
            Label* tlabel = tnode->getLabel();
            tlabel->setNodeNameLabel(std::string("OpNode"));
            tlabel->setTypeNameLabel((tnode->getOp())->getOpName());
            SWLOG_INFO << tlabel->getNodeNameLabel() <<std::endl;
            SWLOG_INFO << tlabel->getTypeNameLabel() <<std::endl;
        }
    }

    template <typename Dtype>
    void testLoweringLabelingPass(IRGraph<Dtype>* graph){
        int nTensorNodes = graph->tensorNodeNum();
        int nOpNodes = graph->opNodeNum();

        for (int i=0; i<nTensorNodes; i++){
            TensorNode<Dtype>* tnode = graph->getTensorNode(i);
            Label* tlabel = tnode->getLabel();
            //do nothing for tensor nodes
            SWLOG_INFO << "Do nothing for " << tlabel->getTypeNameLabel() << " ." << std::endl;
        }

        for (int i=0; i<nOpNodes; i++){
            OpNode<Dtype>* tnode = graph->getOpNode(i);
            Label* tlabel = tnode->getLabel();
            if ((tlabel->getTypeNameLabel()).compare("MatrixMatrixFC") == 0){
                SWLOG_INFO << tlabel->getTypeNameLabel() << " operator is marked to be lowered." << std::endl;
                tlabel->setLowerMark();
            }else{
                SWLOG_INFO << "Do nothing for " << tlabel->getTypeNameLabel() << " operator." << std::endl;
            }
        }
    }

    template <typename Dtype>
    void runLabelingPass(int type, IRGraph<Dtype>* graph){
        SWLOG_INFO << "Start initial labeling pass: " <<std::endl;
        initLabelingPass(graph);
        SWLOG_INFO << "Finish initial labeling pass: "  <<std::endl;
        if(type == 0) {
            SWLOG_INFO << "Start test labeling pass: " <<std::endl;
            testLoweringLabelingPass(graph);
            SWLOG_INFO << "Finish test labeling pass: "  <<std::endl;
        }
    }

    template <typename Dtype>
    void runLoweringPass(IRGraph<Dtype>* graph){
        SWLOG_INFO << "Start lowering pass: " << std::endl;

        int nTensorNodes = graph->tensorNodeNum();
        int nOpNodes = graph->opNodeNum();

        for (int i=0; i<nTensorNodes; i++){
            TensorNode<Dtype>* tnode = graph->getTensorNode(i);
            Label* tlabel = tnode->getLabel();
            //do nothing for tensor nodes
            SWLOG_INFO << "Do nothing for " << tlabel->getTypeNameLabel() << " ." << std::endl;
        }

        for (int i=0; i<nOpNodes; i++){
            OpNode<Dtype>* tnode = graph->getOpNode(i);
            Label* tlabel = tnode->getLabel();
            if(tlabel->getLowerMark()){
                tnode->getOp()->lowering(graph, tnode);
            }
        }

        SWLOG_INFO << "Finish lowering pass. " << std::endl;

    }

};

}
#endif
