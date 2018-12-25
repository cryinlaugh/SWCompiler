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
    }

    template <typename Dtype>
    void initLabelingPass(IRGraph<Dtype>* graph){
        //check and init Label in each node
        int nTensorNodes = graph->tensorNodeNum();
        int nOpNodes = graph->opNodeNum();

        for(int i=0; i<nTensorNodes; i++){
            TensorNode<Dtype>* tnode = graph->getTensorNode(i);
            Label* tlabel = tnode->getLabel();
            tlabel->setTypeNameLabel(std::string(typeid(*tnode).name()));
            SWLOG_INFO << tlabel->getTypeNameLabel() <<std::endl;
            
        }
        for(int i=0; i<nOpNodes; i++){
            SWLOG_INFO << typeid(*(graph->getOpNode(i))).name()<<std::endl;
        }
    }

    template <typename Dtype>
    void runLabelingPass(int type, IRGraph<Dtype>* graph){
        if(type == 0) {
            SWLOG_INFO << "Start initial labeling pass: " <<std::endl;
            initLabelingPass(graph);
            SWLOG_INFO << "Finish initial labeling pass: "  <<std::endl;
        }
    }
};

}
#endif
