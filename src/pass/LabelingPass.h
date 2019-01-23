#ifndef _LABELINGPASS_H
#define _LABELINGPASS_H 
#include "TilingLabel.h"
#include "TileHint.h"
#include "../src/graphIR/IRGraph.h"
#include "OptimizePass.h"
namespace swc{
template<typename Dtype>
class LabelingPass:public OptimizePass<Dtype>{
//private:
    //IRGraph<Dtype>* _graph;
public:
    LabelingPass(IRGraph<Dtype> * graph):OptimizePass<Dtype>(graph){};
    ~LabelingPass(){};

    void initLabelingPass(){

        // //check and init Label in each node
    	int nTensorNodes =this-> _graph->tensorNodeNum();
    	int nOpNodes = this->_graph->opNodeNum();

	    for (int i = 0; i < nTensorNodes; i++) {
	        TensorNode<Dtype>* tnode = this->_graph->getTensorNode(i);
	        Label* tlabel = tnode->getLabel();
	        tlabel->setNodeNameLabel(std::string("TensorNode"));
	        tlabel->setTypeNameLabel(std::string("Tensor"));

	        //SWLOG_INFO << tlabel->getNodeNameLabel() << std::endl;
	        //SWLOG_INFO << tlabel->getTypeNameLabel() << std::endl;

	    }
	    
	    for (int i = 0; i < nOpNodes; i++) {
	        OpNode<Dtype>* tnode = this->_graph->getOpNode(i);
	        Label* tlabel = tnode->getLabel();
	        tlabel->setNodeNameLabel(std::string("OpNode"));
	        tlabel->setTypeNameLabel((tnode->getOp())->getOpName());

	        //SWLOG_INFO << tlabel->getNodeNameLabel() << std::endl;
	        //SWLOG_INFO << tlabel->getTypeNameLabel() << std::endl;
	    }
	}

	void runTiling(){

		int nTensorNodes = this->_graph->tensorNodeNum();
    	int nOpNodes = this->_graph->opNodeNum();
    
	    for (int i = 0; i < nTensorNodes; i++) {
	        TensorNode<Dtype>* tnode = this->_graph->getTensorNode(i);
	        Label* tlabel = tnode->getLabel();
	        TilingLabel* tilinglabel = new TilingLabel(tlabel);
	        
	         


	        TileHint tilehint(tnode->getTensor()->getTensorShape(), 2);

	       	//tbd:tilehint 
	        tilehint.setSimpleTilingByDim(0);

	        






	        tilinglabel->setTilingLabel(tilehint);


	        //tslabel->setNodeNameLabel(std::string("TensorNode"));
	        //tlabel->setTypeNameLabel(std::string("Tensor"));

	        //SWLOG_INFO << tlabel->getNodeNameLabel() << std::endl;
	        //SWLOG_INFO << tlabel->getTypeNameLabel() << std::endl;

	    }
	    


	};
    void run(){
    	//std::cout<<"run tiling "<<std::endl;
        runTiling();
    }





};
}
#endif
