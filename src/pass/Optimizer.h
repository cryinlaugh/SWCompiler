/*************************************************************************
	> File Name: optimizer.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 11 Dec 2018 07:31:15 AM UTC
 ************************************************************************/

#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

namespace swc{

//Forward declarations
template<typename Dtype> class IRGraph;

template<typename Dtype>
class Optimizer{
  public:
    Optimizer(IRGraph<Dtype>* graph):_graph(graph){};
    ~Optimizer(){};

    void runOptimizer();

    void initLabelingPass();

    void testLoweringLabelingPass(); 

    //LabelingPass
    void runLabelingPass(int type);
    
    //LoweringPass
    void runLoweringPass();
  
  private:
    IRGraph<Dtype>* _graph;
};

}
#endif
