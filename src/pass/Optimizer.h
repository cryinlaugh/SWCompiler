/*************************************************************************
	> File Name: optimizer.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 11 Dec 2018 07:31:15 AM UTC
 ************************************************************************/

#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include "LabelingPass.h"
#include "LoweringPass.h"
#include "../src/graphIR/IRGraph.h"
#include <queue>
namespace swc{

template<typename Dtype>
class PassManager{
private:
    std::queue<OptimizePass<Dtype>*> passQueue;
public :
    PassManager(){};
    ~PassManager(){};
    void add(OptimizePass<Dtype>* pass){
        passQueue.push(pass);
    }
    void run(){
        while(!passQueue.empty()){
            OptimizePass<Dtype>*  pass=passQueue.front();
            pass->run();
            passQueue.pop();
        }
    }
};
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
