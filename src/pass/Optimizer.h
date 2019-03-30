/***********************************************
#
#      Filename: Optimizer.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-01-23 10:57:27
# Last Modified: 2019-03-27 10:57:27
***********************************************/


#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include <queue>
#include "../src/graphIR/IRGraph.h"
#include "OptimizePassManager.h"

namespace swc{
template<typename Dtype>
class Optimizer{
  public:
    Optimizer(IRGraph<Dtype>* graph):_graph(graph){};
    ~Optimizer(){};
    void addPass(OptimizePass<Dtype>* pass);
    void run();
  private:
    IRGraph<Dtype>* _graph;
    OptimizePassManager<Dtype> passManager;
};

}
#endif
