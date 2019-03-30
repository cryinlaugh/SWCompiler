/***********************************************
#
#      Filename: OptimizePass.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-01-23 10:57:27
# Last Modified: 2019-03-27 10:57:27
***********************************************/


#ifndef _OPTIMIZEPASS_H
#define _OPTIMIZEPASS_H
#include "../src/graphIR/IRGraph.h"
#include <queue>
namespace swc{

//Forward declarations
//template<typename Dtype> class IRGraph;
template<typename Dtype>  
class OptimizePass{
protected:
    
    IRGraph<Dtype>* _graph;
public:
    OptimizePass(IRGraph<Dtype> * graph):_graph(graph){};
    ~OptimizePass(){};
    virtual void run();

};


}
#endif
