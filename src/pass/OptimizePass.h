/*************************************************************************
    > File Name: optimizer.h
    > Author: cryinlaugh
    > Mail: cryinlaugh@gmail.com
    > Created Time: Tue 11 Dec 2018 07:31:15 AM UTC
 ************************************************************************/

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
    virtual void run(){

        std::cout<<"father run "<<std::endl;
    }


};

}
#endif
