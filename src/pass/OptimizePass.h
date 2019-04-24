/*************************************************************************
    > File Name: optimizer.h
    > Author: cryinlaugh
    > Mail: cryinlaugh@gmail.com
    > Created Time: Tue 11 Dec 2018 07:31:15 AM UTC
 ************************************************************************/

#ifndef _OPTIMIZEPASS_H
#define _OPTIMIZEPASS_H
#include "graphIR/IRGraph.h"
#include <queue>
namespace swc {
namespace pass {

class OptimizePass {
  protected:
    IRGraph *_graph;

  public:
    OptimizePass(IRGraph *graph) : _graph(graph){};
    ~OptimizePass(){};
    void setGraph(IRGraph *graph) { _graph = graph; }
    virtual void run() { std::cout << "father run " << std::endl; }
};

} // namespace pass
} // namespace swc
#endif
