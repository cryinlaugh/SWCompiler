/*************************************************************************
	> File Name: optimizer.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 11 Dec 2018 07:31:15 AM UTC
 ************************************************************************/

#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include <queue>

#include "LabelingPass.h"
#include "LoweringPass.h"
#include "RenamingNodePass.h"
#include "AutodiffPass.h"

namespace swc {

// Forward declarations
class IRGraph;
    
namespace pass {

class PassManager {
  private:
    std::queue<OptimizePass *> passQueue;

  public:
    PassManager(){};
    ~PassManager(){};
    void add(OptimizePass *pass) { passQueue.push(pass); }
    void run() {
        while (!passQueue.empty()) {
            OptimizePass *pass = passQueue.front();
            pass->run();
            passQueue.pop();
        }
    }
};

class Optimizer {
  public:
    Optimizer(IRGraph *graph) : _graph(graph){};
    ~Optimizer(){};
    void runOptimizer();

    void setGraph(IRGraph *graph) { _graph = graph; }

  private:
    IRGraph *_graph;
};

} // namespace pass
} // namespace swc
#endif
