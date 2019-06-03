/*
 * AutodiffPass.h
 * Copyright (C) 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef AUTODIFFPASS_H
#define AUTODIFFPASS_H

#include "graphIR/IRGraph.h"

namespace swc {
namespace pass {

/**
 * @breif AutodiffPass to do the auto differential of the
 * original net and generate a training network.
 */
class AutodiffPass {
  private:
    IRGraph *_graph;
  
  public:
    AutodiffPass(IRGraph *graph){ _graph = graph; };
    ~AutodiffPass(){};

    void getMethod();
    void run();
};

}  //namespace pass 
}  //namespace swc 



#endif /* !AUTODIFFPASS_H */
