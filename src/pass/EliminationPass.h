/*
 * EliminationPass.h_
 * Copyright (C) 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef ELIMINATIONPASS_H_
#define ELIMINATIONPASS_H_

#include "OptimizePass.h"
#include "SWLOG.h"

namespace swc {

// Forward declarations
class IRGraph;

namespace pass {

class EliminationPass : public OptimizePass {

  public:
    EliminationPass(IRGraph *graph) : OptimizePass(graph){};
    ~EliminationPass() { destroy(); };

    void run();
    void destroy();

};

}  //namespace pass
}  //namespace swc



#endif /* !ELIMINATIONPASS_H_ */
