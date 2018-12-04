/*
 * IRGraph.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRGRAPH_H
#define IRGRAPH_H

#include <iostream>


#include "TensorNode.h"
#include "OpNode.h"
#include "IRGraph.h"

namespace swc {

class IRGraph 
{
  public:
    IRGraph();
    ~IRGraph();

    TensorNode* getTensorNode(int i) const { return _tensors[i] };
    OPNode* getOpNode(int i) const { return _operations[i] };
    
    void pushTensorNode(TensorNode *t) { _tensors.push_back(t) };
    void pushOpNode(OpNode *o) { _operations.push_back(o) };

    void setTopology();

  private:
    std::vector<TensorNode* > _tensors;
    std::vector<OpNode* > _operations;
}

} //namespace swc

#endif /* !IRGRAPH_H */
