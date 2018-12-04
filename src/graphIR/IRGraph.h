/*
 * IRGraph.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRGRAPH_H
#define IRGRAPH_H

#include <vector>
#include "TensorNode.h"
#include "OpNode.h"

namespace swc {

template<typename Dtype>
class IRGraph 
{
  public:
    IRGraph(){};
    ~IRGraph(){};

    TensorNode<Dtype>* getTensorNode(int i) const { return _tensors[i]; };
    OpNode<Dtype>* getOpNode(int i) const { return _ops[i]; };
    
    void pushTensorNode(TensorNode<Dtype> *t) { _tensors.push_back(t); };
    void pushOpNode(OpNode<Dtype> *o) { _ops.push_back(o); };

    int ternsorNodeNum() { return _tensors.size(); };
    int opNodeNum() { return _ops.size(); }
    
    void setTopology() {};

  private:
    std::vector<TensorNode<Dtype>* > _tensors;
    std::vector<OpNode<Dtype>* > _ops;
};

} //namespace swc

#endif /* !IRGRAPH_H */
