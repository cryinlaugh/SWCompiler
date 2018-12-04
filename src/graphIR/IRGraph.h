/*
 * IRGraph.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRGRAPH_H
#define IRGRAPH_H

#include "IRNode.h"
#include "IRGraph.h"

namespace swc {

template <typename Dtype>
class IRGraph 
{
  public:
    IRGraph();
    ~IRGraph();

    setFatherNode(vector<IRNode*> fatherNode) {
      _fatherNode = fatherNode;
    }
    setChildNode(vector<IRNode*> ChildNode) {
      _childNode = ChildNode;
    }
    IRNode* getFatherNode(int i) const{
      return _fatherNode[i];
    }
    IRNode* getChildNode(int i) const{
      return _childNode[i];
    }

  private:
    vector<TensorNode<Dtype> * > _tensors;
    vector<OPNode<Dtype> * > _operations;

}

} //namespace swc

#endif /* !IRGRAPH_H */
