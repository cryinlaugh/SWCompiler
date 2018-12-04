/*
 * OpNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef OPNODE_H
#define OPNODE_H

#include "operation.h"

namespace swc {

template <typename Dtype>
class OpNode : public IRNode<Dtype> 
{
  
  public:
    OpNode();
    ~OpNode();

    void setOperation(Operation<Dtype> op) {
      _op = op;
    }

    Operation& Operation() {
      return _op;
    }

  private:
    Operation<Dtype> _op; 
}


#endif /* !OPNODE_H */
