/*
 * OpNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef OPNODE_H
#define OPNODE_H

#include "IRNode.h"
#include "../basicOp/Op.h"

namespace swc {

template <typename Dtype>
class OpNode : public IRNode
{
  
  public:
    OpNode();
    ~OpNode();

    void setOp(Op<Dtype>* op) {
      _op = op;
    }

    Op<Dtype>* getOp() {
      return _op;
    }

  private:
    Op<Dtype>* _op; 
};

} //namespace swc

#endif /* !OPNODE_H */
