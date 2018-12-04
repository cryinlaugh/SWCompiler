/*
 * OpNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef OPNODE_H
#define OPNODE_H

#include "../basicOp/Op.h"

namespace swc {

class OpNode : public IRNode 
{
  
  public:
    OpNode();
    ~OpNode();

    void setOperation(Operation<Dtype>* op) {
      _op = op;
    }

    Operation& Operation() {
      return _op;
    }

  private:
    Operation<Dtype>* _op; 
}


#endif /* !OPNODE_H */
