/*
 * OpNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef OPNODE_H_
#define OPNODE_H_

#include "IRNode.h"

#include "basicOp/Op.h"

namespace swc {

template <typename Dtype>
class OpNode : public IRNode {
 public:
    OpNode() :  _op(NULL) {};
    explicit OpNode(const char name[]) : IRNode(OP_NODE, name) {};
    ~OpNode(){};

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

#endif /* !OPNODE_H_ */
