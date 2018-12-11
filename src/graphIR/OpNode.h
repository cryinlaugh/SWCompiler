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
    OpNode() :  _op(NULL) {};
    OpNode(const char name[]) : IRNode(OP_NODE, name) {};
    ~OpNode(){};

    void setOp(Op<Dtype>* op) {
      _op = op;
    }

    Op<Dtype>* getOp() {
      return _op;
    }

    std::string dotGen();

  private:
    Op<Dtype>* _op; 
};

template <typename Dtype>
std::string OpNode<Dtype>::dotGen() {

  std::string opInfo  = " [";
  std::string opName  = name();
  std::string opType = "BASIC_OP";

  if (_op->getOpType() == BASIC_OP) 
    opType = "BASIC_OP";
  else if (_op->getOpType() == DL_OP) 
    opType = "DL_OP";
  else if (_op->getOpType() == TENSOR_OP) 
    opType = "TENSOR_OP";

  int nInput    = _op->getnInput();
  int nOutput   = _op->getnOutput();

  // generate the opInfo
  opInfo = opInfo + "label = \"Name: " + opName + "\\nOpType: " + opType + "\\n" ;
  opInfo = opInfo + "_nInput: "  + std::to_string(nInput)  + "\\n";
  opInfo = opInfo + "_nOutput: " + std::to_string(nOutput) + "\"];\n";

  // return opInfo;

  return IRNode::dotGen(" [shape = box];\n", opInfo);
}

} //namespace swc

#endif /* !OPNODE_H */
