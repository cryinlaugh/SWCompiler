/*
 * OpNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef OPNODE_H_
#define OPNODE_H_

#include "IRNode.h"

#include "op/Op.h"
#include <sstream>


namespace swc {

template <typename Dtype>
class OpNode : public IRNode {
 public:
    OpNode() :  _op(NULL) {};
    explicit OpNode(const char name[]) : IRNode(OP_NODE, name) {};
    explicit OpNode(const char name[], Op<Dtype>* op) : IRNode(OP_NODE, name), _op(op) {};
    ~OpNode(){};

    void destroy(){
        // printf("free OpNode:%s\n", name().c_str());

        getOp()->destroy();
        getLabel()->destroy();
        // this->~OpNode();
    };

    void setOp(Op<Dtype>* op) {
        _op = op;
    }

    Op<Dtype>* getOp() {
        return _op;
    }

    const std::string getOpName() { return _op->getOpName(); }


    OpNode<Dtype>* clone() const;
    std::string toString() const;
    void setRunOnce() {run_once_=true;}
    bool runable() {
        bool run = run_;
        if(run_once_)
            run_ = false;
         return run;
    }

  private:
    Op<Dtype>* _op; 
    bool run_{true};
    bool run_once_{false};
};

/// must clone _op because destructed in ctor
template <typename Dtype>
OpNode<Dtype>* OpNode<Dtype>::clone() const{
    OpNode<Dtype>* opNode = new OpNode((name()+"_cp").c_str());
    opNode->setOp(_op->clone());
    return opNode;
}
template <typename Dtype>
std::string OpNode<Dtype>::toString() const {
    std::stringstream os;
    os << "OpNode " << name() << "\n"
        << "  op: " << _op->getOpName() << "\n"
        << "    nInput : " << _op->getnInput() << "\n"
        << "    nOutput: " << _op->getnOutput();
    return os.str();
}


} //namespace swc

#endif /* !OPNODE_H_ */
