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

using namespace swc::op;

namespace swc {

//Forward declaration
class StrategyLabel;

class OpNode : public IRNode {
  public:
    OpNode() : op_(NULL){};
    explicit OpNode(std::string name) : IRNode(OP_NODE, name){};
    explicit OpNode(std::string name, Op *op)
        : IRNode(OP_NODE, name), op_(op){};
    ~OpNode(){};

    void destroy() {
        // printf("free OpNode:%s\n", name().c_str());
        SWLOG_DEBUG(1) << "Destroy OpNodeL: " << name() << "\n"; 

        getOp()->destroy();
        getLabel()->destroy();
        // this->~OpNode();
    };

    void setOp(Op *op) { op_ = op; }

    Op *getOp() { return op_; }

    const std::string getOpName() { return op_->getOpName(); }

    OpNode *clone() const;
    OpNode *deepClone() const;
    std::string toString() const;
    void setRunOnce() { run_once_ = true; }
    bool runable() {
        bool run = run_;
        if (run_once_)
            run_ = false;
        return run;
    }

    void autoDiff(IRGraph* graph,
            std::unordered_map<IRNode*, IRNode*> &gradNodeMap){
        SWLOG_DEBUG(4) << "OpNode " << name() << " begin to autodiff" << std::endl;
        Op *_op = op_;
        _op->autoDiff(graph, this, gradNodeMap);
    };

    void checkValid() {
        Op *_op = op_;
        _op->checkValid(this);
        return;
    };

    void outTensorShapeGen(size_t index, TensorShape* tShape) {
        Op *_op = op_;
        _op->outTensorShapeGen(this, index, tShape);
    };

    void genOutTensor() const;

    void setStrategyLabel(StrategyLabel* strategyLabel){
        _strategyLabel = strategyLabel;
    }
    StrategyLabel* getStrategyLabel() { return _strategyLabel; }

  private:
    Op *op_;
    bool run_{true};
    bool run_once_{false};

    StrategyLabel* _strategyLabel{NULL};
};

} // namespace swc
#endif /* !OPNODE_H_ */
