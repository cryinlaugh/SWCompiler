/***********************************************
#
#      Filename: TilingLable.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-01-21 17:54:42
# Last Modified: 2019-01-21 17:54:42
***********************************************/

#ifndef _TILELABEL_H
#define _TILELABEL_H

#include "../graphIR/TensorNode.h"
#include <algorithm>
#include <string>
#include <vector>
namespace swc {

class TilingLabel {
private:

    int _scope;
    TensorNode * _currentnode;
    int _currentstrategy;
    int  _isApplied;

public:

    TilingLabel() {
        _isApplied = false;
    };

    void init(int scope) {
        _scope = scope;
        _isApplied = 0;

    }

    int getScope() {
        return _scope;
    }
    void setCurrentNode(TensorNode* tensornode) {
        _currentnode = tensornode;
    }


    TensorNode *  getCurrentNode() {
        return _currentnode;
    }
    void setCurrentStrategy(int EinSum) {
        _currentstrategy = EinSum;
    }

    int  getCurrentStrategy() {
        return _currentstrategy;
    }
    ~TilingLabel() {};
    int isApplied() {
        //std::cout<<"test1"<<std::endl;
        return _isApplied;
    }
    void setApplied(int status) {
        _isApplied = status;
    }



};


class  StrategyLabel {

private:

    std::vector<int> _strategy;
public:

    StrategyLabel(std::vector<int> strategy) {
        _strategy = strategy;
    }
    ~StrategyLabel() {};

    void setStrategy(std::vector<int> strategy) {

        _strategy = strategy;
    }
    std::vector<int> getStrategy() {
        return _strategy;
    }


};
} // namespace swc

#endif
