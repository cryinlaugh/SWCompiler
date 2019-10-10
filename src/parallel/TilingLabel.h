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
#include <map>
#include <sstream>

namespace swc {


class TilingLabel {
private:
    TensorNode * _currentnode;
    int _currentstrategy;
    // true: TensorNode this belong to was parallelized at least once  
    //  and _currentnode point to para-tnode (with strategy -2, -1 or i)
    bool  _isApplied{false};

    // we expand TilingLabel to holds all history parallelization strategies
    // for a TensorNode; and the map between strategy and para-tnode
    std::map<int, TensorNode*> _strategy_parnode_map; 
    

public:

    TilingLabel(){}

    bool strategyExist(int strategy) {
        return _strategy_parnode_map.count(strategy);
    }
    
    size_t strategySize() { return _strategy_parnode_map.size(); }

    TensorNode* getStrategyParNode(int strategy) {
        return _strategy_parnode_map.at(strategy);
    }

    void insertStrategy(int strategy, TensorNode *par_tnode) {
        _strategy_parnode_map[strategy] = par_tnode;
    }

    // guarantee: has history strategies 
    int selectTransPreStrategy(int strategy=0) {
        // we may use strategy later
        (void)strategy;
        std::ostringstream os;
        os << "history strategy (";

        // at lease the newest strategy
        int best = _currentstrategy;
        for(auto iter : _strategy_parnode_map) {
            os << iter.first <<  ", ";
            if(iter.first < best)
                best = iter.first;
        }

        os << ") select " << best;
        SWLOG_DEBUG(4) << os.str() << "\n"; 

        return best;
    }

    void setCurrentNode(TensorNode* tensornode){
        _currentnode = tensornode;
    }

    TensorNode *  getCurrentNode(){
        return _currentnode;
    }

    // strategy is the parallel-axis of tensor
    // but mind that -2:reduce, -1:replicate, i:cut
    void setCurrentStrategy(int strategy){
        _currentstrategy = strategy;
    }
    int  getCurrentStrategy(){
        return _currentstrategy;
    }

    ~TilingLabel() {};
    bool isApplied() {
        return _isApplied;
    }
    void setApplied(bool flag=true) {
        _isApplied = flag;
    }
};


// StrategyLabel holds one possible parallelization strategy for an OpNode
// e.g. for MatMul, this vector might be {0, -1, 0}
// _strategy.size() = nInputTensorNode + nOutputTensorNode
class StrategyLabel {
private:

    std::vector<int> _strategy;
public:

    StrategyLabel(){}
    StrategyLabel(std::vector<int> strategy){
        _strategy=strategy;
    }
    ~StrategyLabel(){}

    void setStrategy(std::vector<int> strategy){

        _strategy= strategy;
    }
    std::vector<int> getStrategy(){
        return _strategy;
    }
};
} // namespace swc

#endif
