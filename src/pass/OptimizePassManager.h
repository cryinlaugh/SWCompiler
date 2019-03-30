/***********************************************
#
#      Filename: OpitmizePassManager.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-03-27 15:06:44
# Last Modified: 2019-03-27 15:06:44
***********************************************/
#ifndef _OPTIMIZERPASSMANAGER_H
#define _OPTIMIZERPASSMANAGER_H

#include <queue>
#include "OptimizePass.h"
namespace swc{

template<typename Dtype>
class OptimizePassManager{
private:
    std::queue<OptimizePass<Dtype>*> passQueue;
public :
    OptimizePassManager(){};
    ~OptimizePassManager(){};
    void add(OptimizePass<Dtype>* pass){
        passQueue.push(pass);
    }
    
    void run(){
        while(!passQueue.empty()){
            OptimizePass<Dtype>*  pass=passQueue.front();
            pass->run();
            passQueue.pop();
        }
    }
    
};
}
#endif
