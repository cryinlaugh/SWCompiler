/*************************************************************************
	> File Name: optimizer.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 11 Dec 2018 07:31:15 AM UTC
 ************************************************************************/

#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include "IRGraph.h"
#include "SWLOG.h"

namespace swc{

template <typename Dtype>
class Optimizer{
private:
    
public:
    Optimizer(){};
    ~Optimizer(){};

    void runOptimize(IRGraph<Dtype>* graph){
        SWLOG_INFO << "this is log"<<std::endl;
    }
};

}
#endif
