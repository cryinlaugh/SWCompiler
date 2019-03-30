/***********************************************
#
#      Filename: Optimizer.cpp
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-01-23 10:57:27
# Last Modified: 2019-03-27 10:57:27
***********************************************/

#include "Optimizer.h"

#include "SWLOG.h"
#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/IRGraph.h"

namespace swc {

template<typename Dtype>
void Optimizer<Dtype>::addPass(OptimizePass<Dtype>* pass){
    passManager.add(pass);
}


template<typename Dtype>
void Optimizer<Dtype>::run(){
    SWLOG_INFO << "Start doing optimization." << std::endl;
    passManager.run();
    SWLOG_INFO << "Optimization done." << std::endl;

}

INSTANTIATE_CLASS(Optimizer);

} // namespace swc
