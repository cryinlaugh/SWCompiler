/*
 * Optimizer.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-01-03
 */

#include "Optimizer.h"

#include "SWLOG.h"
#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/IRGraph.h"

namespace swc {

template<typename Dtype>
void Optimizer<Dtype>::runOptimizer() {
	// SWLOG_INFO << "Start doing optimization." << std::endl;
    // runLabelingPass(0);
    // runLoweringPass();
    // runLabelingPass(0);
    
    SWLOG_DEBUG << "Start doing optimization." << std::endl;
    PassManager<Dtype> passManager;
    RenamingNodePass<Dtype> renamingpass(_graph);
    passManager.add((OptimizePass<Dtype>*)&renamingpass);
    LabelingPass<Dtype> labelingpass(_graph);
    passManager.add((OptimizePass<Dtype>*)&labelingpass);
    LoweringPass<Dtype> loweringpass(_graph);
    passManager.add((OptimizePass<Dtype>*)&loweringpass);
    passManager.add((OptimizePass<Dtype>*)&labelingpass); // run labeling again
    passManager.run();
    SWLOG_INFO << "Optimization done." << std::endl;
    

}

INSTANTIATE_CLASS(Optimizer);
} // namespace swc
