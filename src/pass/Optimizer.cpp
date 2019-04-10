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

void Optimizer::runOptimizer() {
	// SWLOG_INFO << "Start doing optimization." << std::endl;
    // runLabelingPass(0);
    // runLoweringPass();
    // runLabelingPass(0);
    
    SWLOG_DEBUG << "Start doing optimization." << std::endl;
    PassManager passManager;
    RenamingNodePass renamingpass(_graph);
    passManager.add((OptimizePass*)&renamingpass);
    LabelingPass labelingpass(_graph);
    passManager.add((OptimizePass*)&labelingpass);
    LoweringPass loweringpass(_graph);
    passManager.add((OptimizePass*)&loweringpass);
    passManager.add((OptimizePass*)&labelingpass); // run labeling again
    passManager.run();
    SWLOG_INFO << "Optimization done." << std::endl;
    

}
} // namespace swc
