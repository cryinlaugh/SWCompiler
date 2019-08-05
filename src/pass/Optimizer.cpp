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

namespace swc {

void pass::Optimizer::runOptimizer() {
    // SWLOG_INFO << "Start doing optimization." << std::endl;
    // runLabelingPass(0);
    // runLoweringPass();
    // runLabelingPass(0);

    SWLOG_DEBUG(4) << "Start doing optimization.\n" << std::endl;
    PassManager passManager;
    RenamingNodePass renamingpass(_graph);
    passManager.add((OptimizePass *)&renamingpass);

    //lowering pass: dlop to basic op , or dlop to paralleled code ,so that all ops can be paralleled
    LabelingPass labelingpass(_graph);
    passManager.add((OptimizePass *)&labelingpass);
    LoweringPass loweringpass(_graph);
    passManager.add((OptimizePass *)&loweringpass);
    passManager.add((OptimizePass *)&labelingpass); // run labeling again

    // paralleing pass: assign tiling label to all ops , then applied  to graph
    ParallelingPass parallelingpass(_graph);
    //passManager.add((OptimizePass*)&parallelingpass);
    passManager.run();
    SWLOG_DEBUG(4) << "Optimization done.\n\n" << std::endl;
}
} // namespace swc
