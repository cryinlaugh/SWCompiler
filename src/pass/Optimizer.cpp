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

    SWLOG_DEBUG(4) << "Start doing optimization." << std::endl;
    PassManager passManager;
    RenamingNodePass renamingpass(_graph);
    passManager.add((OptimizePass *)&renamingpass);
    
    //lowering pass: dlop to basic op , or dlop to paralleled code ,so that all ops can be paralleled   
    LabelingPass labelingpass(_graph);
    passManager.add((OptimizePass *)&labelingpass);
    LoweringPass loweringpass(_graph);
    passManager.add((OptimizePass *)&loweringpass);
    passManager.add((OptimizePass *)&labelingpass); // run labeling again

    
    //ParallelingPass parallelingpass(_graph);
    //passManager.add((OptimizePass*)&parallelingpass);
    

    //RenamingNodePass renamingpass2(_graph);
    //passManager.add((OptimizePass*)&renamingpass2);

    //EliminationPass elim(_graph);
    //passManager.add((OptimizePass*)&elim);

    //SubGraphPass sub(_graph);
    //passManager.add((OptimizePass*)&sub);
    
    passManager.run();
    SWLOG_DEBUG(4) << "Optimization done." << std::endl;
}
} // namespace swc
