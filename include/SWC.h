/*************************************************************************
	> File Name: SWC.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Wed 05 Dec 2018 03:37:36 AM UTC
 ************************************************************************/

#ifndef _SWC_H
#define _SWC_H

#include "../src/SWDSL.h"
#include "../src/SWLOG.h"
#include "../src/common.h"

#include "../src/op/dlOp/dlOp.h"
#include "../src/op/tensorOp/tensorOps.h"
#include "../src/tensor/tensor.h"

#include "../src/graphIR/IRGraph.h"
#include "../src/graphIR/OpNode.h"
#include "../src/graphIR/TensorNode.h"

#include "../src/pass/Label.h"
#include "../src/pass/Optimizer.h"

#include "../src/tool/dotGen.h"

#include "../src/diff/AutoDiff.h"


#include "../src/codegen/Codegen.h"

#include "../src/engine/Engine.h"

#include "../src/pass/OptimizePass.h"
#include "../src/pass/LabelingPass.h"
#include "../src/pass/LoweringPass.h"
#include "../src/pass/EliminationPass.h"
#include "../src/pass/RenamingNodePass.h"
// #include "../src/pass/ParallelingPass.h"
#include "../src/pass/ParallelLabelingPass.h"
#include "../src/pass/ParallelLoweringPass.h"
#include "../src/pass/AutodiffPass.h"

#endif
