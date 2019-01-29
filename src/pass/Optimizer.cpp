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
    SWLOG_INFO << "Start doing optimization." << std::endl;
    PassManager<Dtype> passManager;
    LabelingPass<Dtype> labelingpass(_graph);
    passManager.add((OptimizePass<Dtype>*)&labelingpass);
    LoweringPass<Dtype> loweringpass(_graph);
    passManager.add((OptimizePass<Dtype>*)&loweringpass);
    passManager.run();
    SWLOG_INFO << "Optimization done." << std::endl;

}

// template<typename Dtype>
// void Optimizer<Dtype>::initLabelingPass() {
//     //check and init Label in each node
//     int nTensorNodes = _graph->tensorNodeNum();
//     int nOpNodes = _graph->opNodeNum();

//     for (int i = 0; i < nTensorNodes; i++) {
//         TensorNode<Dtype>* tnode = _graph->getTensorNode(i);
//         Label* tlabel = tnode->getLabel();
//         tlabel->setNodeNameLabel(std::string("TensorNode"));
//         tlabel->setTypeNameLabel(std::string("Tensor"));

//         SWLOG_INFO << tlabel->getNodeNameLabel() << std::endl;
//         SWLOG_INFO << tlabel->getTypeNameLabel() << std::endl;

//     }
    
//     for (int i = 0; i < nOpNodes; i++) {
//         OpNode<Dtype>* tnode = _graph->getOpNode(i);
//         Label* tlabel = tnode->getLabel();
//         tlabel->setNodeNameLabel(std::string("OpNode"));
//         tlabel->setTypeNameLabel((tnode->getOp())->getOpName());

//         SWLOG_INFO << tlabel->getNodeNameLabel() << std::endl;
//         SWLOG_INFO << tlabel->getTypeNameLabel() << std::endl;
//     }
// }

// template<typename Dtype>
// void Optimizer<Dtype>::testLoweringLabelingPass() {
//     int nTensorNodes = _graph->tensorNodeNum();
//     int nOpNodes = _graph->opNodeNum();

//     for (int i = 0; i < nTensorNodes; i++) {
//         TensorNode<Dtype>* tnode = _graph->getTensorNode(i);
//         Label* tlabel = tnode->getLabel();
//         //do nothing for tensor nodes
//         SWLOG_INFO << "Do nothing for " << tlabel->getTypeNameLabel() 
//                     << " ." << std::endl;
//     }

//     for (int i = 0; i < nOpNodes; i++) {
//         OpNode<Dtype>* tnode = _graph->getOpNode(i);
//         Label* tlabel = tnode->getLabel();
//         if ((tlabel->getTypeNameLabel()).compare("MatrixMatrixFC") == 0) {
//             SWLOG_INFO << tlabel->getTypeNameLabel() 
//                         << " operator is marked to be lowered." << std::endl;
//             tlabel->setLowerMark();
//         } else {
//             SWLOG_INFO << "Do nothing for " << tlabel->getTypeNameLabel() 
//                         << " operator." << std::endl;
//         }
//     }
// }

// template<typename Dtype>
// void Optimizer<Dtype>::runLabelingPass(int type) {
//     SWLOG_INFO << "Start initial labeling pass: " << std::endl;
//     initLabelingPass();
//     SWLOG_INFO << "Finish initial labeling pass: "  << std::endl;
//     if (type == 0) {
//         SWLOG_INFO << "Start test labeling pass: " << std::endl;
//         testLoweringLabelingPass();
//         SWLOG_INFO << "Finish test labeling pass: "  << std::endl;
//     }
// }

// template<typename Dtype>
// void Optimizer<Dtype>::runLoweringPass() {
//   SWLOG_INFO << "Start lowering pass: " << std::endl;

//   int nTensorNodes = _graph->tensorNodeNum();
//   int nOpNodes = _graph->opNodeNum();

//   for (int i = 0; i < nTensorNodes; i++) {
//     TensorNode<Dtype>* tnode = _graph->getTensorNode(i);
//     Label* tlabel = tnode->getLabel();
//     //do nothing for tensor nodes
//     SWLOG_INFO << "Do nothing for " << tlabel->getTypeNameLabel() 
//                 << " ." << std::endl;
//   }

//   for (int i = 0; i < nOpNodes; i++) {
//     OpNode<Dtype>* tnode = _graph->getOpNode(i);
//     Label* tlabel = tnode->getLabel();
//     if(tlabel->getLowerMark()) {
//       tnode->getOp()->lowering(_graph, tnode);
//     }
//   }

//   SWLOG_INFO << "Finish lowering pass. " << std::endl;

// }


INSTANTIATE_CLASS(Optimizer);

} // namespace swc
