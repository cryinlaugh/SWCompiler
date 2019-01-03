/*
 * dotGen.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-01-03
 */
#include "dotGen.h"

#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/IRGraph.h"

namespace swc {

std::string dotGenIRNode(IRNode* irnode,
                        std::string tensorInfo, 
                        std::string opInfo) {

    std::string              thisNode;
    std::vector<std::string> parentNodes;
    std::vector<std::string> childNodes;

    std::string str_tmp;
    std::string str_total;

    std::vector<std::string> NodeInfo;
    NodeInfo.push_back(tensorInfo);
    NodeInfo.push_back(opInfo); 

    // init parentNodes
    for (int i = 0; i < irnode->parentNum(); ++i) {
        parentNodes.push_back(irnode->getParentNode(i)->name());
    }

    // init childNodes
    for (int i = 0; i < irnode->childNum(); ++i) {
        childNodes.push_back(irnode->getChildNode(i)->name());
    }

    // Generate dot codes.
    str_total = str_total + "    // Generate one Node!\n";

    // Generate the information of this Node
    thisNode = irnode->name();

    NodeType nodeType = irnode->nodeType();

    if (nodeType == TENSOR_NODE) 
        str_tmp = "    " + thisNode + NodeInfo[0];
    else if (nodeType == OP_NODE)
        str_tmp = "    " + thisNode + NodeInfo[1];

    str_total = str_total + str_tmp;

    // Generate -> Children
    for (int i = 0; i < irnode->childNum(); ++i) {
        str_tmp   = "    " + thisNode + " -> " + childNodes[i] + ";\n";
        str_total = str_total + str_tmp;
    }

    str_total = str_total + "\n";

    return str_total;
}

std::string dotGenIRNode(IRNode* irnode) {
    return dotGenIRNode(irnode, " [shape = box];\n", ";\n");
}

template <typename Dtype>
std::string dotGenTensorNode(TensorNode<Dtype>* tnode) {

  std::string tensorInfo = " [shape = record, ";
  std::string tensorName = tnode->name();
  
  // get NDim through "getTensor()->getNDim()"
  int NDim = tnode->getTensor()->getNDim();    
  // generate the tensorInfo
  tensorInfo = tensorInfo + "label = \"{Name: " + tensorName + " |" ;
  tensorInfo = tensorInfo + "NDim: " + std::to_string(NDim) + " |"; 
  
  for (int i = 0; i < NDim; ++i) {
    if (i < NDim-1) { 
      tensorInfo = tensorInfo + "Dim[" + std::to_string(i) + "]:" 
          + std::to_string(tnode->getTensor()->getDim(i)) + " |";
    } else {         
      tensorInfo = tensorInfo + "Dim[" + std::to_string(i) + "]:" 
          + std::to_string(tnode->getTensor()->getDim(i)) + " }\"];\n";
    }
  }
  return dotGenIRNode(tnode, tensorInfo, ";\n");
}

template <typename Dtype>
std::string dotGenOpNode(OpNode<Dtype>* onode) {

  std::string opInfo  = " [";
  std::string opName  = onode->name();
  std::string opType = "BASIC_OP";

  if (onode->getOp()->getOpType() == BASIC_OP) 
    opType = "BASIC_OP";
  else if (onode->getOp()->getOpType() == DL_OP) 
    opType = "DL_OP";
  else if (onode->getOp()->getOpType() == TENSOR_OP) 
    opType = "TENSOR_OP";

  int nInput    = onode->getOp()->getnInput();
  int nOutput   = onode->getOp()->getnOutput();

  // generate the opInfo
  opInfo = opInfo + "label = \"Name: " + opName + "\\nOpType: " + opType + "\\n" ;
  opInfo = opInfo + "_nInput: "  + std::to_string(nInput)  + "\\n";
  opInfo = opInfo + "_nOutput: " + std::to_string(nOutput) + "\"];\n";

  // return opInfo;

  return dotGenIRNode(onode, " [shape = box];\n", opInfo);
}

template<typename Dtype>
void dotGen(IRGraph<Dtype>* graph, std::string dotFileName) {

    std::cout << "Generate the dotFile for drawing." << std::endl;

    std::string dot_Total;

	for (int i = 0; i < graph->tensorNodeNum(); i++) {
		dot_Total = dot_Total + dotGenTensorNode(graph->getTensorNode(i));
	}

	for (int i = 0; i < graph->opNodeNum(); i++) {
		dot_Total = dot_Total + dotGenOpNode(graph->getOpNode(i));
	}

    std::string dot_title = "digraph CG { \n";
    std::string dot_end   = "\n}";

	// dotFile Genrate
    std::ofstream dotfile(dotFileName, std::fstream::out);

	dotfile << dot_title << std::endl;
	dotfile << dot_Total;
	dotfile << dot_end << std::endl;

	// make PNG
    std::string svgFileName = "IRGraph.svg";
    std::string dotGenCMD   = "dot -T svg " + dotFileName + " -o " + svgFileName;

	char *cmd = (char*)dotGenCMD.data();

	if (system(cmd) == 0);
}

template<>
void dotGen(IRGraph<double>* graph) {
	dotGen(graph, "IRGraph.dot");
}

template<>
void dotGen(IRGraph<float>* graph) {
	dotGen(graph, "IRGraph.dot");
}


} // namespace swc
