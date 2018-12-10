/*
 * IRNode.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-11-23
 */

#include "IRNode.h"

namespace swc {

std::string IRNode::dotGen(std::string tensorInfo, std::string opInfo) {

	std::string              thisNode;
	std::vector<std::string> parentNodes;
	std::vector<std::string> childNodes;
	
	std::string str_tmp;
	std::string str_total;

	std::vector<std::string> NodeInfo;
	NodeInfo.push_back(tensorInfo);
	NodeInfo.push_back(opInfo);	

	// init parentNodes
	for (int i = 0; i < parentNum(); ++i) {
		parentNodes.push_back(getParentNode(i)->name());
	}

	// init childNodes
	for (int i = 0; i < childNum(); ++i) {
		childNodes.push_back(getChildNode(i)->name());
	}

	// Generate dot codes.
	str_total = str_total + "	// Generate one Node!\n";

	// Generate the information of this Node
	thisNode = name();

	if (_nodeType == TENSOR_NODE) 
		str_tmp = "	" + thisNode + NodeInfo[0];
	else if (_nodeType == OP_NODE)
		str_tmp = "	" + thisNode + NodeInfo[1];

	str_total = str_total + str_tmp;

	// Generate -> Children
	for (int i = 0; i < childNum(); ++i) {
		str_tmp   = "	" + thisNode + " -> " + childNodes[i] + ";\n";
		str_total = str_total + str_tmp;
	}

	str_total = str_total + "\n";
	
	return str_total;
}

std::string IRNode::dotGen() {

	return IRNode::dotGen(" [shape = box];\n", ";\n");
}

}

// namespace swc
