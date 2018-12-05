/*
 * IRNode.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-11-23
 */

#include "IRNode.h"
// #include <string>

namespace swc {

IRNode::IRNode() {
	_parentNodes = NULL;
	_childNodes = NULL;
}

IRNode::IRNode(std::vector<IRNode*>* parentNodes,
               std::vector<IRNode*>* childNodes,
               std::string           name) 
{
	_parentNodes = parentNodes;
	_childNodes  = childNodes;
	_name        = name;
}

IRNode::~IRNode() {
	printf("free:%s\n", _name.c_str());
}

void IRNode::init(std::vector<IRNode*>* parentNodes,
                  std::vector<IRNode*>* childNodes,
                  std::string           name) 
{
	_parentNodes = parentNodes;
	_childNodes  = childNodes;
	_name        = name;
}

std::string IRNode::dotGen() {

	std::vector<std::string> parentNodes;
	std::vector<std::string> childNodes;
	
	std::string str_tmp;
	std::string str_total;
	std::string thisNode;
	std::string NodeEx[2] = { ";\n", " [shape = box];\n" };

	// init parentNodes
	for (int i = 0; i < parentNum(); ++i) {
		// str_tmp = to_string(i);
		parentNodes.push_back(getParentNode(i)->name());
	}

	// init childNodes
	for (int i = 0; i < childNum(); ++i) {
		// str_tmp = to_string(i);
		childNodes.push_back(getChildNode(i)->name());
	}

	// init thisNode
	thisNode = name();

	// Generate dot codes.
	str_total = str_total + "	// Generate one Node!\n";

	// 生成不同 parentNodess 的类型 (Tensor or OP)
	for (int i = 0; i < parentNum(); ++i) {

		if (parentNodes[i].substr(0,1) == "O") 
			str_tmp = "	" + parentNodes[i] + NodeEx[0];
		else if (parentNodes[i].substr(0,1) == "T")
			str_tmp = "	" + parentNodes[i] + NodeEx[1];

		str_total = str_total + str_tmp;
	}

	// str_total = str_total + "\n";

	// 生成不同 childNodess 的类型 (Tensor or OP)
	for (int i = 0; i < childNum(); ++i) {

		if (childNodes[i].substr(0,1) == "O") 
			str_tmp = "	" + childNodes[i] + NodeEx[0];
		else if (childNodes[i].substr(0,1) == "T")
			str_tmp = "	" + childNodes[i] + NodeEx[1];

		str_total = str_total + str_tmp;
	}

	str_total = str_total + "\n";

	// // 生成 Father -> This
	// for (int i = 0; i < parentNum(); ++i) {
	// 	str_tmp   = "	" + parentNodes[i] + " -> " + thisNode + ";\n";
	// 	str_total = str_total + str_tmp;
	// }	

	// str_total = str_total + "\n";

	// 生成 This   -> Children
	for (int i = 0; i < childNum(); ++i) {
		str_tmp   = "	" + thisNode + " -> " + childNodes[i] + ";\n";
		str_total = str_total + str_tmp;
	}

	str_total = str_total + "\n";
	
	return str_total;
}

} //namespace swc
