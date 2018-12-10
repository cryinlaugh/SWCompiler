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

	// Generate Types (Tensor or OP)
	for (int i = 0; i < parentNum(); ++i) {

		if (_nodeType == TENSOR_NODE) 
			str_tmp = "	" + parentNodes[i] + NodeEx[0];
		else if (_nodeType == OP_NODE)
			str_tmp = "	" + parentNodes[i] + NodeEx[1];

		str_total = str_total + str_tmp;
	}

	// str_total = str_total + "\n";

	// Generate Types (Tensor or OP)
	for (int i = 0; i < childNum(); ++i) {

		if (_nodeType == TENSOR_NODE) 
			str_tmp = "	" + childNodes[i] + NodeEx[0];
		else if (_nodeType == OP_NODE)
			str_tmp = "	" + childNodes[i] + NodeEx[1];

		str_total = str_total + str_tmp;
	}

	str_total = str_total + "\n";

	// // Generate Father -> This
	// for (int i = 0; i < parentNum(); ++i) {
	// 	str_tmp   = "	" + parentNodes[i] + " -> " + thisNode + ";\n";
	// 	str_total = str_total + str_tmp;
	// }	

	// str_total = str_total + "\n";

	// Generate -> Children
	for (int i = 0; i < childNum(); ++i) {
		str_tmp   = "	" + thisNode + " -> " + childNodes[i] + ";\n";
		str_total = str_total + str_tmp;
	}

	str_total = str_total + "\n";
	
	return str_total;
}

} //namespace swc
