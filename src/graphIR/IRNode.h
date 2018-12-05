/*
 * IRNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRNODE_H
#define IRNODE_H

#include <string>
#include "../common.h"

namespace swc {

class IRNode {

public:
	IRNode();
	IRNode(std::vector<IRNode*>* parentNodes,
	       std::vector<IRNode*>* childNodes,
	       std::string           name);

	~IRNode();

	void init(std::vector<IRNode*>* parentNodes,
	          std::vector<IRNode*>* childNodes,
	          std::string           name);

	void setParentNodes(std::vector<IRNode*>* parentNodes) {
		_parentNodes = parentNodes;
	}
	void setChildNodes(std::vector<IRNode*>* childNodes) {
		_childNodes = childNodes;
	}
	
	const std::vector<IRNode*>* getParentNodes() const { return _parentNodes; }
	const std::vector<IRNode*>* getChildNode()   const { return _childNodes;  }

	IRNode* getParentNode(int i) const { return (*_parentNodes)[i]; }
	IRNode* getChildNode(int i)  const { return (*_childNodes)[i];  }

	const std::string name() const { return _name; };
	void setName(std::string name) { _name = name; };

	inline const int parentNum() const { return (*_parentNodes).size(); }
	inline const int childNum()  const { return (*_childNodes).size();  }

	// dot generation
	std::string dotGen(); 


private:
	std::vector<IRNode*>* _parentNodes;
	std::vector<IRNode*>* _childNodes;
	std::string           _name;
};

} //namespace swc


#endif /* !IRNODE_H */
