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

class IRNode
{
  public:
    
    IRNode();
    IRNode(const NodeType nodeType, const char name[]) : 
      _name(std::string(name)), _nodeType(nodeType) {};
    ~IRNode() { printf("free:%s\n", _name.c_str()); };
    
    void pushParentNode() {};
    template<typename T, typename... Types> 
    void pushParentNode(const T& firstArg, const Types&... args) {
      _parentNodes.push_back(firstArg);
      pushParentNode(args...);
    }
    
    void pushChildNode() {};
    template<typename T, typename... Types> 
    void pushChildNode(const T& firstArg, const Types&... args) {
      _childNodes.push_back(firstArg);
      pushChildNode(args...);
    }
    
    void exlinkUpperNode() {};
    template<typename T, typename... Types> 
    void exlinkUpperNode(T& firstArg, Types&... args) {
      _parentNodes.push_back(firstArg);
      firstArg->pushChildNode(this);
      exlinkUpperNode(args...);
    }

    const std::vector<IRNode*>* getParentNodes() const {
      return &_parentNodes;
    }
    const std::vector<IRNode*>* getChildNode() const {
      return &_childNodes;
    }

    const IRNode* const getParentNode(int i) const{
      return _parentNodes[i];
    }
    const IRNode* const getChildNode(int i) const{
      return _childNodes[i];
    }

    const std::string name() const { return _name; };
    void setName(std::string name) { _name = name; };

    inline const int parentNum() const {
      return _parentNodes.size();
    }
    inline const int childNum() const {
      return _childNodes.size();
    }

	  // dot generation
	  std::string dotGen(std::string tensorInfo, std::string opInfo);
    std::string dotGen();

  private:
    std::vector<IRNode*> _parentNodes;
    std::vector<IRNode*> _childNodes;
    std::string _name;

    NodeType _nodeType;
};

} //namespace swc


#endif /* !IRNODE_H */
