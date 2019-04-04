/*
 * IRNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRNODE_H_
#define IRNODE_H_

#include <iostream>
#include <string>
#include "common.h"

#include "pass/Label.h"

namespace swc {

class IRNode {
 public:
  IRNode();
  IRNode(const NodeType nodeType, const char name[], IRNode* parent = nullptr)
    : _name(std::string(name)), 
      _nodeType(nodeType), 
      _label(new Label()) { 
        _topologyId = 0; 
        if(parent)
          exlinkUpperNode(parent);
  }
  ~IRNode() { printf("free:%s\n", _name.c_str()); }

  virtual void destroy(){};

  void pushParentNode() {};
  template<typename T, typename... Types> 
  void pushParentNode(const T& firstArg, const Types&... args) {
    _parentNodes.push_back(firstArg);
    pushParentNode(args...);
  }

  void delParentNode() {};
  template<typename T, typename... Types> 
  void delParentNode(const T& firstArg, const Types&... args) {
    if (!delVecMember(_parentNodes, firstArg)) {
      std::cout << "Del Parent Failed" << firstArg->name() << std::endl;
    }
    delParentNode(args...);
  }

  void pushChildNode() {};
  template<typename T, typename... Types> 
  void pushChildNode(const T& firstArg, const Types&... args) {
    _childNodes.push_back(firstArg);
    pushChildNode(args...);
  }

  void delChildNode() {};
  template<typename T, typename... Types> 
  void delChildNode(const T& firstArg, const Types&... args) {
    if (!delVecMember(_childNodes, firstArg)) {
      std::cout << "Del Parent Failed" << firstArg->name() << std::endl;
    }
    delChildNode(args...);
  }

  void exlinkUpperNode() {};
  template<typename T, typename... Types> 
  void exlinkUpperNode(const T& firstArg, const Types&... args) {
    pushParentNode(firstArg);
    firstArg->pushChildNode(this);
    exlinkUpperNode(args...);
  }

  void destroyUpperNode() {};
  template<typename T, typename... Types> 
  void destroyUpperNode(const T& firstArg, const Types&... args) {
    delParentNode(firstArg);
    firstArg->delChildNode(this);
    destroyUpperNode(args...);
  }

  const std::vector<IRNode*>* getParentNodes() const {
    return &_parentNodes;
  }
  const std::vector<IRNode*>* getChildNodes() const {
    return &_childNodes;
  }

  IRNode* getParentNode(int i) const{ return _parentNodes[i]; }
  IRNode* getChildNode(int i) const{ return _childNodes[i]; }

  inline const std::string name() const { return _name; };
  inline void setName(std::string name) { _name = name; };

  inline const int parentNum() const { return _parentNodes.size(); }
  inline const int childNum() const { return _childNodes.size(); }

  inline const int topologyId() const { return _topologyId; }
  inline void setTopologyId(int topologyId) { _topologyId = topologyId; }

  inline const NodeType nodeType() const { return _nodeType; }
  inline void setNodeType(NodeType nodeType) { _nodeType = nodeType; }
  
  Label* getLabel() const{
      return _label;
  }
  void setLabel(Label* label){
      _label = label;
  }

  virtual IRNode* clone() const = 0;

 private:
  std::vector<IRNode*> _parentNodes;
  std::vector<IRNode*> _childNodes;
  std::string _name;

  NodeType _nodeType;
  int _topologyId;
  Label* _label;
};

} //namespace swc


#endif /* !IRNODE_H_ */
