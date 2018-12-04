/*
 * IRNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRNODE_H
#define IRNODE_H

#include <iostream>
namespace swc {

class IRNode 
{
  public:
    IRNode();
    ~IRNode();

    setFatherNode(std::vector<IRNode*> fatherNode) {
      _fatherNode = fatherNode;
    }
    setChildNode(std::vector<IRNode*> ChildNode) {
      _childNode = ChildNode;
    }
    IRNode* getFatherNode(int i) const{
      return _fatherNode[i];
    }
    IRNode* getChildNode(int i) const{
      return _childNode[i];
    }

  private:
    std::vector<IRNode*> _fatherNode;
    std::vector<IRNode*> _childNode;

}

} //namespace swc


#endif /* !IRNODE_H */
