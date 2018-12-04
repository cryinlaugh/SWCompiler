/*
 * IRNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRNODE_H
#define IRNODE_H

#include "../common.h"

namespace swc {

class IRNode
{
  public:
    IRNode();
    ~IRNode();

    void setFatherNode(std::shared_ptr<std::vector<IRNode*> > fatherNode) {
      _fatherNode = fatherNode;
    }
    void setChildNode(std::shared_ptr<std::vector<IRNode*> > childNode) {
      _childNode = childNode;
    }
    IRNode* getFatherNode(int i) const{
      return (*_fatherNode)[i];
    }
    IRNode* getChildNode(int i) const{
      return (*_childNode)[i];
    }

    std::string name() {
      return _name;
    }

    void setName(std::string name) {
      _name = name;
    }

    inline int fatherNum() {
      return (*_fatherNode).size();
    }

    inline int childNum() {
      return (*_childNode).size();
    }

  private:
    std::shared_ptr<std::vector<IRNode*> > _fatherNode;
    std::shared_ptr<std::vector<IRNode*> > _childNode;
    std::string _name;
};

} //namespace swc


#endif /* !IRNODE_H */
