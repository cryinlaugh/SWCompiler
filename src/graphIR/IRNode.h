/*
 * IRNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef IRNODE_H
#define IRNODE_H

namespace swc {

template <typename Dtype>
class IRNode {
  
  public:
    IRNode();
    ~IRNode();

    setFatherNode(vector<IRNode*> fatherNode) {
      _fatherNode = fatherNode;
    }
    setChildNode(vector<IRNode*> ChildNode) {
      _childNode = ChildNode;
    }
    IRNode* getFatherNode(int i) const{
      return _fatherNode[i];
    }
    IRNode* getChildNode(int i) const{
      return _childNode[i];
    }

  private:
    vector<IRNode<Dtype> * > _fatherNode;
    vector<IRNode<Dtype> * > _childNode;

}

} //namespace swc


#endif /* !IRNODE_H */
