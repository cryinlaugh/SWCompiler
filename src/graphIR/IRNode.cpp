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

IRNode::IRNode()
{
  _parentNodes = NULL;
  _childNodes = NULL;
}

IRNode::IRNode(std::vector<IRNode*>* parentNodes,
               std::vector<IRNode*>* childNodes,
               std::string name)
{
  _parentNodes = parentNodes;
  _childNodes = childNodes;
  _name = name;
}

IRNode::~IRNode() 
{
  printf("free:%s\n", _name.c_str());
}

void IRNode::init(std::vector<IRNode*>* parentNodes,
                  std::vector<IRNode*>* childNodes,
                  std::string name)
{
  _parentNodes = parentNodes;
  _childNodes = childNodes;
  _name = name;
}

} //namespace swc
