/*
 * IRGraph.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-12-04
 */


#include "IRGraph.h"

IRGraph::IRGraph()
{
  _tensors = NULL;
  _operations = NULL;
}

IRGraph::~IRGraph() {}

void IRGraph::setTopology()
{
  return;
}
