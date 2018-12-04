/*
 * IRGraph.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-12-04
 */


#include "IRGraph.h"

namespace swc {

template<typename Dtype>
IRGraph<Dtype>::IRGraph()
{
  _tensors = NULL;
  _ops = NULL;
}

template<typename Dtype>
IRGraph<Dtype>::~IRGraph() {}

template<typename Dtype>
void IRGraph<Dtype>::setTopology()
{
  return;
}

}
