/*
 * testGraph.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-12-04
 */


#include "SWC.h"

using namespace swc;
using namespace std;

int main() {

  printf ("start!\n");
 
  TensorNode<float> tNode1("Tensor1");
  TensorNode<float> tNode3("Tensor3");
  printf("Init size of tnode: %ld\n", sizeof(tNode1));
  
  OpNode<float> oNode2("Op2");
  printf("Init size of onode: %ld\n", sizeof(oNode2));
  oNode2.pushParentNode(&tNode1, &tNode3);
  tNode1.pushChildNode(&oNode2);
  tNode3.pushChildNode(&oNode2);

  TensorNode<float> tNode2("Tensor2");
  tNode2.pushParentNode(&oNode2);
  oNode2.pushChildNode(&tNode2);
  
  TensorNode<float> tNode4("Tensor4");
  tNode4.pushParentNode(&oNode2);
  oNode2.pushChildNode(&tNode4);
  
  TensorNode<float> tNode5("Tensor5");
  tNode5.pushParentNode(&oNode2);
  oNode2.pushChildNode(&tNode5);
  
  OpNode<float> oNode1("Op1");
  oNode2.pushParentNode(&tNode5, &tNode2);
  tNode2.pushChildNode(&oNode1);
  tNode5.pushChildNode(&oNode1);

  printf("After push/size of tnode: %ld\n", sizeof(tNode1));
  printf("After push/size of onode: %ld\n", sizeof(oNode2));
  ///////////////////////////
  //    t1   t3
  //     \  /
  //      o2
  //     / / \
  //   /   \   \
  //  t5   t2  t4
  //   \   /
  //     o1
  //     

  IRGraph<float> graph;

  printf("Before push, size of graph: %ld\n", sizeof(graph));

  graph.pushTensorNode(&tNode1);
  graph.pushTensorNode(&tNode2);
  graph.pushTensorNode(&tNode3);
  graph.pushTensorNode(&tNode4);
  graph.pushTensorNode(&tNode5);
  graph.pushOpNode(&oNode1);
  graph.pushOpNode(&oNode2);

  dotGen(graph);

  printf("After push, size of graph: %ld\n", sizeof(graph));
  
  printf ("Generate graph done!\n");
  
  for (int i = 0; i < graph.tensorNodeNum(); i++) {
    printf("ID:%d, ", i);
    printf("Name:%s, ", graph.getTensorNode(i)->name().c_str());
    printf("in:%d, ", graph.getTensorNode(i)->parentNum());
    printf("out:%d\n", graph.getTensorNode(i)->childNum());
  }

  for (int i = 0; i < graph.opNodeNum(); i++) {
    printf("ID:%d, ", i);
    printf("Name:%s, ", graph.getOpNode(i)->name().c_str());
    printf("in:%d, ", graph.getOpNode(i)->parentNum());
    printf("out:%d\n", graph.getOpNode(i)->childNum());
  }

  return 0;
}
