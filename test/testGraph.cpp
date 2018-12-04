/*
 * testGraph.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-12-04
 */


#include "IRGraph.h"
#include "IRNode.h"
#include "common.h"

using namespace swc;
using namespace std;

int main() {

  printf ("start!\n");
  //Actually data hold node
  TensorNode<float> tNode1;
  TensorNode<float> tNode2;
  TensorNode<float> tNode3;
  TensorNode<float> tNode4;
  TensorNode<float> tNode5;
  OpNode<float> oNode1;
  OpNode<float> oNode2;
  printf("size of tnode: %d  onode: %d\n", sizeof(tNode1), sizeof(oNode1));
  
  vector<IRNode*> t1parent;
  vector<IRNode*> t2parent;
  vector<IRNode*> t3parent;
  vector<IRNode*> t4parent;
  vector<IRNode*> t5parent;
  vector<IRNode*> o1parent;
  vector<IRNode*> o2parent;
  vector<IRNode*> t1child;
  vector<IRNode*> t2child;
  vector<IRNode*> t3child;
  vector<IRNode*> t4child;
  vector<IRNode*> t5child;
  vector<IRNode*> o1child;
  vector<IRNode*> o2child;
  
  t1child.push_back(&oNode2);
  t3child.push_back(&oNode2);

  o2parent.push_back(&tNode1);
  o2parent.push_back(&tNode3);
  o2child.push_back(&tNode5);
  o2child.push_back(&tNode2);
  o2child.push_back(&tNode4);
  
  t5parent.push_back(&oNode2);
  t5child.push_back(&oNode1);
  t2parent.push_back(&oNode2);
  t2child.push_back(&oNode1);
  t4parent.push_back(&oNode2);

  o1parent.push_back(&tNode5);
  o1parent.push_back(&tNode2);

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
  tNode1.init(&t1parent, &t1child, string("Tensor1"));
  tNode2.init(&t2parent, &t2child, string("Tensor2"));
  tNode3.init(&t3parent, &t3child, string("Tensor3"));
  tNode4.init(&t4parent, &t4child, string("Tensor4"));
  tNode5.init(&t5parent, &t5child, string("Tensor5"));
  oNode1.init(&o1parent, &o1child, string("Op1"));
  oNode2.init(&o2parent, &o2child, string("Op2"));

  //TensorNode<float>* > tensorNodeVec;
  //OpNode<float>* > opNodeVec;
  //
  //tensorNodeVec.push_back(&tNode1);
  //tensorNodeVec.push_back(&tNode2);
  //tensorNodeVec.push_back(&tNode3);
  //tensorNodeVec.push_back(&tNode4);
  //tensorNodeVec.push_back(&tNode5);
  //opNodeVec.push_back(&oNode1);
  //opNodeVec.push_back(&oNode2);

  IRGraph<float> graph;

  printf("Before push, size of graph: %d\n", sizeof(graph));

  graph.pushTensorNode(&tNode1);
  graph.pushTensorNode(&tNode2);
  graph.pushTensorNode(&tNode3);
  graph.pushTensorNode(&tNode4);
  graph.pushTensorNode(&tNode5);
  graph.pushOpNode(&oNode1);
  graph.pushOpNode(&oNode2);

  printf("After push, size of graph: %d\n", sizeof(graph));
  
  printf ("Generate graph done!\n");
  
  for (int i = 0; i < graph.ternsorNodeNum(); i++) {
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
