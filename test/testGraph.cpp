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

int main() {

  printf ("start!\n");

  swc::TensorNode<float> tNode1;
  swc::TensorNode<float> tNode2;
  swc::TensorNode<float> tNode3;
  swc::TensorNode<float> tNode4;
  swc::TensorNode<float> tNode5;
  swc::OpNode<float> oNode1;
  swc::OpNode<float> oNode2;

  tNode1.setName(std::string("Tensor1"));
  tNode2.setName(std::string("Tensor2"));
  tNode3.setName(std::string("Tensor3"));
  tNode4.setName(std::string("Tensor4"));
  tNode5.setName(std::string("Tensor5"));
  oNode1.setName(std::string("Op1"));
  oNode2.setName(std::string("Op2"));

  std::vector<swc::IRNode*> t1father;
  std::vector<swc::IRNode*> t2father;
  std::vector<swc::IRNode*> t3father;
  std::vector<swc::IRNode*> t4father;
  std::vector<swc::IRNode*> t5father;
  std::vector<swc::IRNode*> o1father;
  std::vector<swc::IRNode*> o2father;
  std::vector<swc::IRNode*> t1child;
  std::vector<swc::IRNode*> t2child;
  std::vector<swc::IRNode*> t3child;
  std::vector<swc::IRNode*> t4child;
  std::vector<swc::IRNode*> t5child;
  std::vector<swc::IRNode*> o1child;
  std::vector<swc::IRNode*> o2child;
  
  
  t1child.push_back(&oNode2);
  t3child.push_back(&oNode2);

  o2father.push_back(&tNode1);
  o2father.push_back(&tNode3);
  o2child.push_back(&tNode5);
  o2child.push_back(&tNode2);
  o2child.push_back(&tNode4);
  
  t5father.push_back(&oNode2);
  t5child.push_back(&oNode1);
  t2father.push_back(&oNode2);
  t2child.push_back(&oNode1);
  t4father.push_back(&oNode2);

  o1father.push_back(&tNode5);
  o1father.push_back(&tNode2);

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
  tNode1.setChildNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t1child));
  tNode2.setChildNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t2child));
  tNode3.setChildNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t3child));
  tNode4.setChildNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t4child));
  tNode5.setChildNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t5child));
  oNode1.setChildNode(std::shared_ptr<std::vector<swc::IRNode*> >(&o1child));
  oNode2.setChildNode(std::shared_ptr<std::vector<swc::IRNode*> >(&o2child));
  
  tNode1.setFatherNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t1father));
  tNode2.setFatherNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t2father));
  tNode3.setFatherNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t3father));
  tNode4.setFatherNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t4father));
  tNode5.setFatherNode(std::shared_ptr<std::vector<swc::IRNode*> >(&t5father));
  oNode1.setFatherNode(std::shared_ptr<std::vector<swc::IRNode*> >(&o1father));
  oNode2.setFatherNode(std::shared_ptr<std::vector<swc::IRNode*> >(&o2father));

  return 0;
  
  std::vector<TensorNode<float>* > tensorNodeVec;
  std::vector<OpNode<float>* > opNodeVec;

  tensorNodeVec.push_back(&tNode1);
  tensorNodeVec.push_back(&tNode2);
  tensorNodeVec.push_back(&tNode3);
  tensorNodeVec.push_back(&tNode4);
  tensorNodeVec.push_back(&tNode5);
  opNodeVec.push_back(&oNode1);
  opNodeVec.push_back(&oNode2);
  
  IRGraph<float> graph;

  graph.setTensorNodes(std::shared_ptr<std::vector<TensorNode<float>* > >(&tensorNodeVec));
  graph.setOpNodes(std::shared_ptr<std::vector<OpNode<float>* > > (&opNodeVec));

  printf ("Generate graph done!\n");
  
  for (int i = 0; i < graph.ternsorNodeNum(); i++) {
    printf("ID:%d, ", i);
    printf("Name:%s, ", graph.getTensorNode(i)->name().c_str());
    printf("in:%d, ", graph.getTensorNode(i)->fatherNum());
    printf("out:%d\n", graph.getTensorNode(i)->childNum());
  }

  for (int i = 0; i < graph.opNodeNum(); i++) {
    printf("ID:%d, ", i);
    printf("Name:%s, ", graph.getOpNode(i)->name().c_str());
    printf("in:%d, ", graph.getOpNode(i)->fatherNum());
    printf("out:%d\n", graph.getOpNode(i)->childNum());
  }

  return 0;
}
