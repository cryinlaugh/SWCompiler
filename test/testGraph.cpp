/*
 * testGraph.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-12-04
 */


#include "SWC.h"
#include "dotGen.h"

#define Dtype float

using namespace swc;
using namespace std;

int main() {

  printf ("start!\n");
 
  TensorNode<Dtype>* tNode1 = new TensorNode<Dtype>("Tensor1");
  TensorNode<Dtype>* tNode3 = new TensorNode<Dtype>("Tensor3");
  printf("Init size of tnode: %ld\n", sizeof(*tNode1));
  
  OpNode<Dtype>* oNode2 = new OpNode<Dtype>("Op2");
  printf("Init size of onode: %ld\n", sizeof(*oNode2));
  oNode2->pushParentNode(tNode1, tNode3);
  tNode1->pushChildNode(oNode2);
  tNode3->pushChildNode(oNode2);

  TensorNode<Dtype>* tNode2 = new TensorNode<Dtype>("Tensor2");
  tNode2->pushParentNode(oNode2);
  oNode2->pushChildNode(tNode2);
  
  TensorNode<Dtype>* tNode4 = new TensorNode<Dtype>("Tensor4");
  tNode4->pushParentNode(oNode2);
  oNode2->pushChildNode(tNode4);
  
  TensorNode<Dtype>* tNode5 = new TensorNode<Dtype>("Tensor5");
  tNode5->pushParentNode(oNode2);
  oNode2->pushChildNode(tNode5);
  
  OpNode<Dtype>* oNode1 = new OpNode<Dtype>("Op1");
  oNode1->pushParentNode(tNode5, tNode2);
  tNode2->pushChildNode(oNode1);
  tNode5->pushChildNode(oNode1);

  TensorNode<Dtype>* tNode6 = new TensorNode<Dtype>("Tensor6");
  tNode6->pushParentNode(oNode1);
  oNode1->pushChildNode(tNode6);
  printf("After push/size of tnode: %ld\n", sizeof(tNode1));
  printf("After push/size of onode: %ld\n", sizeof(oNode2));

  printf("//////////////////////\n");
  printf("//    t1   t3         \n");  
  printf("//     \\  /          \n");   
  printf("//      o2            \n");  
  printf("//     / / \\         \n");  
  printf("//   /   \\   \\      \n");  
  printf("//  t5   t2  t4       \n");    
  printf("//   \\   /           \n");  
  printf("//     o1             \n");   
  printf("//     /              \n");   
  printf("//    t6              \n");   
  printf("//////////////////////\n"); 

  IRGraph<Dtype> graph;

  printf("Before push, size of graph: %ld\n", sizeof(graph));
  graph.pushTensorNode(tNode1);
  graph.pushTensorNode(tNode2);
  graph.pushTensorNode(tNode3);
  graph.pushTensorNode(tNode4);
  graph.pushTensorNode(tNode5);
  graph.pushTensorNode(tNode6);
  graph.pushOpNode(oNode1);
  graph.pushOpNode(oNode2);
  printf("After push, size of graph: %ld\n", sizeof(graph));

  graph.findInOut();
  graph.updateTopology();
  graph.updateTopoNodeList();

  printf("\nGraph Struct:\nIn:");
  for (int i = 0; i < graph.inNodeNum(); i++)
    printf("%s  ", graph.getInNode(i)->name().c_str());
  printf("\nOut:");
  for (int i = 0; i < graph.outNodeNum(); i++)
    printf("%s  ", graph.getOutNode(i)->name().c_str());
  printf("\n\nTopology List:\n");
  for (int i = 0; i < graph.topologyNum(); i++) {
    printf("TopologyID: %d\t", i);
    for (int j = 0; j < graph.getNumInTopoLevel(i); j++)
      printf("%s  ", graph.getNodeInTopo(i, j)->name().c_str());
    printf("\n");
  }

  printf("\nNode Info:\n");
  for (int i = 0; i < graph.tensorNodeNum(); i++) {
    printf("ID:%d, ", i);
    printf("TopologyID:%d, ", graph.getTensorNode(i)->topologyId());
    printf("Name:%s, ", graph.getTensorNode(i)->name().c_str());
    printf("in:%d, ", graph.getTensorNode(i)->parentNum());
    printf("out:%d\n", graph.getTensorNode(i)->childNum());
  }

  for (int i = 0; i < graph.opNodeNum(); i++) {
    printf("ID:%d, ", i);
    printf("TopologyID:%d, ", graph.getOpNode(i)->topologyId());
    printf("Name:%s, ", graph.getOpNode(i)->name().c_str());
    printf("in:%d, ", graph.getOpNode(i)->parentNum());
    printf("out:%d\n", graph.getOpNode(i)->childNum());
  }

  //subgraph replacement
  //define subgraph
  OpNode<Dtype>* oNode1s = new OpNode<Dtype>("Op1s");
  OpNode<Dtype>* oNode2s = new OpNode<Dtype>("Op2s");

  TensorNode<Dtype>* tNode1s = new TensorNode<Dtype>("Tensor1s");
  TensorNode<Dtype>* tNode2s = new TensorNode<Dtype>("Tensor2s");
  TensorNode<Dtype>* tNode3s = new TensorNode<Dtype>("Tensor3s");
  TensorNode<Dtype>* tNode4s = new TensorNode<Dtype>("Tensor4s");
  tNode1s->exlinkUpperNode(oNode1s);
  tNode2s->exlinkUpperNode(oNode1s);
  tNode3s->exlinkUpperNode(oNode2s);
  tNode4s->exlinkUpperNode(oNode2s);

  OpNode<Dtype>* oNode3s = new OpNode<Dtype>("Op3s");
  OpNode<Dtype>* oNode4s = new OpNode<Dtype>("Op4s");
  oNode3s->exlinkUpperNode(tNode1s, tNode3s);
  oNode4s->exlinkUpperNode(tNode2s, tNode4s);

  TensorNode<Dtype>* tNode5s = new TensorNode<Dtype>("Tensor5s");
  TensorNode<Dtype>* tNode6s = new TensorNode<Dtype>("Tensor6s");
  tNode5s->exlinkUpperNode(oNode3s);
  tNode6s->exlinkUpperNode(oNode4s);
  
  OpNode<Dtype>* oNode5s = new OpNode<Dtype>("Op5s");
  oNode5s->exlinkUpperNode(tNode5s, tNode6s);

  //replaceMent P/C pointer 
  oNode2->destroyUpperNode(tNode1, tNode3);
  oNode1s->exlinkUpperNode(tNode1);
  oNode2s->exlinkUpperNode(tNode3);
  
  tNode2->destroyUpperNode(oNode2);
  tNode2->exlinkUpperNode(oNode5s);
  
  tNode4->destroyUpperNode(oNode2);
  tNode4->exlinkUpperNode(oNode5s);
  
  tNode5->destroyUpperNode(oNode2);
  tNode5->exlinkUpperNode(oNode5s);

  graph.pushOpNode(oNode1s, oNode2s, oNode3s, oNode4s, oNode5s);
  graph.pushTensorNode(tNode1s, tNode2s, tNode3s, tNode4s, tNode5s, tNode6s);
  
  //Now without free
  graph.delOpNode(oNode2);

  printf("///////////////////////////////////////////////////\n");
  printf("//    t1         t3        /                       \n");  
  printf("//     \\         /         /                      \n");   
  printf("//    o1s       o2s        /          o2           \n");  
  printf("//    / \\       /  \\       /          ||         \n");  
  printf("//   t1s  t2s  t3s  t4s    /          \\/          \n");  
  printf("//    \\     \\ /     /      /                     \n");  
  printf("//      \\    /\\    /       /     o1s       o2s   \n");  
  printf("//       \\  /   \\ /        /    / \\       /  \\ \n");  
  printf("//        o3s    o4s       /  t1s  t2s  t3s  t4s   \n");  
  printf("//         \\     /         /   \\     \\ /     /  \n");  
  printf("//         t5s  t6s        /     \\    /\\    /    \n");  
  printf("//           \\  /          /      \\  /   \\ /    \n");  
  printf("//            o5s          /      o3s    o4s       \n");  
  printf("//           / / \\         /       \\     /       \n");  
  printf("//         /   \\   \\       /       t5s  t6s      \n");  
  printf("//        t5   t2  t4      /         \\  /         \n");    
  printf("//         \\   /           /         o5s          \n");  
  printf("//           o1            /                       \n");   
  printf("//           /             /                       \n");   
  printf("//          t6             /                       \n");   
  printf("///////////////////////////////////////////////////\n"); 

  graph.findInOut();
  graph.updateTopology();
  graph.updateTopoNodeList();

  printf("\nGraph Struct:\nIn:");
  for (int i = 0; i < graph.inNodeNum(); i++)
    printf("%s  ", graph.getInNode(i)->name().c_str());
  printf("\nOut:");
  for (int i = 0; i < graph.outNodeNum(); i++)
    printf("%s  ", graph.getOutNode(i)->name().c_str());
  printf("\n\nTopology List:\n");
  for (int i = 0; i < graph.topologyNum(); i++) {
    printf("TopologyID: %d\t", i);
    for (int j = 0; j < graph.getNumInTopoLevel(i); j++)
      printf("%s  ", graph.getNodeInTopo(i, j)->name().c_str());
    printf("\n");
  }

  printf("\nNode Info:\n");
  for (int i = 0; i < graph.tensorNodeNum(); i++) {
    printf("ID:%d, ", i);
    printf("TopologyID:%d, ", graph.getTensorNode(i)->topologyId());
    printf("Name:%s, ", graph.getTensorNode(i)->name().c_str());
    printf("in:%d, ", graph.getTensorNode(i)->parentNum());
    printf("out:%d\n", graph.getTensorNode(i)->childNum());
  }

  for (int i = 0; i < graph.opNodeNum(); i++) {
    printf("ID:%d, ", i);
    printf("TopologyID:%d, ", graph.getOpNode(i)->topologyId());
    printf("Name:%s, ", graph.getOpNode(i)->name().c_str());
    printf("in:%d, ", graph.getOpNode(i)->parentNum());
    printf("out:%d\n", graph.getOpNode(i)->childNum());
  }

  return 0;
}
