/*
 * SWDSL.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SWDSL_H
#define SWDSL_H

#include "SWC.h"
#include "SWLOG.h"

using namespace swc;
using namespace std;

//check TensorNode
#define CHECKT(tname) \
  SWLOG_INFO << "======================================================" << endl;\
  SWLOG_INFO << "Topology ID: " << tname->topologyId() << endl; \
  SWLOG_INFO << "Name: " << tname->name().c_str() << endl; \
  SWLOG_INFO << "NDim: " << tname->getTensor()->getNDim() << endl; \
  for (int i = 0; i < tname->getTensor()->getNDim(); i++) {\
    SWLOG_INFO << "Dim[" << i << "]: " << tname->getTensor()->getDim(i) << endl;}


//TensorNode
#define TENSOR(name, args...) \
  TensorNode<Dtype>* name = new TensorNode<Dtype>(#name);\
  TensorShape* name##_TensorShape = new TensorShape(\
      new vector<unsigned long>({ args })); \
  Tensor<Dtype>* name##_Tensor = new Tensor<Dtype>(name##_TensorShape);\
  name->setTensor(name##_Tensor);

//check OpNode
#define CHECKO(oname) \
  SWLOG_INFO << "======================================================" << endl;\
  SWLOG_INFO << "Topology ID: " << oname->topologyId() << endl; \
  SWLOG_INFO << "Name: " << oname->name().c_str() << endl; \

//OpNode
#define OP(name, method) \
  OpNode<Dtype>* name = new OpNode<Dtype>(#name);\
  method<Dtype>* name##_Op = new method<Dtype>();\
  name->setOp(name##_Op);


//link FATHER
#define LINKPARENT(self, parent...) \
  self->pushParentNode(parent);
  
//link CHILD
#define LINKCHILD(self, child...) \
  self->pushChildNode(child);

//link CHILD
#define LINKUPPER(self, upperNode...) \
  self->exlinkUpperNode(upperNode);

#define G(name) IRGraph<Dtype>* name = new IRGraph<Dtype>();

#define GpT(name, tensorNodes...) \
  name->pushTensorNode(tensorNodes);

#define GpO(name, OpNodes...) \
  name->pushOpNode(OpNodes);


#define CHECKG(g) \
    printf ( "Generate MLP layer done!\n");\
    for (int i = 0; i < g->tensorNodeNum(); i++) {\
        printf( "ID:%d, ", i);\
        printf( "TopologyID:%d, ", g->getTensorNode(i)->topologyId());\
        printf( "Name:%s, ", g->getTensorNode(i)->name().c_str());\
        printf( "in:%d, ", g->getTensorNode(i)->parentNum());\
        printf( "out:%d\n", g->getTensorNode(i)->childNum());\
    }\
\
    for (int i = 0; i < g->opNodeNum(); i++) {\
        printf( "ID:%d, ", i);\
        printf( "TopologyID:%d, ", g->getOpNode(i)->topologyId());\
        printf( "Name:%s, ", g->getOpNode(i)->name().c_str());\
        printf( "in:%d, ", g->getOpNode(i)->parentNum());\
        printf( "out:%d\n", g->getOpNode(i)->childNum());\
    }


#endif /* !SWDSL_H */
