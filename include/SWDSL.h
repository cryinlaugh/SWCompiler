/*
 * SWDSL.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SWDSL_H
#define SWDSL_H

#include "SWC.h"

using namespace swc;
using namespace std;

//check TensorNode
#define CHECKT(prefix, id) \
  cout << "======================================================" << endl;\
  cout << "Name: " << prefix##TensorNode_##id->name().c_str() << endl; \
  cout << "NDim: " << prefix##TensorNode_##id->getTensor()->getNDim() << endl; \
  for (int i = 0; i < prefix##TensorNode_##id->getTensor()->getNDim(); i++) \
    cout << "Dim[" << i << "]: " << prefix##TensorNode_##id->getTensor()->getDim(i) << endl;


//TensorNode
#define TENSOR(prefix, id, name, dimX, dimY) \
  TensorNode<Dtype>* prefix##TensorNode_##id = new TensorNode<Dtype>(#name);\
  TensorShape* prefix##TensorShape_##id = new TensorShape(\
      new vector<unsigned long>({ dimX, dimY })); \
  Tensor<Dtype>* prefix##Tensor_##id = new Tensor<Dtype>(prefix##TensorShape_##id);\
  prefix##TensorNode_##id->setTensor(prefix##Tensor_##id);

//check OpNode
#define CHECKO(prefix, id) \
  cout << "======================================================" << endl;\
  cout << "Name: " << prefix##OpNode_##id->name().c_str() << endl; \

//OpNode
#define OP(prefix, id, name, method) \
  OpNode<Dtype>* prefix##OpNode_##id = new OpNode<Dtype>(#name);\
  method<Dtype>* prefix##Op_##id = new method<Dtype>();\
  prefix##OpNode_##id->setOp(prefix##Op_##id);


//link FATHER
#define LINKPARENT(self, parent...) \
  self->pushParentNode(parent);
  
//link CHILD
#define LINKCHILD(self, child...) \
  self->pushChildNode(child);

//link CHILD
#define LINKUPPER(self, upperNode...) \
  self->exlinkUpperNode(upperNode);

#define T(prefix, id) prefix##TensorNode_##id
#define O(prefix, id) prefix##OpNode_##id

#define G(name) IRGraph<Dtype>* name = new IRGraph<Dtype>();

#define GpT(name, tensorNodes...) \
  name->pushTensorNode(tensorNodes);

#define GpO(name, OpNodes...) \
  name->pushOpNode(OpNodes);



#define checkG(g) \
    printf ( "Generate MLP layer done!\n");\
    for (int i = 0; i < g->ternsorNodeNum(); i++) {\
        printf( "ID:%d, ", i);\
        printf( "Name:%s, ", g->getTensorNode(i)->name().c_str());\
        printf( "in:%d, ", g->getTensorNode(i)->parentNum());\
        printf( "out:%d\n", g->getTensorNode(i)->childNum());\
    }\
\
    for (int i = 0; i < g->opNodeNum(); i++) {\
        printf( "ID:%d, ", i);\
        printf( "Name:%s, ", g->getOpNode(i)->name().c_str());\
        printf( "in:%d, ", g->getOpNode(i)->parentNum());\
        printf( "out:%d\n", g->getOpNode(i)->childNum());\
    }


#endif /* !SWDSL_H */
