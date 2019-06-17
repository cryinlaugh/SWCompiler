/*
 * SWDSL.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SWDSL_H
#define SWDSL_H

// check TensorNode
#define CHECKT(tname)                                                          \
    SWLOG_INFO << "======================================================"     \
               << endl;                                                        \
    SWLOG_INFO << "Topology ID: " << tname->topologyId() << endl;              \
    SWLOG_INFO << "Name: " << tname->name().c_str() << endl;                   \
    SWLOG_INFO << "NDim: " << tname->getTensor()->getNDim() << endl;           \
    for (int i = 0; i < tname->getTensor()->getNDim(); i++) {                  \
        SWLOG_INFO << "Dim[" << i << "]: " << tname->getTensor()->getDim(i)    \
                   << endl;                                                    \
    }

// TensorNode
#define TENSOR(name, args...)                                                  \
    TensorNode *name = new TensorNode(#name);                                  \
    TensorShape *name##_TensorShape =                                          \
        new TensorShape(new vector<unsigned long>({args}));                    \
    Tensor *name##_Tensor = new Tensor(name##_TensorShape);                    \
    name->setTensor(name##_Tensor)



//set init for tensor
#define INIT(name, type, args)      \
    name##_Tensor->setTensorInit(type,args)
        

// check OpNode
#define CHECKO(oname)                                                          \
    SWLOG_INFO << "======================================================"     \
               << endl;                                                        \
    SWLOG_INFO << "Topology ID: " << oname->topologyId() << endl;              \
    SWLOG_INFO << "Name: " << oname->name().c_str() << endl

// OpNode
#define OP(name, method)                                                       \
    OpNode *name = new OpNode(#name);                                          \
    method *name##_Op = new method();                                          \
    name->setOp(name##_Op)

// link FATHER
#define LINKPARENT(self, parent...) self->pushParentNode(parent)

// link CHILD
#define LINKCHILD(self, child...) self->pushChildNode(child)

// exlink FATHER
#define LINKUPPER(self, upperNode...) self->exlinkUpperNode(upperNode)

// ex destroy FATHER
#define DESTROYUPPER(self, upperNode...) self->destroyUpperNode(upperNode)

#define G(name) IRGraph *name = new IRGraph()

#define GpT(name, tensorNodes...) name->pushTensorNode(tensorNodes)

#define GpO(name, OpNodes...) name->pushOpNode(OpNodes)


#define CHECKG(g)                                                              \
    printf("Generate MLP layer done!\n");                                      \
    for (int i = 0; i < g->tensorNodeNum(); i++) {                             \
        printf("ID:%d, ", i);                                                  \
        printf("TopologyID:%d, ", g->getTensorNode(i)->topologyId());          \
        printf("Name:%s, ", g->getTensorNode(i)->name().c_str());              \
        printf("in:%d, ", g->getTensorNode(i)->parentNum());                   \
        printf("out:%d\n", g->getTensorNode(i)->childNum());                   \
    }                                                                          \
                                                                               \
    for (int i = 0; i < g->opNodeNum(); i++) {                                 \
        printf("ID:%d, ", i);                                                  \
        printf("TopologyID:%d, ", g->getOpNode(i)->topologyId());              \
        printf("Name:%s, ", g->getOpNode(i)->name().c_str());                  \
        printf("in:%d, ", g->getOpNode(i)->parentNum());                       \
        printf("out:%d\n", g->getOpNode(i)->childNum());                       \
    }

// The DSL to generate a train network
// the main parameters include:
//  1. inference graph name
//  2. method
//  3. further method-paras
//  
//  now support SGD for paras: learning rate
#define TRAIN(graph, parameters...)                              \
    swc::pass::AutodiffPass auto_diff_path(graph);               \
    auto_diff_path.getMethods(parameters);                       \
    auto_diff_path.show();                                       \
    G(graph##_train);                                            \
    auto_diff_path.run(graph##_train);

#endif /* !SWDSL_H */
