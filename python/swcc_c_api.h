#include "SWC.h"


extern "C"
{
    swc::IRGraph *IRGraph();
    // swc::TensorShape *TensorShape();
    // swc::TensorNode *TensorNode(const char* name);
    swc::TensorNode* TensorNode(const char* name, int ndim, size_t *dims);
    // swc::OpNode *OpNode(const char* name);
    swc::OpNode *OpNode(const char* name, const char* op);

    void IRGraph_pushTensorNode(swc::IRGraph *graph, swc::TensorNode *tnode);
    void IRGraph_pushOpNode(swc::IRGraph *graph, swc::OpNode *onode);
    
    void OpNode_toString(swc::OpNode *op, char *str);
    void TensorNode_toString(swc::TensorNode *t, char *str);

    void OpNode_link(swc::OpNode* a, swc::TensorNode* b);
    void TensorNode_link(swc::TensorNode* a, swc::OpNode* b);

    void IRGraph_addOpNode(swc::IRGraph*, swc::OpNode *o);
    void IRGraph_addTensorNode(swc::IRGraph*, swc::TensorNode *t);
    const char *IRGraph_summary(swc::IRGraph*);
    
    void IRGraph_dotGen(swc::IRGraph* graph, const char* path); 
} // C  scope