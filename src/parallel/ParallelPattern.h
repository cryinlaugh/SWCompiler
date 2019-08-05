/***********************************************
#
#      Filename: src/parallel/ParallelPattern.cpp
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-07-05 11:04:16
# Last Modified: 2019-07-05 11:04:16
***********************************************/
#include "op/Op.h"
#include "op/dlOp/dlOp.h"
#include "graphIR/IRGraph.h"
#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"
#include "TilingLabel.h"
using namespace swc::op;
namespace swc {
class ForkPattern{
private:
    TensorNode* _tensornode;
    int _num;
public:
    ForkPattern(TensorNode * tensornode, int num) {

        _tensornode = tensornode;
        _num = num;
    };
    ~ForkPattern() {};

    void apply(int strategy, IRGraph * irgraph) {
        TilingLabel * tlabel = _tensornode->getTilingLabel();
        TensorShape * originshape = _tensornode->getTensor()->getTensorShape();
        TensorNode *tilenode;
        if(strategy >= 0) {
            TensorShape* tileTensorShape = originshape->getTiledShape(strategy, _num);
            tilenode = new TensorNode(_tensornode->name() + "_tile", new Tensor(tileTensorShape));
        } else if (strategy == -1) {
            tilenode = new TensorNode(_tensornode->name() + "_replicate", new Tensor(originshape));
        } else
            tilenode = new TensorNode(_tensornode->name() + "_unknown", new Tensor(originshape));
        
        
        OpNode *opnode = new OpNode(_tensornode->name() + "_fork");
        opnode->setOp(new ScatterOp());
        
        tilenode->exlinkUpperNode(opnode);
        opnode->exlinkUpperNode(_tensornode);



        
        irgraph->pushTensorNode(tilenode);
        irgraph->pushOpNode(opnode);
        irgraph->updateTopology();

        
        tlabel->setCurrentNode(tilenode);
        tlabel->setCurrentStrategy(strategy);
        tlabel->setApplied();

    }

};

class TransformPattern {
private:

    TensorNode * _tensornode;
    int _num;

public:
    TransformPattern(TensorNode * tensornode, int num) {
        _tensornode = tensornode;
        _num =num;
    };
    ~TransformPattern() {};
    void apply(int strategy, IRGraph * irgraph) {
        TilingLabel * tlabel = _tensornode->getTilingLabel();
        TensorShape * originshape = _tensornode->getTensor()->getTensorShape();
        TensorNode *tilenode;
        if(strategy >= 0) {
            TensorShape* tileTensorShape = originshape->getTiledShape(strategy, _num);
            tilenode = new TensorNode(_tensornode->name() + "_tile", new Tensor(tileTensorShape));
        } else if (strategy == -1) {
            tilenode = new TensorNode(_tensornode->name() + "_replicate", new Tensor(originshape));
        } else
            tilenode = new TensorNode(_tensornode->name() + "_unknown", new Tensor(originshape));
        
        OpNode *opnode = new OpNode(_tensornode->name() + "_transform");
        opnode->setOp(new ScatterOp());
        tilenode->exlinkUpperNode(opnode);
        opnode->exlinkUpperNode(tlabel->getCurrentNode());

        irgraph->pushTensorNode(tilenode);
        irgraph->pushOpNode(opnode);

        irgraph->updateTopology();
        tlabel->setCurrentNode(tilenode);
        tlabel->setCurrentStrategy(strategy);
        tlabel->setApplied();

    }

};

class JoinPattern{
private:
    TensorNode * _tensornode;
    int _num;
public:

    JoinPattern(TensorNode * tensornode, int num){
        _tensornode = tensornode;
        _num = num;
    };
    ~JoinPattern();

    void apply(int strategy, IRGraph * irgraph) {
        TilingLabel * tlabel = _tensornode->getTilingLabel();

        TensorShape * originshape = _tensornode->getTensor()->getTensorShape();
        TensorNode *tilenode;
        if(strategy >= 0) {
            
            TensorShape* tileTensorShape = originshape->getTiledShape(strategy, _num);
            tilenode = new TensorNode(_tensornode->name() + "_tile", new Tensor(tileTensorShape));
        } else if (strategy == -2) {
            tilenode = new TensorNode(_tensornode->name() + "_reduce", new Tensor(originshape));
        } else
            tilenode = new TensorNode(_tensornode->name() + "_unknown", new Tensor(originshape));

        OpNode *opnode = new OpNode(_tensornode->name() + "_join");
        opnode->setOp(new GatherOp());

        opnode->exlinkUpperNode(tilenode);
        _tensornode->exlinkUpperNode(opnode);

        irgraph->pushTensorNode(tilenode);
        irgraph->pushOpNode(opnode);
        irgraph->updateTopology();
        tlabel->setCurrentNode(tilenode);
        tlabel->setCurrentStrategy(strategy);
        tlabel->setApplied();

    }


};











}
