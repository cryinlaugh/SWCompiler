/*************************************************************************
	> File Name: src/pass/ElimTransposePass.h
	> Author: wayne
	> Mail:  
	> Created Time: Sat 14 Sep 2019 04:16:28 PM UTC
 ************************************************************************/
#ifndef _ELIMTRANSPOSEPASS_H_
#define _ELIMTRANSPOSEPASS_H_

#include "OptimizePass.h"
#include "SWLOG.h"
#include "op/Op.h"
#include "op/dlOp/dlOp.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "SWDSL.h"

namespace swc {
namespace pass {

class ElimTransposePass : public OptimizePass {
    using OptimizePass::_graph;
    bool isReverse(std::vector<size_t> &a, std::vector<size_t> &b) {
        if(a.size() != b.size())
            return false;

        // nchw2nhwc: 0231
        // nhwc2nchw: 0312
        // std::vector<size_t> composed_shuffle(a.size());
        size_t idx = 0;
        for(auto s : b) {
            assert(s<a.size() && "illegal shuffle\n");
            if(a.at(s) != idx)
                return false;
            idx++;
        }

        return true;
    }
public:
    ElimTransposePass(IRGraph *g) : OptimizePass(g) {}
    ~ElimTransposePass() {}

    void run() {
        int nOpNodes = _graph->opNodeNum();
        for (int i = 0; i < nOpNodes; i++) {
            auto *opnode = _graph->getOpNode(i);  
            auto *op = dynamic_cast<TransposeOp*>(opnode->getOp());
            if(op == nullptr) 
                continue;

            auto *trans_in = (TensorNode*)opnode->getParentNode(0); 
            auto *trans_out = (TensorNode*)opnode->getChildNode(0); 
        
            // maybe many childnodes
            for(int op_ci=0; op_ci<trans_out->childNum(); op_ci++) {
                auto next_opnode = (OpNode*)trans_out->getChildNode(op_ci);
                /*
                auto next_opnode = trans_out->childNum() ? (OpNode*)trans_out->getChildNode(0) : nullptr; 
                if(next_opnode == nullptr)
                    continue; 
                */

                auto *next_op = dynamic_cast<TransposeOp*>(next_opnode->getOp());
                if(next_op == nullptr)
                    continue; 

                std::vector<size_t> shuffle = op->getShuffle();
                std::vector<size_t> next_shuffle = next_op->getShuffle();
                
                if(isReverse(shuffle, next_shuffle)) {
                    SWLOG_DEBUG(10) << opnode->name() << " and " << next_opnode->name() << " has reversed shuffle\n";
                
                    // we will not delete double transpose, this job should be done in DCE
                    
                    auto *next_transop_out = (TensorNode*)next_opnode->getChildNode(0);
                    next_transop_out->replaceUseKeepOrder(trans_in);
                    /*
                    for(auto child : next_transop_out->getChildNodes()) {
                        DESTROYUPPER(child, next_transop_out);
                        LINKUPPER(child, trans_in);   
                    }
                    */
                }
            } // for i
        } // for opNodes
    }

}; 
}
}

#endif
