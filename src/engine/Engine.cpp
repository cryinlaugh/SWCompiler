/*************************************************************************
	> File Name: Engine.cpp
	> Author: wayne
	> Mail:  
	> Created Time: Sat 14 Sep 2019 10:40:39 AM UTC
 ************************************************************************/
#include "Engine.h"
#include "graphIR/IRGraph.h"
#include "graphIR/IRNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "op/Op.h"
#include "op/dlOp/dlOp.h"
#include "pass/Optimizer.h"
#include "pass/EliminationPass.h"
#include "pass/ElimTransposePass.h"
#include "pass/LabelingPass.h"
#include "pass/LoweringPass.h"
#include "pass/RenamingNodePass.h"
#include "pass/ParallelLabelingPass.h"
#include "pass/ParallelLoweringPass.h"
#include "pass/AutodiffPass.h"
#include "SWLOG.h"
#include "SWDSL.h"
#include "tool/dotGen.h"
#include "codegen/Codegen.h"
#include "common.h"
#include <map>

namespace swc {
using namespace op;
using namespace codegen;
using namespace pass;

void Engine::compile() {
    auto config = graph_->getConfig();

    // labeling passes will set lowermark
    if(config.train_mode)
        runTrainPasses();
    else
        runInferPasses();

    // parallelization should be done 
    // before graph layout transformation
    // since parallel strategies are generated
    // according to einSum representation like "nchw"
    if(config.mpi) {
        runParallelPasses(); 
    }

    // transform mkl-dnn supported operators to
    // nchw layout format
    if(config.mkldnn)
        transformForMKLDNN();

    optimize();

}

void Engine::runInferPasses() {
    PassManager passManager;
    auto renamingpass = new RenamingNodePass(graph_); 
    auto labelingpass = new LabelingPass(graph_); 
    auto loweringpass = new LoweringPass(graph_);

    passManager.add(labelingpass);
    passManager.add(loweringpass);
    // run labeling again for new nodes from lowering
    passManager.add(labelingpass);
    // renaming currently cannot be removed because
    // Codegen use node name, but duplicated name
    // may be introduced in lowering pass
    passManager.add(renamingpass); 

    passManager.run();

    // update nodesByTopology 
    // because new nodes added
    graph_->findInOut();
    graph_->updateTopology();

    // if mode is inference, we need to set topoOutnodes's isOut mark
    // or all nodes will be elimianted
    graph_->setOutMark();
}

void Engine::runTrainPasses() {

    auto config = graph_->getConfig();
    std::string method = config.train_config.optimizer;
    float lr =  config.train_config.lr;
    float decay =  config.train_config.decay;
    float momentum =  config.train_config.momentum;
    float batch =  config.train_config.batch;
    
    auto diffpass = new AutodiffPass(graph_);
    diffpass->getMethods(method, lr, decay, momentum, batch);
    // diffpass->show();
    // autodiff on graph_ directly,
    // after this pass, graph_ is a train graph
    diffpass->run();

    // update nodesByTopology 
    graph_->findInOut();
    graph_->updateTopology();

    PassManager passManager;
    auto labelingpass = new LabelingPass(graph_); 
    auto loweringpass = new LoweringPass(graph_);
    auto renamingpass = new RenamingNodePass(graph_); 

    passManager.add(labelingpass);
    passManager.add(loweringpass);
    // run labeling again for new nodes from lowering
    passManager.add(labelingpass);
    passManager.add(renamingpass); 

    passManager.run();

    // update nodesByTopology 
    // because new nodes added
    graph_->findInOut();
    graph_->updateTopology();
}

void Engine::runParallelPasses() {

    PassManager passManager;

    auto para_labeling = new ParallelLabelingPass(graph_); 
    auto para_lowering = new ParallelLoweringPass(graph_); 
    auto renaming  = new RenamingNodePass(graph_);
    auto eliming = new EliminationPass(graph_); 

    passManager.add(para_labeling);
    passManager.add(para_lowering);
    passManager.add(renaming);
    // passManager.add(eliming);

    passManager.run();

    dotGen(graph_, "mlp_para_before_elim.dot");

    eliming->run();

    // update nodesByTopology 
    // because new nodes added
    graph_->findInOut();
    graph_->updateTopology();
}

void Engine::transformForMKLDNN() {
    SWLOG_DEBUG(10) << "engine specific transform before codegen begin \n";
    auto config = graph_->getConfig();

    // currently only do tranform for MKLDNN
    if(!config.mkldnn)
        return;

    for(int i=0; i<graph_->opNodeNum(); i++) {
        OpNode *node = graph_->getOpNode(i);
        if(dynamic_cast<Conv2dOp*>(node->getOp())) {
            // for input and weight
            for(int i=0; i<2; i++) {
                auto src = (TensorNode*)node->getParentNode(i); 

                SWLOG_DEBUG(10) << src->name() << " mem layout: " << MEM_LAYOUT.at(src->getTensor()->getMemLayout()) << "\n";
                if(src->getTensor()->getNDim() != 4)
                    continue;
                if(src->getMemLayout() == layout_nchw)
                    continue; 

                SWLOG_DEBUG(10) << "Conv_src" << i << src->name() << " transpose to NCHW\n";

                std::string trans_op_name = "trans_" + src->name();
                auto trans_op = new OpNode(trans_op_name, new TransposeOp(NHWC2NCHW));
                //DESTROYUPPER(node, src);
                LINKUPPER(trans_op, src);
                
                Tensor *trans_out_t = new Tensor(src->getTensor()->getShuffledTensorShape(NHWC2NCHW), src->getDataType(), layout_nchw);
                std::string trans_out_name = src->name() + "_t";
                auto trans_out = new TensorNode(trans_out_name, trans_out_t, trans_op);
                src->replaceUseKeepOrder(node, trans_out);
            
                graph_->pushOpNode(trans_op);
                graph_->pushTensorNode(trans_out);
            }
             
            auto dst = (TensorNode*)node->getChildNode(0); 
            if(dst->getMemLayout() == layout_nchw)
                continue; 
            SWLOG_DEBUG(10) << "Conv_dst" << dst->name() << " transpose from NCHW to NHWC\n";
            
            // break original out and conv
            DESTROYUPPER(dst, node);
            // create new conv out
            Tensor *conv_out_t = new Tensor(dst->getTensor()->getShuffledTensorShape(NHWC2NCHW), dst->getDataType(), layout_nchw);
            std::string conv_out_name = dst->name() + "_nchw"; 
            // add conv_out as node's child
            auto conv_out = new TensorNode(conv_out_name, conv_out_t, node);
            
            std::string trans_out_op_name = "trans_to_" + dst->name();
            auto trans_out_op = new OpNode(trans_out_op_name, new TransposeOp(NCHW2NHWC));
            LINKUPPER(trans_out_op, conv_out);
            LINKUPPER(dst, trans_out_op);
            
            graph_->pushTensorNode(conv_out);
            graph_->pushOpNode(trans_out_op);
            
            /*
            auto src = (TensorNode*)node->getParentNode(0); 
            auto weight = (TensorNode*)node->getParentNode(1); 
            auto dst = (TensorNode*)node->getChildNode(0); 
            SWLOG_DEBUG(10) << src->name() << " mem layout: " << MEM_LAYOUT.at(src->getTensor()->getMemLayout()) << "\n";
            SWLOG_DEBUG(10) << weight->name() << " mem layout: " << MEM_LAYOUT.at(weight->getTensor()->getMemLayout()) << "\n";
            if(weight->getMemLayout() == layout_nchw)
                continue; 

            SWLOG_DEBUG(10) << "Conv_w " << weight->name() << " transpose to NCHW\n";

            std::string trans_op_name = "trans_" + weight->name();
            auto trans_op = new OpNode(trans_op_name, new TransposeOp(NHWC2NCHW));
            //DESTROYUPPER(node, weight);
            LINKUPPER(trans_op, weight);
            
            Tensor *trans_out_t = new Tensor(weight->getTensor()->getShuffledTensorShape(NHWC2NCHW), weight->getDataType(), layout_nchw);
            std::string trans_out_name = weight->name() + "_t";
            auto trans_out = new TensorNode(trans_out_name, trans_out_t, trans_op);
            // LINKUPPER(node, trans_out);
            // !!!!
            // DESTROYUPPER(node, weight) and LINKUPPER(node,trans_out)
            // will cause conv parents: 0-data 1-bias 2-weight
            weight->replaceUseKeepOrder(node, trans_out);
        
            graph_->pushOpNode(trans_op);
            graph_->pushTensorNode(trans_out);
            SWLOG_DEBUG(10) << "Conv_out " << dst->name() << " transpose from NCHW\n";
            
            // break original out and conv
            DESTROYUPPER(dst, node);
            // create new conv out
            Tensor *conv_out_t = new Tensor(dst->getTensor()->getShuffledTensorShape(NHWC2NCHW), dst->getDataType(), layout_nchw);
            std::string conv_out_name = dst->name() + "_nchw"; 
            // add conv_out as node's child
            auto conv_out = new TensorNode(conv_out_name, conv_out_t, node);
            
            std::string trans_out_op_name = "trans_to_" + dst->name();
            auto trans_out_op = new OpNode(trans_out_op_name, new TransposeOp(NCHW2NHWC));
            LINKUPPER(trans_out_op, conv_out);
            LINKUPPER(dst, trans_out_op);
            
            graph_->pushTensorNode(conv_out);
            graph_->pushOpNode(trans_out_op);
            */ 
        }

        if(dynamic_cast<MatrixMatrixFCBiasOp*>(node->getOp())
            || dynamic_cast<MatrixMatrixMulOp*>(node->getOp())
        ) {
            /*
             * our framework: when import caffe2, trans w from oCiC to iCoC(chw, OC), trans [in] from our nhwc to nchw
             * consequently, for mkldnn
             * remove trans for w, if exist (right for importer w-t-wT-fc, wrong for user defined w-FC)
             * 1. trans w from iCoC to oCiC
             * 2. when codegen, if in 4D, view w as 4D-nchw/oihw
            */
            /*
             * if Caffe2 imported, in already NCHW
             * if user defined, its possible in is NHWC   
             */
            // auto in = (TensorNode*)node->getParentNode(0); 

            auto weight = (TensorNode*)node->getParentNode(1); 
            SWLOG_DEBUG(10) << weight->name() << " mem layout: " << MEM_LAYOUT.at(weight->getTensor()->getMemLayout()) << "\n";
            weight->setMemLayout(layout_cn); // in SWC MM, we call iCoC as cn
            
            std::string trans_op_name = "trans_" + weight->name();
            auto trans_op = new OpNode(trans_op_name, new TransposeOp({1, 0}));
            //DESTROYUPPER(node, weight);
            LINKUPPER(trans_op, weight);
            
            Tensor *trans_out_t = new Tensor(weight->getTensor()->getShuffledTensorShape({1,0}), weight->getDataType(), layout_nc);
            std::string trans_out_name = weight->name() + "_t";
            auto trans_out = new TensorNode(trans_out_name, trans_out_t, trans_op);
            weight->replaceUseKeepOrder(node, trans_out);
        
            graph_->pushOpNode(trans_op);
            graph_->pushTensorNode(trans_out);
             
            SWLOG_DEBUG(10) << "FC_w " << weight->name() << " transpose to oCiC\n";          
        }

        if(dynamic_cast<BatchNormalizationOp*>(node->getOp())
            || dynamic_cast<MaxPoolOp*>(node->getOp())
            || dynamic_cast<AvgPoolOp*>(node->getOp())
            || dynamic_cast<ReluOp*>(node->getOp()) ) {
            auto src = (TensorNode*)node->getParentNode(0); 
            auto dst = (TensorNode*)node->getChildNode(0); 
            SWLOG_DEBUG(10) << src->name() << " mem layout: " << MEM_LAYOUT.at(src->getTensor()->getMemLayout()) << "\n";
            if(src->getTensor()->getNDim() != 4)
                continue;
            if(src->getMemLayout() == layout_nchw)
                continue; 

            SWLOG_DEBUG(10) << "BN_in" << src->name() << " transpose to NCHW\n";

            std::string trans_op_name = "trans_" + src->name();
            auto trans_op = new OpNode(trans_op_name, new TransposeOp(NHWC2NCHW));
            //DESTROYUPPER(node, src);
            LINKUPPER(trans_op, src);
            
            Tensor *trans_out_t = new Tensor(src->getTensor()->getShuffledTensorShape(NHWC2NCHW), src->getDataType(), layout_nchw);
            std::string trans_out_name = src->name() + "_t";
            auto trans_out = new TensorNode(trans_out_name, trans_out_t, trans_op);
            src->replaceUseKeepOrder(node, trans_out);
        
            graph_->pushOpNode(trans_op);
            graph_->pushTensorNode(trans_out);
             
            if(dst->getMemLayout() == layout_nchw)
                continue; 
            SWLOG_DEBUG(10) << "BN_dst" << dst->name() << " transpose from NCHW to NHWC\n";
            
            // break original out and conv
            DESTROYUPPER(dst, node);
            // create new conv out
            Tensor *conv_out_t = new Tensor(dst->getTensor()->getShuffledTensorShape(NHWC2NCHW), dst->getDataType(), layout_nchw);
            std::string conv_out_name = dst->name() + "_nchw"; 
            // add conv_out as node's child
            auto conv_out = new TensorNode(conv_out_name, conv_out_t, node);
            
            std::string trans_out_op_name = "trans_to_" + dst->name();
            auto trans_out_op = new OpNode(trans_out_op_name, new TransposeOp(NCHW2NHWC));
            LINKUPPER(trans_out_op, conv_out);
            LINKUPPER(dst, trans_out_op);
            
            graph_->pushTensorNode(conv_out);
            graph_->pushOpNode(trans_out_op);
        }
        if(dynamic_cast<ElementAddOp*>(node->getOp()) ) {
            int num_srcs = node->parentNum();
            for(int i=0; i<num_srcs; i++) {
                auto src = (TensorNode*)node->getParentNode(i); 

                SWLOG_DEBUG(10) << src->name() << " mem layout: " << MEM_LAYOUT.at(src->getTensor()->getMemLayout()) << "\n";
                if(src->getTensor()->getNDim() != 4)
                    continue;
                if(src->getMemLayout() == layout_nchw)
                    continue; 

                SWLOG_DEBUG(10) << "Sum_src" << i << src->name() << " transpose to NCHW\n";

                std::string trans_op_name = "trans_" + src->name();
                auto trans_op = new OpNode(trans_op_name, new TransposeOp(NHWC2NCHW));
                //DESTROYUPPER(node, src);
                LINKUPPER(trans_op, src);
                
                Tensor *trans_out_t = new Tensor(src->getTensor()->getShuffledTensorShape(NHWC2NCHW), src->getDataType(), layout_nchw);
                std::string trans_out_name = src->name() + "_t";
                auto trans_out = new TensorNode(trans_out_name, trans_out_t, trans_op);
                src->replaceUseKeepOrder(node, trans_out);
            
                graph_->pushOpNode(trans_op);
                graph_->pushTensorNode(trans_out);
            }
             
            auto dst = (TensorNode*)node->getChildNode(0); 
            if(dst->getMemLayout() == layout_nchw)
                continue; 
            SWLOG_DEBUG(10) << "BN_dst" << dst->name() << " transpose from NCHW to NHWC\n";
            
            // break original out and conv
            DESTROYUPPER(dst, node);
            // create new conv out
            Tensor *conv_out_t = new Tensor(dst->getTensor()->getShuffledTensorShape(NHWC2NCHW), dst->getDataType(), layout_nchw);
            std::string conv_out_name = dst->name() + "_nchw"; 
            // add conv_out as node's child
            auto conv_out = new TensorNode(conv_out_name, conv_out_t, node);
            
            std::string trans_out_op_name = "trans_to_" + dst->name();
            auto trans_out_op = new OpNode(trans_out_op_name, new TransposeOp(NCHW2NHWC));
            LINKUPPER(trans_out_op, conv_out);
            LINKUPPER(dst, trans_out_op);
            
            graph_->pushTensorNode(conv_out);
            graph_->pushOpNode(trans_out_op);
        }
    }

    graph_->updateTopology();

    // new transpose node need labeling  
    pass::LabelingPass labelingpass(graph_);
    labelingpass.run();

    SWLOG_DEBUG(10) << "engine specific transform before codegen end\n";
}

void Engine::optimize() {

    pass::ElimTransposePass elimpass(graph_); 
    elimpass.run();

    pass::EliminationPass elim(graph_);
    elim.run();
}

std::string Engine::genCode() {

    if(generator_ == nullptr) {
        Config config = graph_->getConfig();
        if(config.mpi)
            generator_ = new ParallelCodegen(graph_, config);
        else
            generator_ = new Codegen(graph_, config);
    }

    std::string code = generator_->generate();
    return code;
}

} // namespace swc
