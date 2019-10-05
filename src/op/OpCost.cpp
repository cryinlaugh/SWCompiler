/*************************************************************************
	> File Name: src/op/dlOp/dlOpCost.cpp
	> Author: wayne
	> Mail:  
	> Created Time: Thu 26 Sep 2019 09:54:45 AM UTC
 ************************************************************************/
#include "dlOp/dlOp.h"
#include "basicOp/basicOps.h"
#include "graphIR/IRNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"

#include "op/comSizeModel.h"

#include <sstream>
#include <algorithm>

#define comSizeModel comSizeModel2

namespace swc {
namespace op {


template <typename T>
std::string dumpDims(std::vector<T> vec) {
    if(vec.size() == 0)
        return "()";
    std::ostringstream stream;
    stream << "(";
    for(auto i : vec) {
        stream << i << ",";
    }
    std::string str = stream.str();
    return str.substr(0, str.length() - 1) + ")";
}

// config tell us about network topology etc.
size_t ScatterOp::getCost(OpNode *node, Config& config){
    (void) config;
    auto *from = (TensorNode*)node->getParentNode(0);
    size_t size = from->getTensor()->getSizeInBytes(); 

    int axis = this->axis_;
    //  i: master send size/p to each worker
    // -1: master broadcast size to all workers
    if(axis == -1)
        return comSizeModel(size, BCAST, config);
    if(axis >= 0)
        return comSizeModel(size, SCATTER, config);

    return 0;
}

// axis(strategy): -1 rep , i scatter 
size_t ScatterOp::getSimCost(size_t bytes, Config& config, int axis) {
    (void)config;
    (void)axis;
    return comSizeModel(bytes, SCATTER, config);
    if(axis == -1)
        return comSizeModel(bytes, BCAST, config);
    if(axis >= 0)
        return comSizeModel(bytes, SCATTER, config);
}

std::string ScatterOp::getCostTrace(OpNode *node, Config& config){
    std::ostringstream stream;

    std::string name = node->name();
    auto *from = (TensorNode*)node->getParentNode(0);
    auto *to = (TensorNode*)node->getChildNode(0);
    
    size_t  comm = from->getTensor()->getSizeInBytes();

    size_t  norm_comm = getCost(node, config); 

    int axis = this->axis_;

    if(axis == -1)
        stream << name << " " << "Bcast" << " " << comm << " " << norm_comm << " ";

    if(axis >= 0)
        stream << name << " " << "Send" << " " << comm << " " << norm_comm << " ";

    stream << dumpDims(from->getDims()) << " "
        << dumpDims(to->getDims()) << " "
        << "_" << " " << axis << "\n";

    return stream.str();
}

size_t GatherOp::getCost(OpNode *node, Config& config) {
    (void) config;
    auto *to = (TensorNode*)node->getChildNode(0);
    size_t size = to->getTensor()->getSizeInBytes(); 

    //  i: master recv size/p to each worker
    // -2: master reduce size to all workers
    return comSizeModel(size, GATHER, config);
}

std::string GatherOp::getCostTrace(OpNode *node, Config& config) {
    (void)config;

    std::ostringstream stream;

    std::string name = node->name();
    auto *from = (TensorNode*)node->getParentNode(0);
    auto *to = (TensorNode*)node->getChildNode(0);

    size_t  comm = to->getTensor()->getSizeInBytes();

    size_t  norm_comm = getCost(node, config); 

    int axis = this->axis_;

    if(axis == -2)
    stream << name << " " << "Reduce"<< " " << comm << " " << norm_comm << " ";

    if(axis >= 0)
        stream << name << " " << "Recv" << " " << comm << " " << norm_comm << " ";

    stream << dumpDims(from->getDims()) << " "
        << dumpDims(to->getDims()) << " "
        << axis << " " << "_" << "\n";

    return stream.str();
}

size_t GatherOp::getSimCost(size_t bytes, Config& config, int axis) {
    (void) config;
    (void) axis;
    return comSizeModel(bytes, GATHER, config);
}

size_t TransformOp::getCost(OpNode *node, Config& config) {
    (void) config;
    auto *from = (TensorNode*)node->getParentNode(0);
    size_t size = from->getTensor()->getSizeInBytes(); 

    int pre = this->preAxis_;   // pre {i, -1, -2}
    int post = this->postAxis_; // post{i, -1}
    // int p = this->degree_;

    size_t comm = 0;

    // i->j: each work send and recv size in total (pieces of data) 
    if(pre>=0 && post>=0)
        comm = comSizeModel(size, RECV_SEND, config);

    // i->j: master recv size, then broadcast size
    if(pre>=0 && post==-1)
        comm = comSizeModel(size, RECV_BCAST, config);

    // -2->i: a.master reduce then send size/p to each worker
    //  or b.each worker reduce its own part(not continuous)
    //  we took a but comm has two parts
    if(pre==-2 && post>=0)
        comm = comSizeModel(size, REDUCE_SEND, config);

    // -2->-1 a. allreduce size
    // or b. master reduce size, then broadcast size
    if(pre==-2 && post==-1)
        comm = comSizeModel(size, REDUCE_BCAST, config);

    // -1->i a. each worker already has replicate,  
    // if continuous, just let post pointer reference to its own part
    // if strided, just memcpy or self sendrecv is ok 
    if(pre==-1 && post>=0)
        comm = comSizeModel(size, SELF_CP, config);
    
    return comm;
}



std::string TransformOp::getCostTrace(OpNode *node, Config& config) {
    (void)config;
    std::ostringstream stream;

    std::string name = node->name();

    auto *from = (TensorNode*)node->getParentNode(0);
    auto *to = (TensorNode*)node->getChildNode(0);
    size_t size = from->getTensor()->getSizeInBytes(); 

    int pre = this->preAxis_;   // pre {i, -1, -2}
    int post = this->postAxis_; // post{i, -1}
    int p = this->degree_;

    stream << dumpDims(from->getDims()) << " "
        << dumpDims(to->getDims()) << " "
        << pre << " " << post;
    std::string suffix = stream.str();

    stream.str("");

    size_t norm_comm = getCost(node, config);

    // i->j: each work send and recv size in total (pieces of data) 
    if(pre>=0 && post>=0) {
        stream << name << " " << "SendRecv " << size << " " << norm_comm << " " << suffix << "\n";
    }

    // i->j: master recv size, then broadcast size
    if(pre>=0 && post==-1) {
        stream << name << " " << "Recv " << size << " " << "0" << " " << suffix << "\n"
            << name << " " << "Bcast " << size << " " << norm_comm << " " << suffix << "\n";
            
    }

    // -2->i: a.master reduce then send size/p to each worker
    //  or b.each worker reduce its own part(not continuous)
    //  we took a but comm has two parts
    if(pre==-2 && post>=0) {
        stream << name << " " << "Reduce " << size << " " << "0" << " " << suffix << "\n"
            << name << " " << "Send " << size << " " << norm_comm << " " << suffix << "\n";
    }

    // -2->-1 a. allreduce size
    // or b. master reduce size, then broadcast size
    if(pre==-2 && post==-1) {
        stream << name << " " << "Reduce " << size << " " << "0" << " " << suffix << "\n"
            << name << " " << "Bcast " << size << " " << norm_comm << " " << suffix << "\n";
        // stream << name << " " << "AllReduce," << size << "\n";
    }

    // -1->i a. each worker already has replicate,  
    // if continuous, just let post pointer reference to its own part
    // if strided, just memcpy or self sendrecv is ok 
    if(pre==-1 && post>=0) {
        stream << name << " " << "SelfSR " << size/p << " " << norm_comm << " " << suffix << "\n";
    }
    
    return stream.str();
}

// mind that size is original tensor
// actual size of para_tensor should be size/p
size_t TransformOp::getSimCost(size_t bytes, Config& config, int pre, int post) {
    int degree = config.mpi_size; 

    // para_tensor size
    size_t size = pre==-2 ? bytes : bytes / degree;

    size_t comm = 0;
    // i->j: each work send and recv size in total (pieces of data) 
    if(pre>=0 && post>=0)
        comm = comSizeModel(size, RECV_SEND, config);

    // i->j: master recv size, then broadcast size
    if(pre>=0 && post==-1)
        comm = comSizeModel(size, RECV_BCAST, config);

    // -2->i: a.master reduce then send size/p to each worker
    //  or b.each worker reduce its own part(not continuous)
    //  we took a, but comm has two parts
    if(pre==-2 && post>=0)
        comm = comSizeModel(size, REDUCE_SEND, config);

    // -2->-1 a. allreduce size
    // or b. master reduce size, then broadcast size
    if(pre==-2 && post==-1)
        comm = comSizeModel(size, REDUCE_BCAST, config);

    // -1->i a. each worker already has replicate,  
    // if continuous, just let post pointer reference to its own part
    // if strided, just memcpy or self sendrecv is ok 
    if(pre==-1 && post>=0)
        comm = comSizeModel(size, SELF_CP, config);
    
    return comm;
    
}

size_t ReduceOp::getCost(OpNode *node, Config& config) {
    (void)config;
    auto *to = (TensorNode*)node->getChildNode(0);
    size_t size = to->getTensor()->getSizeInBytes(); 

    // -2: master reduce size to all workers
    return comSizeModel(size, REDUCE, config);
}

std::string ReduceOp::getCostTrace(OpNode *node, Config& config) {
    (void)config;

    std::ostringstream stream;

    std::string name = node->name();
    auto *from = (TensorNode*)node->getParentNode(0);
    auto *to = (TensorNode*)node->getChildNode(0);

    size_t comm = to->getTensor()->getSizeInBytes(); 

    size_t norm_comm = getCost(node, config);

    stream << name << " " << "Reduce"<< " " << comm << " " << norm_comm << " ";


    stream << dumpDims(from->getDims()) << " "
        << dumpDims(to->getDims()) << " "
        << -2 << " " << "_" << "\n";

    return stream.str();
}

size_t ReduceOp::getSimCost(size_t bytes, Config& config, int axis) {
    (void) axis;
    (void) config;
    return comSizeModel(bytes, REDUCE, config);
}

size_t BroadcastOp::getCost(OpNode *node, Config& config) {
    (void)config;
    auto *to = (TensorNode*)node->getChildNode(0);
    size_t size = to->getTensor()->getSizeInBytes(); 

    // -1: master bcast size to all workers
    return comSizeModel(size, BCAST, config);
}

std::string BroadcastOp::getCostTrace(OpNode *node, Config& config) {
    (void)config;

    std::ostringstream stream;

    std::string name = node->name();
    auto *from = (TensorNode*)node->getParentNode(0);
    auto *to = (TensorNode*)node->getChildNode(0);

    size_t comm = to->getTensor()->getSizeInBytes(); 

    size_t  norm_comm = getCost(node, config); 

    stream << name << " " << "Bcast"<< " " << comm << " " << norm_comm << " ";


    stream << dumpDims(from->getDims()) << " "
        << dumpDims(to->getDims()) << " "
        << -2 << " " << "_" << "\n";

    return stream.str();
}

size_t BroadcastOp::getSimCost(size_t bytes, Config& config, int axis) {
    (void) axis;
    (void) config;
    return comSizeModel(bytes, BCAST, config);
}

} // namespace op
} // namespace swc
