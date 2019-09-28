/*************************************************************************
	> File Name: src/op/dlOp/dlOpCost.cpp
	> Author: wayne
	> Mail:  
	> Created Time: Thu 26 Sep 2019 09:54:45 AM UTC
 ************************************************************************/
#include "dlOp/dlOp.h"
#include "basicOp/basicOps.h"
#include "../graphIR/IRNode.h"
#include "../graphIR/OpNode.h"
#include "../graphIR/TensorNode.h"
#include <sstream>
#include <algorithm>

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

size_t ScatterOp::getCost(OpNode *node){
    auto *from = (TensorNode*)node->getParentNode(0);
    size_t size = from->getTensor()->getSizeInBytes(); 

    //  i: master send size/p to each worker
    // -2: master broadcast size to all workers
    return size; 
}

// axis(strategy): -1 rep , i scatter 
size_t ScatterOp::getSimCost(size_t bytes, int degree, int axis) {
    (void)degree;
    (void)axis;
    return bytes; 
}

std::string ScatterOp::getCostTrace(OpNode *node){
    std::ostringstream stream;

    std::string name = node->name();
    auto *from = (TensorNode*)node->getParentNode(0);
    auto *to = (TensorNode*)node->getChildNode(0);
    
    size_t  comm = getCost(node);

    int axis = this->axis_;

    if(axis == -1)
        stream << name << " " << "Bcast" << " " << comm << " ";

    if(axis >= 0)
        stream << name << " " << "Send" << " " << comm << " ";

    stream << dumpDims(from->getDims()) << " "
        << dumpDims(to->getDims()) << " "
        << "_" << " " << axis << "\n";

    return stream.str();
}

size_t GatherOp::getCost(OpNode *node) {
    auto *to = (TensorNode*)node->getChildNode(0);
    size_t size = to->getTensor()->getSizeInBytes(); 

    //  i: master recv size/p to each worker
    // -2: master reduce size to all workers
    return size; 
}

std::string GatherOp::getCostTrace(OpNode *node) {
    std::ostringstream stream;

    std::string name = node->name();
    auto *from = (TensorNode*)node->getParentNode(0);
    auto *to = (TensorNode*)node->getChildNode(0);

    size_t  comm = getCost(node);

    int axis = this->axis_;

    if(axis == -2)
    stream << name << " " << "Reduce"<< " " << comm << " ";

    if(axis >= 0)
        stream << name << " " << "Recv" << " " << comm << " ";

    stream << dumpDims(from->getDims()) << " "
        << dumpDims(to->getDims()) << " "
        << axis << " " << "_" << "\n";

    return stream.str();
}

size_t GatherOp::getSimCost(size_t bytes, int degree, int axis) {
    (void) axis;
    (void) degree;
    return bytes;
}

size_t TransformOp::getCost(OpNode *node) {
    auto *from = (TensorNode*)node->getParentNode(0);
    size_t size = from->getTensor()->getSizeInBytes(); 

    int pre = this->preAxis_;   // pre {i, -1, -2}
    int post = this->postAxis_; // post{i, -1}
    int p = this->degree_;

    size_t comm = 0;

    // i->j: each work send and recv size in total (pieces of data) 
    if(pre>=0 && post>=0)
        comm = size;

    // i->j: master recv size, then broadcast size
    if(pre>=0 && post==-1)
        comm = size*2;

    // -2->i: a.master reduce then send size/p to each worker
    //  or b.each worker reduce its own part(not continuous)
    //  we took a but comm has two parts
    if(pre==-2 && post>=0)
        comm = size*2;   

    // -2->-1 a. allreduce size
    // or b. master reduce size, then broadcast size
    if(pre==-2 && post==-1)
        comm = size;// TODO, 

    // -1->i a. each worker already has replicate,  
    // if continuous, just let post pointer reference to its own part
    // if strided, just memcpy or self sendrecv is ok 
    if(pre==-1 && post>=0)
        comm = size/p; 
    
    return comm;
}



std::string TransformOp::getCostTrace(OpNode *node) {
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

    // i->j: each work send and recv size in total (pieces of data) 
    if(pre>=0 && post>=0) {
        stream << name << " " << "SendRecv " << size << " " << suffix << "\n";
    }

    // i->j: master recv size, then broadcast size
    if(pre>=0 && post==-1) {
        stream << name << " " << "Recv " << size << "\n"
            << name << " " << "Bcast " << size << " " << suffix << "\n";
            
    }

    // -2->i: a.master reduce then send size/p to each worker
    //  or b.each worker reduce its own part(not continuous)
    //  we took a but comm has two parts
    if(pre==-2 && post>=0) {
        stream << name << " " << "Reduce " << size << " " << suffix << "\n"
            << name << " " << "Send " << size << " " << suffix << "\n";
    }

    // -2->-1 a. allreduce size
    // or b. master reduce size, then broadcast size
    if(pre==-2 && post==-1) {
        stream << name << " " << "Reduce " << size << " " << suffix << "\n"
            << name << " " << "Bcast " << size << " " << suffix << "\n";

        // stream << name << " " << "AllReduce," << size << "\n";
    }

    // -1->i a. each worker already has replicate,  
    // if continuous, just let post pointer reference to its own part
    // if strided, just memcpy or self sendrecv is ok 
    if(pre==-1 && post>=0) {
        stream << name << " " << "SelfSR " << size/p << " " << suffix << "\n";
    }
    
    return stream.str();
}

// mind that size is original tensor
// actual size of para_tensor should be size/p
size_t TransformOp::getSimCost(size_t bytes, int degree, int pre, int post) {
    // para_tensor size
    size_t size = bytes / degree;

    size_t comm = 0;
    // i->j: each work send and recv size in total (pieces of data) 
    if(pre>=0 && post>=0)
        comm = size;

    // i->j: master recv size, then broadcast size
    if(pre>=0 && post==-1)
        comm = size*2;

    // -2->i: a.master reduce then send size/p to each worker
    //  or b.each worker reduce its own part(not continuous)
    //  we took a, but comm has two parts
    if(pre==-2 && post>=0)
        comm = size*2;   

    // -2->-1 a. allreduce size
    // or b. master reduce size, then broadcast size
    if(pre==-2 && post==-1)
        comm = size;// TODO, 

    // -1->i a. each worker already has replicate,  
    // if continuous, just let post pointer reference to its own part
    // if strided, just memcpy or self sendrecv is ok 
    if(pre==-1 && post>=0)
        comm = size/degree; 
    
    return comm;
    
}


} // namespace op
} // namespace swc
