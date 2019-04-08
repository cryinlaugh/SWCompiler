/*************************************************************************
    > File Name: Codegen.h
    > Author: wayne
    > Mail:
    > Created Time: 二  1/22 10:18:36 2019
 ************************************************************************/
#ifndef _CODEGEN_H_
#define _CODEGEN_H_

#include <unordered_map>
#include <set>
#include <sstream>
#include "common.h"
#include "MemoryAllocator.h"

namespace swc{
    template <typename Dtype> class IRGraph;
    template <typename Dtype> class TensorNode;
    template <typename Dtype> class OpNode;
    template <typename Dtype> class Tensor;
} // namespace swc

namespace swc{
namespace codegen{

template <typename Dtype>
class Codegen{
public:
    Codegen(){}
    Codegen(IRGraph<Dtype>* graph) : graph_(graph){}
    ~Codegen(){ destroy(); }

    std::string UniqueName(std::string name);

    void codeGenInit();

    // generate code for IRGraph
    std::string generate();
    //generate function calls for opNodes
    void genFuncCalls();
    void genFuncCalls(IRGraph<Dtype>* graph_);

    void genFuncCall(OpNode<Dtype>* op);
    void genFuncCallCUDA(OpNode<Dtype>* op);

    // generate malloc for tensor data
    void genMemAllocs();
    void genMemFree();

    std::string genTensorMemAlloc(TensorNode<Dtype>* tnode);

    // creaet allocators for devices according to config
    // and set baseptr name
    // build device-allocator map
    void initMemoryAllocators();

    // build tensor*-<base, offset> map
    void allocateMemAddr();
    void allocateMemAddr(IRGraph<Dtype>* graph_);

    // declare var before alloc for mpi (if needed)
    void emitVarDeclarations();

    // allocate statement of cpu/gpu mem
    void emitMemAllocations();

    // initialize tensors
    // data0 = cpu0_baseptr + addr;
    // load(data0, 6272, 0, "mnist_images_8.bin");
    void emitTensorInitializations();
    void emitTensorInitializations(IRGraph<Dtype>* graph_, std::set<Tensor<Dtype>*>* visited);

    void switchTo(IRGraph<Dtype>* subGraph);
    void switchFrom(IRGraph<Dtype>* subGraph);

    void dispathOpNode(OpNode<Dtype>* op);
    void emitMemcpyFromTo(Tensor<Dtype>* from, Device from_dev, size_t offset, size_t size,
            Tensor<Dtype>* to, Device to_dev);
    void emitCUDAInit();

    // specialize template< typename Dtype>
    std::string dtype();

    // finish codegen
    void Finish();
private:

    void destroy();
    void genIndent(){
        for(int i=0; i<indent_; i++)
            stream_ << "    ";
    }
    std::ostringstream stream_;
    int indent_;
    IRGraph<Dtype>* graph_;
    IRGraph<Dtype>* active_graph_;
    bool flag_multiGPU {false};
    bool flag_multiStream {false};
    bool flag_MPI  {false};
    bool flag_use_cublas{false};
    std::unordered_map<std::string, int> names_map_;
    std::vector<std::shared_ptr<MemoryAllocator>> mem_allocators_;

    std::unordered_map<Tensor<Dtype>*, std::string> tensors_name_map_;
    std::unordered_map<Tensor<Dtype>*, std::pair<std::string, uint64_t>> tensors_offset_map_;
    // std::unordered_map<Tensor<Dtype>*, std::string> tensors_base_map_;

    std::unordered_map<Device, MemoryAllocator*> dev_allocator_map_;
    std::unordered_map<MemoryAllocator*, std::string> allocator_membase_map_;
};

} // namespace codegen
} // namespace swc

#endif
