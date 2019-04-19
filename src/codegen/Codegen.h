/*************************************************************************
    > File Name: Codegen.h
    > Author: wayne
    > Mail:
    > Created Time: 二  1/22 10:18:36 2019
 ************************************************************************/
#ifndef _CODEGEN_H_
#define _CODEGEN_H_

#include "MemoryAllocator.h"
#include "common.h"
#include <set>
#include <sstream>
#include <unordered_map>

namespace swc {
class IRGraph;
class TensorNode;
class OpNode;
class Tensor;
} // namespace swc

namespace swc {
namespace codegen {

class Codegen {
  public:
    Codegen() {}
    Codegen(IRGraph *graph) : graph_(graph) {}
    ~Codegen() { destroy(); }

    std::string UniqueName(std::string name);

    void codeGenInit();

    void emitCUDAInit();

    // creaet allocators for devices according to config
    // and set baseptr name
    // build device-allocator map
    void initMemoryAllocators();

    // generate code for IRGraph
    std::string generate();

    //----------------------------------------------------------
    // generate malloc for tensor data
    void emitMemAllocs();
    void emitMemFree();

    // build tensor*-<base, offset> map
    void allocateMemAddr();
    void allocateMemAddr(IRGraph *graph_);

    // declare var before alloc for mpi (if needed)
    void emitVarDeclarations();
    // allocate statement of cpu/gpu mem
    void emitMemAllocations();

    // initialize tensors
    // data0 = cpu0_baseptr + addr;
    // load(data0, 6272, 0, "mnist_images_8.bin");
    void emitTensorInitializations();
    void emitTensorInitializations(IRGraph *graph_,
                                   std::set<Tensor *> *visited);

    // may need to allocate for specific tensornode (e.g. different data type)
    std::string emitTensorMemAlloc(TensorNode *tnode);
    //----------------------------------------------------------
    // generate function calls for opNodes
    void emitFuncCalls();
    void emitFuncCalls(IRGraph *graph_);

    void emitFuncCall(OpNode *op);
    void emitFuncCallCUDA(OpNode *op);

    void switchTo(IRGraph *subGraph);
    void switchFrom(IRGraph *subGraph);

    void dispathOpNode(OpNode *op);
    void emitMemcpyFromTo(Tensor *from, Device from_dev, size_t offset,
                          size_t size, Tensor *to, Device to_dev);

    // specialize template< typename Dtype>
    std::string dtype();

    // finish codegen
    void Finish();

  private:
    void destroy();
    void genIndent() {
        for (int i = 0; i < indent_; i++)
            stream_ << "    ";
    }

    std::ostringstream stream_;
    int indent_;
    IRGraph *graph_;
    IRGraph *active_graph_;
    bool flag_multiGPU{true};
    bool flag_multiStream{true};
    bool flag_MPI{false};
    bool flag_use_cublas{true};
    std::unordered_map<std::string, int> names_map_;
    std::vector<std::shared_ptr<MemoryAllocator>> mem_allocators_;

    std::unordered_map<Tensor *, std::string> tensors_name_map_;
    std::unordered_map<Tensor *, std::pair<std::string, uint64_t>>
        tensors_offset_map_;
    // std::unordered_map<Tensor*, std::string> tensors_base_map_;

    std::unordered_map<Device, MemoryAllocator *> dev_allocator_map_;
    std::unordered_map<MemoryAllocator *, std::string> allocator_membase_map_;
};

} // namespace codegen
} // namespace swc

#endif
