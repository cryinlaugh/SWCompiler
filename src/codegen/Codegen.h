/*************************************************************************
    > File Name: Codegen.h
    > Author: wayne
    > Mail:
    > Created Time: 二  1/22 10:18:36 2019
 ************************************************************************/
#ifndef _CODEGEN_H_
#define _CODEGEN_H_

#include "CodeWriter.h"
#include "MakefileBuilder.h"
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
/**
 *   \brief generate C++/CUDA C source form IRGraph
 */
class Codegen {
  public:
    Codegen() {}
    Codegen(IRGraph *graph) : graph_(graph) {}
    /// to be depreciated
    Codegen(IRGraph *graph, CodegenConfig &config) : graph_(graph) {
        config_ = config;
        /*
        flag_multiGPU = config.flag_multiGPU;
        flag_multiStream = config.flag_multiStream;
        flag_MPI = config.flag_MPI;
        flag_use_cublas = config.flag_use_cublas;
        */
    }
    ~Codegen() { destroy(); }

    /// ensure each Variable (Tensor) has unique name
    /// \param name name of TensorNode, variable, whatever
    std::string UniqueName(std::string name);

    /// emit gflags definition
    void emitGflagsDef() {
        writer_
            << R"(DEFINE_string(snapshot, "", "Optional; the snapshot to resume training.");)"
            << "\n";
    }

    void emitDataLoaderInit();

    /// initialization before code emitting
    void codeGenInit();

    /// emit CUDA related code. e.g. cuBlas handle, cudaStream creating
    void emitCUDAInit();

    /// create allocators for devices according to config
    /// and set baseptr name
    /// build device-allocator map
    void initMemoryAllocators();

    /// generate code for IRGraph
    std::string generate();

    //----------------------------------------------------------
    /** \brief generate malloc for tensor data
     *
     *   step1: collect overrall dev mem requirements and variable offsets\n
     *   step2: emit mem allocation statements\n
     *   step3: emit tensor variable declarations and initializations
     */
    void emitMemAllocs();
    /// emit free memory codes
    void emitMemFree();

    /// build Tensor* -> <base, offset> map for L1 Graph
    void allocateMemAddr();
    /// build Tensor* -> <base, offset> map for L2(Device) subGraph
    void allocateMemAddr(IRGraph *graph_);

    /// declare var before alloc for mpi (if needed)
    void emitVarDeclarations();
    /// allocate statement of cpu/gpu mem
    void emitMemAllocations();

    /// initialize tensors for L1 IRGraph

    void emitTensorAddresses();
    /// initialize tensors for L2(Device) subGraph
    void emitTensorAddresses(IRGraph *graph_, std::set<Tensor *> *visited);

    /// data0 = cpu0_baseptr + addr;
    /// load(data0, 6272, 0, "mnist_images_8.bin");
    void emitTensorInitializations();
    /// initialize tensors for L2(Device) subGraph
    void emitTensorInitializations(IRGraph *graph_,
                                   std::set<Tensor *> *visited);

    void emitTensorInitFromSnapshot(IRGraph *graph_,
                                    std::set<Tensor *> *visited);
    void emitSaveSnapshot();

    /// may need to allocate for specific tensornode (e.g. different data type)
    std::string emitTensorMemAlloc(TensorNode *tnode);
    //----------------------------------------------------------
    void emitExecute();
    /// generate function calls for opNodes
    void emitFuncCalls();
    /// generate function calls for opNodes of L2(Device) subGraph
    void emitFuncCalls(IRGraph *graph_);

    /// generate function call for opNode
    void emitFuncCall(OpNode *op);
    /// generate CUDA kernel function call for opNode
    void emitFuncCallCUDA(OpNode *op);

    /// context swith to device e.g. cudaSetDevice()
    void switchTo(IRGraph *subGraph);
    /// context Swith back to L1 IRGraph
    void switchFrom(IRGraph *subGraph);

    /// dispatch OpNode for memcpy or kernel func call
    void dispathOpNode(OpNode *op);
    void emitMemcpyFromTo(Tensor *from, Device from_dev, size_t from_offset,
                          size_t size, Tensor *to, Device to_dev,
                          size_t to_offset);
    /// to be depreciated
    std::string dtype();

    /// finish codegen
    void finish() {}

    void emitMPIInit();
    void emitMPIFinalize();

    int getMPISendRecvTag(Tensor *);
    bool delMPISendRecvTag(Tensor *);

  private:
    void destroy();

    std::string getTypeString(Tensor *);

    CodeWriter writer_;
    MakefileBuilder makefile_builder_;
    IRGraph *graph_;
    IRGraph *active_graph_;

    CodegenConfig config_;
    /*
    bool flag_multiGPU{false};
    bool flag_multiStream{false};
    bool flag_MPI{true};
    bool flag_use_cublas{false};
    */

    std::unordered_map<std::string, int> names_map_;
    std::vector<std::shared_ptr<MemoryAllocator>> mem_allocators_;

    std::unordered_map<Tensor *, std::string> tensors_name_map_;
    std::unordered_map<Tensor *, std::pair<std::string, uint64_t>>
        tensors_offset_map_;
    // std::unordered_map<Tensor*, std::string> tensors_base_map_;
    std::vector<Tensor *> mpi_sendRecv_tags_;

    std::unordered_map<Device, MemoryAllocator *> dev_allocator_map_;
    std::unordered_map<MemoryAllocator *, std::string> allocator_membase_map_;
};

} // namespace codegen
} // namespace swc

#endif
