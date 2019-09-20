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
#include <map>

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
    Codegen(IRGraph *graph);
    /// to be depreciated
    Codegen(IRGraph *graph, Config &config);

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

    /// init Makefile wrt. CodegenConfig
    virtual void initMakefileBuilder();

    /// initialization before code emitting
    void codeGenInit();

    /// emit CUDA related code. e.g. cuBlas handle, cudaStream creating
    void emitCUDAInit();

    /// Dataloader for batch executions
    void emitDataLoaderInit();

    /// create allocators for devices according to config
    /// and set baseptr name
    /// build device-allocator map
    void initMemoryAllocators();

    /// generate code for IRGraph
    virtual void emitHeader();
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
    virtual void emitMemFree();

    /// build Tensor* -> <base, offset> map for L1 Graph
    virtual void allocateMemAddr();
    /// build Tensor* -> <base, offset> map for L2(Device) subGraph
    void allocateMemAddr(IRGraph *graph_);

    /// declare var before alloc for mpi (if needed)
    virtual void emitVarDeclarations();
    /// allocate statement of cpu/gpu mem
    virtual void emitMemAllocations();
    void emitMemAllocation(std::string buffer, size_t bytes, Device& dev);

    /// initialize tensors for L1 IRGraph
    virtual void emitTensorAddresses();
    /// initialize tensors for L2(Device) subGraph
    void emitTensorAddresses(IRGraph *graph_, std::set<Tensor *> *visited);

    /// data0 = cpu0_baseptr + addr;
    /// load(data0, 6272, 0, "mnist_images_8.bin");
    virtual void emitTensorInitializations();
    /// initialize tensors for L2(Device) subGraph
    void emitTensorInitializations(IRGraph *graph_,
                                   std::set<Tensor *> *visited);

    /// snapshot
    void emitTensorInitFromSnapshot(IRGraph *graph_,
                                    std::set<Tensor *> *visited);
    void emitSaveSnapshot();

    /// print outputs(loss, prob etc.)
    void emitPrintGraphOutputs();

    /// may need to allocate for specific tensornode (e.g. different data type)
    std::string emitTensorMemAlloc(TensorNode *tnode);
    //----------------------------------------------------------
    void emitExecute();
    /// generate function calls for opNodes
    virtual void emitFuncCalls();
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
    void dispatchOpNode(OpNode *op);
    void emitMemcpyFromTo(Tensor *from, Device from_dev, size_t from_offset,
                          size_t size, Tensor *to, Device to_dev,
                          size_t to_offset);
    /// to be depreciated
    std::string dtype();

    /// finish codegen
    void finish() {}

    virtual void emitEnvInit();
    virtual void emitEnvFinalize();


    // int getMPISendRecvTag(Tensor *);
    // bool delMPISendRecvTag(Tensor *);

protected:
    void destroy();

    std::string getTypeString(Tensor *);

    CodeWriter headerWriter_;
    CodeWriter writer_;
    MakefileBuilder makefile_builder_;

    IRGraph *graph_;

    Config config_;

    std::unordered_map<std::string, int> names_map_;

    std::vector<std::shared_ptr<MemoryAllocator>> mem_allocators_;
    /// For parallel
    std::shared_ptr<MemoryAllocator> p_mem_alllocator_;

    std::map<Tensor *, std::string> tensors_name_map_;
    std::map<Tensor *, std::pair<std::string, uint64_t>>
        tensors_offset_map_;


    /// to use Device as key, we implement std::hash() of Device in common.h
    /// if implemented with std::map, we must define comparison of Device
    std::unordered_map<Device, MemoryAllocator *> dev_allocator_map_;
};


class ParallelCodegen : public Codegen {
public:
    ParallelCodegen(IRGraph *graph, Config &config) : Codegen(graph, config) { }

    /// add mpi header
    void emitHeader() override;

    void allocateMemAddr() override;
    void emitVarDeclarations() override;
    void emitMemAllocations() override;
    void emitMemFree() override;
    void emitMemFree(std::string name, Device dev);
    void emitTensorAddresses() override;
    void emitTensorInitializations() override;
    void emitTensorInitialization(TensorNode* tnode);
    void emitFuncCalls() override;
    // void emitFuncCall(OpNode *op, CodeWriter& writer);
    void heteroBegin();
    void heteroEnd();
    void dispatchOpNode(OpNode *op, int side=-1/*0:master, ~0: worker*/);

    void emitEnvInit() override;
    void emitEnvFinalize() override;

    void initMakefileBuilder() override;

    void emitMPIInit();
    void emitMPIFinalize();
private:
    CodeWriter masterWriter_; // rank == 0
    CodeWriter workerWriter_; // rank!=0
    bool hetero_pending_{false};

    std::vector<TensorNode *> _master_tensors;
    std::vector<TensorNode *> _parallel_tensors;

    // std::unordered_map<Tensor*, std::string> tensors_base_map_;
    std::vector<Tensor *> mpi_sendRecv_tags_;
    
    // std::vector<OpNode*> _scheduled_opnodes;
    void masterWorkerDispatcher(OpNode *node, int side/*master:0, worker:1*/);
    void transformOpDispatcher(OpNode *node);
    void reduceOpDispatcher(OpNode *node);
    int getMPISendRecvTag(Tensor *);
    bool delMPISendRecvTag(Tensor *);
    std::vector<OpNode*> schedule();
};

} // namespace codegen
} // namespace swc

#endif
