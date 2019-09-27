/*************************************************************************
    > File Name: Codegen.cpp
    > Author: wayne
    > Mail:
    > Created Time: 二  1/22 10:32:13 2019
 ************************************************************************/

#include "Codegen.h"
#include "SWC.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_set>
#include <limits>

using namespace swc::op;

namespace swc {
namespace codegen {

static std::string deviceToStr(const Device &d) {
    std::ostringstream os;
    if(d.rank == INT_MAX) {
        return "para";
    }
    if (d.type == DeviceType::CPU) {
            os << "cpu" << d.id;
    } else if (d.type == DeviceType::GPU) {
        os << "gpu" << d.id;
    }
    return os.str();
}

static std::pair<size_t, size_t>
convertToDim2(const std::vector<size_t> &dims) {
    size_t second = 1;
    for (size_t i = 1; i < dims.size(); i++)
        second *= dims[i];

    return std::make_pair(dims[0], second);
}

static std::string emitArrayDefAndInit(std::string name,
                                       const std::vector<size_t> &dims) {
    std::ostringstream os;
    os << "const size_t " << name << "[] = {";
    for (auto dim : dims)
        os << dim << ", ";

    std::string str = os.str();
    return str.substr(0, str.length() - 2) + "};\n";
}

std::string Codegen::getTypeString(Tensor *tensor) {
    switch (tensor->getDataType()) {
    case DataType::Float_t:
        return "float";
    case DataType::Double_t:
        return "double";
    case DataType::Int32_t:
        return "int";
    default:
        SWLOG_ERROR << "UNKNOWN DataType\n";
        return nullptr;
    }
}

Codegen::Codegen(IRGraph *graph) : graph_(graph) {
    config_ = graph->getConfig();
}
Codegen::Codegen(IRGraph *graph, Config &config) {
    graph_ = graph; 
    graph_->setConfig(config);
    config_ = config;
}

void Codegen::destroy() {
    graph_ = nullptr;
    names_map_.clear();
    tensors_name_map_.clear();
    tensors_offset_map_.clear();
}

void Codegen::initMemoryAllocators() {
    Device cpu0;
    Device gpu0;
    gpu0.type = DeviceType::GPU;
    gpu0.id = 0;
    Device gpu1;
    gpu1.type = DeviceType::GPU;
    gpu1.id = 1;

    Device cpup;
    cpup.rank = INT_MAX;

    auto m_cpu0 = std::make_shared<MemoryAllocator>(cpu0, "cpu0", 0xFFFFFFFF);
    auto m_gpu0 = std::make_shared<MemoryAllocator>(gpu0, "gpu0", 0xFFFFFFFF);
    auto m_gpu1 = std::make_shared<MemoryAllocator>(gpu1, "gpu1", 0xFFFFFFFF);
    p_mem_alllocator_ = std::make_shared<MemoryAllocator>(cpup, "cpup", 0xFFFFFFFF);

    mem_allocators_.push_back(m_cpu0);
    mem_allocators_.push_back(m_gpu0);
    mem_allocators_.push_back(m_gpu1);

    dev_allocator_map_[cpu0] = m_cpu0.get();
    dev_allocator_map_[gpu0] = m_gpu0.get();
    dev_allocator_map_[gpu1] = m_gpu1.get();
    dev_allocator_map_[cpup] = p_mem_alllocator_.get();

    m_cpu0->setBasePtrName(UniqueName(deviceToStr(cpu0) + "_baseptr"));
    p_mem_alllocator_->setBasePtrName(UniqueName(deviceToStr(cpup) + "_baseptr"));

    m_gpu0->setBasePtrName(UniqueName(deviceToStr(gpu0) + "_baseptr"));
    m_gpu1->setBasePtrName(UniqueName(deviceToStr(gpu1) + "_baseptr"));
}
void Codegen::codeGenInit() {
    initMemoryAllocators();
    initMakefileBuilder();
}

void Codegen::emitEnvInit() {
    emitCUDAInit();
    if(config_.mkldnn)
        emitMKLDNNInit();
}

void Codegen::emitEnvFinalize() {
    writer_ << "return 0;\n";
}

void Codegen::emitMKLDNNInit() {
    //TODO multistream for multi cpu
    UniqueName("mkldnn_eng");
    UniqueName("mkldnn_s");
    writer_ << "engine mkldnn_eng(engine::kind::cpu, 0);\n"; 
    // TODO: name stream may conflict to cuda stream[]
    writer_ << "stream mkldnn_s(mkldnn_eng);\n";
}

void Codegen::emitCUDAInit() {
    // TODO create stream depending on number of device or config
    // one stream per device
    int N = 0;
    for (auto allocator : mem_allocators_) {
        Device dev = allocator->getDevice();
        if (dev.type == DeviceType::GPU)
            N++;
    }
    if (N == 0)
        return;

    if (config_.cublas) {
        UniqueName("stat");
        UniqueName("handle");

        writer_ << "cublasStatus_t stat;\n";
        writer_ << "cublasHandle_t handle;\n";
        writer_ << "stat = cublasCreate(&handle);\n";
        writer_ << "if (stat != CUBLAS_STATUS_SUCCESS) {\n";
        writer_ << "    printf (\"CUBLAS initialization failed\\n\");\n";
        writer_ << "    return EXIT_FAILURE;\n";
        writer_ << "}\n\n";
    }
    if (config_.cuda_stream) {
        writer_ << "cudaStream_t stream[" << N << "];\n";
        writer_ << "for(int i=0; i<" << N << "; i++)\n";
        writer_.indentInc();
        writer_ << "cudaStreamCreate(&stream[i]);\n\n";
        writer_.indentDec();
    }
}


std::string Codegen::UniqueName(std::string name) {
    auto iter = names_map_.find(name);
    if (iter != names_map_.end()) {
        std::string uname = name;
        std::ostringstream os;
        while (names_map_.count(uname) != 0) {
            os << name << (++iter->second);
            uname = os.str();
        }
        name = uname;
    }
    names_map_[name] = 0;
    return name;
}

void Codegen::initMakefileBuilder() {
   makefile_builder_.setCXXCompiler("/usr/bin/c++");

    makefile_builder_.addIncDir("/usr/local/include");
    makefile_builder_.addLibDir("/usr/local/lib");

    makefile_builder_.addCXXSrc("Graph.cpp");
    if(config_.train_mode) {
        makefile_builder_.addCXXSrc("utils/DataLoader.cpp");
        makefile_builder_.addCXXSrc("caffe2.pb.cc");

        makefile_builder_.addLib("protobuf");
        makefile_builder_.addLib("gflags");
        makefile_builder_.addLib("pthread");
    }

    if(config_.use_dataloader) {
        makefile_builder_.addCXXSrc("utils/DataLoader.cpp");
    }

    if(config_.mkldnn) {
        makefile_builder_.addIncDir("$(HOME)/install/mkldnn_lnx_1.0.2_cpu_gomp/include");
        makefile_builder_.addLibDir("$(HOME)/install/mkldnn_lnx_1.0.2_cpu_gomp/lib");
        makefile_builder_.addLib("mkldnn");
    }
}

void Codegen::emitHeader() {
    headerWriter_ << "/*******************************************************************"
          "******\n"
       << "  > File Name: Graph.cpp\n"
       // << "  > Author: none\n"
       // << "  > Mail:  \n"
       // << "  > Created Time: "
       << "  > IRGraph\n"
       << "  > |-TensorNode " << graph_->tensorNodeNum() << "\n"
       << "  > |-opNode     " << graph_->opNodeNum() << "\n"
       << " *******************************************************************"
          "*****/\n";

    headerWriter_<< "#include <iostream>\n"
            << "#include <random>\n"
            << "#include <stdlib.h>\n"
            << "#include <math.h>\n"
            << "#include <chrono>\n"
            << "#include \"utils/image.h\"\n";

    if(config_.mkldnn) {
        headerWriter_ <<"#include \"mkldnn.hpp\"\n";
    }

    //------------------TODO:BEGIN-------------------------------
    // TODO: mv mpi and cuda specific statements to derived class
    // UPDATE: mpi removed

    if (config_.cuda) {
        headerWriter_ << "#include <cuda.h>\n"
                << "#include <cublas_v2.h>\n";
        // #include "utils/cuda_kernels.cu"
        // headerWriter_ << CUDA_CODE;
        headerWriter_ << "#include \"utils/cuda_kernels.h\"\n";
    }
    //------------------TODO:END---------------------------------
    // #include "kernels.h"
    // headerWriter_ << KERNELS_CODE;
    headerWriter_ << "#include \"utils/kernels.h\"\n";

    if (config_.train_mode) {
        headerWriter_ << "#include \"utils/DataLoader.h\"\n"
            << "#include \"utils/utils.h\"\n";
    }

    if(config_.use_dataloader) {
        headerWriter_ << "#include \"utils/DataLoader.h\"\n";
    }

    // namespace
    headerWriter_ << "using namespace std;\n";
    if(config_.mkldnn) {
        headerWriter_ << "using namespace mkldnn;\n";
    }


    if (config_.train_mode) {
        headerWriter_ << "#include \"gflags/gflags.h\"\n"
                << "#include <google/protobuf/io/coded_stream.h>\n"
                << "#include <google/protobuf/io/zero_copy_stream_impl.h>\n\n";

        emitGflagsDef();
    }


}

std::string Codegen::generate() {
    codeGenInit();

    emitHeader();

    writer_ << "\n"
            << "int main(int argc, char** argv) {\n";
    writer_.indentInc();

    writer_ << "auto t_begin = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();\n";

    if (config_.train_mode) {
        writer_ << "gflags::ParseCommandLineFlags(&argc, &argv, true);\n";
    }

    emitEnvInit();

    writer_ << "\n// variable declaration and initiation\n";
    emitMemAllocs();

    writer_ << "\n// call op routine functions\n";
    emitExecute();

    writer_ << "\n// free memory\n";
    emitMemFree();

    writer_ << "auto t_end = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();\n";

    if(config_.mkldnn)
        writer_ << "std::cout << \"Net with MKLDNN run time \" << (t_end - t_begin) << \" ms\\n\";\n";
    else
        writer_ << "std::cout << \"Net run time \" << (t_end - t_begin) << \" ms\\n\";\n";

    emitEnvFinalize();

    writer_.indentDec();
    writer_ << "}\n";

    std::ofstream fout("Graph.cpp", std::fstream::out);
    fout << headerWriter_.get_code() + writer_.get_code();
    fout.close();

    fout.flush();


    fout.open("G_Makefile", std::fstream::out);
    fout << makefile_builder_.generate();
    fout.close();

    return headerWriter_.get_code() + writer_.get_code();
}

void Codegen::emitMemAllocs() {
    SWLOG_DEBUG(4) << "genMemAllocs \n";

    allocateMemAddr();

    emitVarDeclarations();

    emitMemAllocations();

    emitTensorAddresses();

    // emitDataLoader here, because if restore from snapshot,
    // dataloader.shift() will be called 
    if (config_.train_mode)
        emitDataLoaderInit();
    else if(config_.use_dataloader) {
        emitInferDataLoaderInit();
    }


    emitTensorInitializations();
}
void Codegen::allocateMemAddr() {
    SWLOG_DEBUG(4) << "begin allocateMemAddr...\n";

    allocateMemAddr(graph_);
    for (int i = 0; i < graph_->opNodeNum(); i++) {
        OpNode *opnode = graph_->getOpNode(i);
        if (auto graphOp = dynamic_cast<SubGraphOp *>(opnode->getOp())) {
            if (graphOp->getGraph())
                SWLOG_DEBUG(4)
                    << "allocateMemAddr on subG: " << opnode->name() << "\n";
            allocateMemAddr(graphOp->getGraph());
        }
    }
    SWLOG_DEBUG(4) << "end allocateMemAddr...\n";
}
void Codegen::allocateMemAddr(IRGraph *graph) {

	for (int i = 0; i < graph->topologyNum(); i++) {
        for (int j = 0; j < graph->getNumInTopoLevel(i); j++) {
            auto node = graph->getNodeInTopo(i, j);
            if (node->nodeType() == TENSOR_NODE) {
                auto *tnode = (TensorNode *)node;
                Tensor *tensor = tnode->getTensor();

                if (tensors_name_map_.count(tensor))
                    continue;

                std::string bufferName = UniqueName(tnode->name());

                size_t size = tensor->getSizeInBytes();

                Label *label = tnode->getLabel();
                Device dev = label->getDeviceLabel();

                SWLOG_DEBUG(1) << "allocateMemAddr topo(" << i
                               <<", " << j << ") "
                               << tnode->name() << " " << size
                               << " as " << bufferName
                               << " on dev(" << dev.rank << ", "
                               << static_cast<int>(dev.type) << ", "
                               << dev.id << ")."
                               << "\n";

                auto *allocator = dev_allocator_map_.at(dev);
                if (!allocator) {
                    SWLOG_ERROR << "allocator" << static_cast<int>(dev.type) << " "
                                << dev.id << " not found\n";
                }
                uint64_t addr = allocator->allocate(tensor, size);
                std::string base = allocator->getBasePtrName();

                tensors_name_map_[tensor] = bufferName;
                // TODO data type
                tensors_offset_map_[tensor] = std::make_pair(base, addr);
            }  // if tensor node
        } // for topo j
    } // for topo i
}

void Codegen::emitVarDeclarations() {
    SWLOG_DEBUG(4) << "begin emitVarDeclarations...\n";

    // std::string dtype = this->dtype();
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (size == 0)
            continue;
        // writer_ << dtype << " *" << base << ";\n";
        writer_ << "char *" << base << ";\n";
    }

    /*
    for(int i=0 ; i<graph_->tensorNodeNum(); i++){
        auto *tensor = graph_->getTensorNode(i)->getTensor();
        std::string name = tensors_name_map_[tensor];
        writer_ << dtype << " *" << name << ";\n";
    }
    */

    for (auto it : tensors_name_map_) {
        auto *tensor = it.first;
        std::string dtype = getTypeString(tensor);

        writer_ << dtype << " *" << it.second << ";\n";
    }

    writer_ << "\n";

    SWLOG_DEBUG(4) << "end emitVarDeclarations...\n";
}

/// to implement different level mem alloca
/// may not be a good idea.
/// TODO: deal with node-cpu-device abstraction
void Codegen::emitMemAllocations() {
    SWLOG_DEBUG(4) << "begin emitMemAllocations...\n";
    // std::string dtype = this->dtype();
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (size == 0)
            continue;

        emitMemAllocation(base, size, dev);

    }

    if(p_mem_alllocator_->getMemAllocated()) {

        auto dev = p_mem_alllocator_->getDevice();
        std::string base = p_mem_alllocator_->getBasePtrName();
        uint64_t size = p_mem_alllocator_->getMemAllocated();

        emitMemAllocation(base, size, dev);
    }
    writer_ << "\n";
    SWLOG_DEBUG(4) << "end emitMemAllocations...\n";
}
void Codegen::emitMemAllocation(std::string buffer, size_t bytes, Device& dev) {
    switch (dev.type) {
    case DeviceType::CPU:
        // writer_ << base << " = (" << dtype << "*)malloc(" << size <<
        // ");\n";
        writer_ << buffer << " = (char*)malloc(" << bytes << ");\n";
        break;
    case DeviceType::GPU:
        writer_ << "\n";
        writer_ << "cudaSetDevice(" << dev.id << ");\n";
        writer_ << "cudaMalloc(&" << buffer << ", " << bytes << ");\n";
        break;
    default:
        SWLOG_ERROR << "Unknown DeviceType\n";
        break;
    }

}

void Codegen::emitTensorAddresses() {
    SWLOG_DEBUG(4) << "begin emitTensorAddresses...\n";

    std::set<Tensor *> visited_tensors;

    emitTensorAddresses(graph_, &visited_tensors);

    for (int i = 0; i < graph_->opNodeNum(); i++) {
        OpNode *opnode = graph_->getOpNode(i);
        if (auto graphOp = dynamic_cast<SubGraphOp *>(opnode->getOp())) {
            if (auto ngraph = graphOp->getGraph()) {
                switchTo(ngraph);
                emitTensorAddresses(ngraph, &visited_tensors);
                switchFrom(ngraph);
                writer_ << "\n";
            }
        }
    }

    SWLOG_DEBUG(4) << "end emitTensorAddresses...\n";
}

void Codegen::emitTensorAddresses(IRGraph *graph,
                                  std::set<Tensor *> *visited_tensors) {
    for (int i = 0; i < graph->tensorNodeNum(); i++) {
        auto *tnode = graph->getTensorNode(i);
        auto *tensor = tnode->getTensor();

        if (visited_tensors->count(tensor))
            continue;
        visited_tensors->insert(tensor);

        std::string dtype = getTypeString(tensor);

        std::string name = tensors_name_map_[tensor];
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];
        writer_ << name << " = reinterpret_cast<" << dtype << "*>(" << base
                << " + " << offset << ");\n";
    } // tensor loop

    writer_ << "\n";
}

/// if flag_MPI this func deal with
/// the MASTER(0) process
void Codegen::emitTensorInitializations() {
    SWLOG_DEBUG(4) << "begin emitTensorInitializations...\n";

    std::set<Tensor *> visited_tensors;


    if (config_.train_mode) {
        emitTensorInitFromSnapshot(graph_, &visited_tensors);
    } else {
        emitTensorInitializations(graph_, &visited_tensors);
    }

    for (int i = 0; i < graph_->opNodeNum(); i++) {
        OpNode *opnode = graph_->getOpNode(i);
        if (auto graphOp = dynamic_cast<SubGraphOp *>(opnode->getOp())) {
            if (auto ngraph = graphOp->getGraph()) {
                // switchTo(ngraph);
                emitTensorInitializations(ngraph, &visited_tensors);
                // switchFrom(ngraph);
                writer_ << "\n";
            }
        }
    }

    SWLOG_DEBUG(4) << "end emitTensorInitializations...\n";
}

void Codegen::emitTensorInitializations(IRGraph *graph_,
                                        std::set<Tensor *> *visited_tensors) {
    for (int i = 0; i < graph_->tensorNodeNum(); i++) {
        auto *tnode = graph_->getTensorNode(i);
        auto *tensor = tnode->getTensor();

        if (visited_tensors->count(tensor))
            continue;
        visited_tensors->insert(tensor);

        std::string dtype = getTypeString(tensor);

        std::string name = tensors_name_map_[tensor];
        uint64_t size = tensor->size();
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];

        TensorInitInfo info = tensor->getTensorInitInfo();
        switch (tensor->getTensorInitType()) {
        case TensorInitType::NONE:
            break;
        case TensorInitType::XAVIER: {
            // TODO
            writer_ << "initTensorXavier(" << name << ", " << size << ", "
                    << info.getFilterSize() << ");\n";
            break;
        }
        case TensorInitType::CONSTANT: {
            writer_ << "initTensorConstant(" << name << ", " << size << ", "
                    << info.getConstant() << ");\n";
            break;
        }
        case TensorInitType::ZERO: {
            writer_ << "initTensorZero(" << name << ", " << size << ");\n";
            break;
        }
        case TensorInitType::FILE: {
            writer_ << "load(" << name << ", " << size << ", "
                    << info.getOffset() << ", "
                    << "\"" << info.getFilePath() << "\");\n";
            break;
        }
        case TensorInitType::PARENTOP: {
            auto *op = (OpNode *)tnode->getParentNode(0);
            dispatchOpNode(op);
            break;
        }
        default:
            SWLOG_DEBUG(1) << name << " TensorInitType= NONE\n";
            break;

        } // switch
    }     // tensor loop

    writer_ << "\n";
}

void Codegen::emitTensorInitFromSnapshot(IRGraph *graph_,
                                         std::set<Tensor *> *visited_tensors) {
    writer_ << "if(FLAGS_snapshot.size() > 0) {\n";
    writer_.indentInc();
    writer_ << "std::cout << \"restoring from snapshot \"  << FLAGS_snapshot "
               "<< std::endl;\n";
    writer_ << "caffe2::NetDef net;"
            << "\n";
    writer_
        << "std::ifstream ff(FLAGS_snapshot, std::ios::in | std::ios::binary);"
        << "\n";

    writer_ << "google::protobuf::io::IstreamInputStream filestr(&ff);"
            << "\n";
    writer_ << "google::protobuf::io::CodedInputStream codedstr(&filestr);"
            << "\n";
    writer_ << "codedstr.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);"
            << "\n";
    writer_ << "bool parseNet = net.ParseFromCodedStream(&codedstr);"
            << "\n";

    for (int i = 0; i < graph_->tensorNodeNum(); i++) {
        auto *tnode = graph_->getTensorNode(i);
        auto *tensor = tnode->getTensor();

        if (visited_tensors->count(tensor))
            continue;
        visited_tensors->insert(tensor);

        std::string dtype = getTypeString(tensor);

        std::string name = tensors_name_map_[tensor];
        uint64_t size = tensor->size();
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];

        TensorInitInfo info = tensor->getTensorInitInfo();

        switch (tensor->getTensorInitType()) {
        case TensorInitType::XAVIER:
        case TensorInitType::CONSTANT:
        case TensorInitType::ZERO:
            writer_ << "loadFromSnapShot(net, \"" << name << "\", " << name
                    << ", " << size << ");"
                    << "\n";
            break;
        case TensorInitType::NONE:
            break;
        case TensorInitType::FILE: {
            writer_ << "load(" << name << ", " << size << ", "
                    << info.getOffset() << ", "
                    << "\"" << info.getFilePath() << "\");\n";
            break;
        }
        case TensorInitType::PARENTOP: {
            auto *op = (OpNode *)tnode->getParentNode(0);
            dispatchOpNode(op);
            break;
        }
        default:
            SWLOG_DEBUG(1) << name << " TensorInitType= NONE\n";
            break;

        } // switch
    }     // tensor loop

    writer_ << "int snap_iter = getIterFromSnapShot(net);"
            << "\n";
    writer_ << "loader.shift(snap_iter);"
            << "\n";
    writer_ << "iter = snap_iter;"
            << "\n";
    writer_ << "std::cout << \"snapshot iter= \" << snap_iter << std::endl;"
            << "\n";
    writer_.indentDec();
    writer_ << "} else { "
            << "\n";
    writer_.indentInc();
    visited_tensors->clear();
    emitTensorInitializations(graph_, visited_tensors);
    writer_.indentDec();
    writer_ << "}"
            << "\n";
}

void Codegen::emitSaveSnapshot() {
    writer_ << "\n";
    writer_ << "if(iter % " << config_.train_config.snapshot << " == 0) {\n";
    writer_.indentInc();
    writer_ << R"(std::cout << "iter = " << iter << ", snapshot\n";)"
            << "\n"
            << "caffe2::NetDef net;\n";
    writer_ << "caffe2::Argument *iter_arg = net.add_arg();\n"
            << "iter_arg->set_name(\"iter\");\n"
            << "iter_arg->set_i(iter);"
            << "\n";

    std::unordered_set<Tensor *> visited_tensors;

    for (int i = 0; i < graph_->tensorNodeNum(); i++) {
        auto *tnode = graph_->getTensorNode(i);
        auto *tensor = tnode->getTensor();

        if(visited_tensors.count(tensor)) {
            continue;
        } else {
            visited_tensors.insert(tensor);
        }

        std::string dtype = getTypeString(tensor);

        std::string name = tensors_name_map_[tensor];
        uint64_t size = tensor->size();
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];

        TensorInitInfo info = tensor->getTensorInitInfo();

        switch (tensor->getTensorInitType()) {
        case TensorInitType::XAVIER:
        case TensorInitType::CONSTANT:
        case TensorInitType::ZERO:
            writer_ << "addOp(net, \"" << name << "\", " << name << ", " << size
                    << ");"
                    << "\n";
            SWLOG_DEBUG(8) << "[emitSaveSnapshot] " << name << " TensorInitType="
                    << static_cast<int>(tensor->getTensorInitType()) << "\n";
            break;
        case TensorInitType::NONE:
        case TensorInitType::FILE:
        case TensorInitType::PARENTOP:
            break;
        default:
            break;
        } // switch
    }     // tensor loop

    writer_
        << R"(std::string snapfile = "snapshot_iter"+std::to_string(iter)+".model";)"
        << "\n";
    writer_ << "std::ofstream ff(snapfile, "
               "std::ios::out|std::ios::trunc|std::ios::binary);"
            << "\n";
    writer_ << "net.SerializeToOstream(&ff);"
            << "\n";
    writer_ << "ff.close();"
            << "\n";
    writer_ << R"(std::cout << "save snapshot to " << snapfile << "\n";)"
            << "\n";
    writer_.indentDec();
    writer_ << "}"
            << "\n";
}

void Codegen::emitPrintGraphOutputs() {
    writer_ << "\n";
    if(config_.train_mode)
        writer_ << "if(iter % " << config_.train_config.display<< " == 0) {\n";
    else if(config_.use_dataloader)
        writer_ << "if(iter % " << config_.display<< " == 0) {\n";
    writer_.indentInc();

    writer_ << R"(std::cout << "iterations " << iter << "\n";)"
            << "\n";
    /*
    for(int i=0; i<graph_->outNodeNum(); i++) {
        TensorNode * outnode = graph_->getOutNode(i);
        Tensor* out = outnode->getTensor();
        std::cout << out << "\n";
        int m = out->getDim(0);
        int n = out->getNDim()==2 ? out->getDim(1) : 1;
        writer_ << "// OutNode " << i << ": " << outnode->name() << "\n";
        writer_ << "std::cout << \"" << outnode->name() <<":\\n\";" << "\n";
        writer_ << "printMatrix(" <<  tensors_name_map_[out] << ", "
            << m << ", " << n << ");\n";
    }
    */
    for(auto &tnode : graph_->getDisplayTensorNodes()) {
        auto* out = tnode->getTensor();
        int m = out->getDim(0);
        int n = out->getNDim()==2 ? out->getDim(1) : 1;
        writer_ << "// " << tnode->name() << "\n";
        writer_ << "std::cout << \"" << tnode->name() <<":\\n\";" << "\n";
        writer_ << "printMatrix(" <<  tensors_name_map_[out] << ", "
            << m << ", " << n << ");\n";
    }

    writer_.indentDec();
    writer_ << "}"
            << "\n";
}

std::string Codegen::emitTensorMemAlloc(TensorNode *tnode) {
    std::string bufferName =
        tnode->name(); // ensure IRNode name unique before Codegen
    int dims = tnode->getTensor()->getNDim();

    size_t size = 1;
    for (int dim = 0; dim < dims; dim++)
        size *= tnode->getTensor()->getDim(dim);

    DataType dtype = tnode->getDataType();

    switch (dtype) {
    case DataType::Float_t:
        writer_ << "float *" << bufferName
                << " = (float *)malloc(sizeof(float) * " << size << ");\n";
        break;
    case DataType::Double_t:
        writer_ << "double *" << bufferName
                << " = (double *)malloc(sizeof(double) * " << size << ");\n";
        break;
    default:
        SWLOG_ERROR << "UNKNOWN DataType\n";
    }

    tensors_name_map_[tnode->getTensor()] = bufferName;
    return bufferName;
}

std::string Codegen::getBytesProtoString(BytesProto proto) {
    switch (proto) {
    case ONE_BYTE_AS_INT:
        return "ONE_BYTE_AS_INT";
    case FOUR_BYTES_AS_FLOAT:
        return "FOUR_BYTES_AS_FLOAT";
    default:
        return "ONE_BYTE_AS_INT";
    }
}

std::string Codegen::getInitialLizerString(const std::vector<size_t> &dims) {
    std::ostringstream os;
    os << "{";
    for (auto dim : dims)
        os << dim << ", ";

    std::string str = os.str();
    return str.substr(0, str.length() - 2) + "}";
}

void Codegen::emitDataLoaderInit() {
    TensorNode *label = graph_->getTrainLabelNode();
    TensorNode *data = graph_->getTrainDataNode();
    assert((label && data) && "Train label or data null");
    // DataLoader loader(filename, BytesProto::ONE_BYTE_AS_INT,
    // BytesProto::FOUR_BYTES_AS_FLOAT, 1, 60000, {8u}, {8u, 28u, 28u, 1u});
    //
    std::string var_file = UniqueName("dataloader_train_source");
    writer_ << "std::string " << var_file << " = \""
            << config_.train_config.train_data_file << "\";\n";
    writer_ << "DataLoader loader(";
    // writer_ << "\"" << config_.train_config.train_data_file << "\", ";
    writer_ << var_file << ", ";
    writer_ << getBytesProtoString(config_.train_config.label_bytes) << ", ";
    writer_ << getBytesProtoString(config_.train_config.data_bytes) << ", ";
    writer_ << config_.train_config.max_epoch << ", ";
    writer_ << config_.train_config.train_data_samples << ", ";
    writer_ << getInitialLizerString(label->getDims()) << ", ";
    writer_ << getInitialLizerString(data->getDims());
    writer_ << ");\n";

    writer_ << "size_t iter = 0; "
            << "\n\n";
}

void Codegen::emitInferDataLoaderInit() {
    TensorNode *label = graph_->getInferLabelNode();
    TensorNode *data = graph_->getInferDataNode();
    assert((label && data) && "Test label or data null");

    std::string var_file = UniqueName("dataloader_infer_source");
    writer_ << "std::string " << var_file << " = \""
            << config_.dataloader_src<< "\";\n";
    writer_ << "DataLoader loader(";
    writer_ << var_file << ", ";
    writer_ << getBytesProtoString(config_.label_bytes) << ", ";
    writer_ << getBytesProtoString(config_.data_bytes) << ", ";
    // writer_ << config_.train_config.max_epoch << ", ";
    writer_ << "1, ";
    writer_ << config_.dataloader_samples << ", ";
    writer_ << getInitialLizerString(label->getDims()) << ", ";
    writer_ << getInitialLizerString(data->getDims());
    writer_ << ");\n";

    writer_ << "size_t iter = 0; "
            << "\n\n";
}

void Codegen::emitExecute() {
    SWLOG_DEBUG(4) << "begin emitExecute ...\n";
    if (config_.train_mode) {
        TensorNode *label = graph_->getTrainLabelNode();
        TensorNode *data = graph_->getTrainDataNode();
        if(!tensors_name_map_.count(label->getTensor())) {
            SWLOG_DEBUG(4) << "label tensor " << label->name() << " " << label->getTensor() << " not in map ...\n";
            for(auto iter : tensors_name_map_) {
                std::cout << iter.first << " " << iter.second << "\n";
            }
            exit(0);
        }
        if(!tensors_name_map_.count(data->getTensor())) {
            SWLOG_DEBUG(4) << "data tensor " << data->name() << " " << data->getTensor() << " not in map ...\n";
            exit(0);
        }
        std::string label_var = tensors_name_map_.at(label->getTensor());
        std::string data_var = tensors_name_map_.at(data->getTensor());
        /*
        std::string label_var = label->name();
        std::string data_var = data->name();
        */
        writer_ << "while(loader.next(" << label_var << ", " << data_var
                << ")) {\n";

        writer_.indentInc();
    } else if (config_.use_dataloader) {
        TensorNode *label = graph_->getInferLabelNode();
        TensorNode *data = graph_->getInferDataNode();

        assert(tensors_name_map_.count(label->getTensor()) && "label not allocated");
        assert(tensors_name_map_.count(data->getTensor()) && "data not allocated");

        std::string label_var = tensors_name_map_.at(label->getTensor());
        std::string data_var = tensors_name_map_.at(data->getTensor());

        writer_ << "while(loader.next(" << label_var << ", " << data_var
                << ")) {\n";

        writer_.indentInc();
    }

    emitFuncCalls();

    if (config_.train_mode) {

        writer_ << "\n";
        writer_ << "iter++;\n";
        if (config_.train_config.snapshot) {
            emitSaveSnapshot();
        }

        if(config_.train_config.display) {
            emitPrintGraphOutputs();
        }

        writer_.indentDec();

        writer_ << "} //while\n";
    }else if(config_.use_dataloader) {
        writer_ << "\n";
        writer_ << "iter++;\n";

        if(config_.display > 0) {
            emitPrintGraphOutputs();
        }

        writer_.indentDec();
        writer_ << "} //while\n";

    }
    SWLOG_DEBUG(4) << "begin emitExecute ...\n";
}

void Codegen::emitFuncCalls() {
    SWLOG_DEBUG(4) << "begin emitFuncCalls ...\n";
    for (int i = 0; i < graph_->topologyNum(); i++)
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto node = graph_->getNodeInTopo(i, j);
            if (node->nodeType() == OP_NODE) {
                auto opnode = (OpNode *)node;
                writer_ << "\n";
                writer_ << "// topology(" << i << ", " << j
                        << "): " << opnode->name() << " : "
                        << opnode->getOpName() << "\n";
                if (auto graphOp =
                        dynamic_cast<SubGraphOp *>(opnode->getOp())) {
                    if (auto ngraph = graphOp->getGraph()) {
                        // switchTo(ngraph);
                        emitFuncCalls(ngraph);
                        // switchFrom(ngraph);
                        writer_ << "\n";
                    }
                } else {
                    dispatchOpNode(opnode);
                }
            }
        }
    SWLOG_DEBUG(4) << "end emitFuncCalls ...\n";
}
void Codegen::emitFuncCalls(IRGraph *graph_) {
    for (int i = 0; i < graph_->topologyNum(); i++)
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto node = graph_->getNodeInTopo(i, j);
            if (node->nodeType() == OP_NODE) {
                writer_ << "// topology(" << i << ", " << j
                        << "): " << node->name() << "\n";
                auto opnode = (OpNode *)node;
                if (auto graphOp =
                        dynamic_cast<SubGraphOp *>(opnode->getOp())) {
                        (void)graphOp;
                } else {
                    dispatchOpNode(opnode);
                }
            }
        }
}

void Codegen::switchTo(IRGraph *ngraph) {
    Device dev = ngraph->getDeviceLabel();
    if (dev.type == DeviceType::CPU) {
        // TODO MPI

    } else if (dev.type == DeviceType::GPU) {
        writer_ << "cudaSetDevice(" << dev.id << ");\n";
    }
}

void Codegen::switchFrom(IRGraph *ngraph) { (void)ngraph; }

void Codegen::dispatchOpNode(OpNode *op) {
    if (!op->runable())
        return;

    Label *label = op->getLabel();
    Device dev = label->getDeviceLabel();
    if (auto scatter = dynamic_cast<ScatterOp *>(op->getOp())) {
        auto *from = ((TensorNode *)op->getParentNode(0));
        auto *from_tensor = from->getTensor();
        Device from_dev = from->getLabel()->getDeviceLabel();
        auto *to = ((TensorNode *)op->getChildNode(0));
        auto *to_tensor = to->getTensor();

        size_t offset = scatter->getOffset();
        size_t size = to_tensor->getSizeInBytes();

        emitMemcpyFromTo(from_tensor, from_dev, offset, size, to_tensor, dev,
                         0);
    } else if (auto gather = dynamic_cast<GatherOp *>(op->getOp())) {
        auto *from = ((TensorNode *)op->getParentNode(0));
        auto *from_tensor = from->getTensor();
        auto *to = ((TensorNode *)op->getChildNode(0));
        auto *to_tensor = to->getTensor();
        Device to_dev = to->getLabel()->getDeviceLabel();

        size_t offset = gather->getOffset();
        size_t size = from_tensor->getSizeInBytes();

        emitMemcpyFromTo(from_tensor, dev, 0, size, to_tensor, to_dev, offset);
    } else {
        switch (dev.type) {
        case DeviceType::CPU:
            emitFuncCall(op);
            break;
        case DeviceType::GPU:
            emitFuncCallCUDA(op);
            break;
        default:
            SWLOG_ERROR << "unknown device type in dispatchOpNode\n";
        }
    }
}

void Codegen::emitMemcpyFromTo(Tensor *from, Device from_dev,
                               size_t from_offset, size_t size, Tensor *to,
                               Device to_dev, size_t to_offset) {
    std::string fname = tensors_name_map_[from];
    std::string tname = tensors_name_map_[to];

    writer_ << "// Memcpy from " << fname << " to " << tname << "\n";

    if (from_dev.type == DeviceType::CPU && to_dev.type == DeviceType::GPU) {
        if (config_.cuda_stream) {
            writer_ << "cudaMemcpyAsync(" << tname << "+" << to_offset << ", "
                    << fname << "+" << from_offset << ", " << size << ", "
                    << "cudaMemcpyHostToDevice, stream[" << to_dev.id
                    << "]);\n";
        } else {
            writer_ << "cudaMemcpy(" << tname << "+" << to_offset << ", "
                    << fname << "+" << from_offset << ", " << size << ", "
                    << "cudaMemcpyHostToDevice);\n";
        }
    }

    if (from_dev.type == DeviceType::GPU && to_dev.type == DeviceType::CPU) {
        if (config_.cuda_stream) {
            writer_ << "cudaMemcpyAsync(" << tname << "+" << to_offset << ", "
                    << fname << "+" << from_offset << ", " << size << ", "
                    << "cudaMemcpyDeviceToHost, stream[" << from_dev.id
                    << "]);\n";

        } else {
            writer_ << "cudaMemcpy(" << tname << "+" << to_offset << ", "
                    << fname << "+" << from_offset << ", " << size << ", "
                    << ", " << size << ", "
                    << "cudaMemcpyDeviceToHost);\n";
        }
    }
}

void Codegen::emitFuncCall(OpNode *op) {
    if(config_.compute_op_annotation) {
        writer_ << "/*\n";
    }
    
    DataType dtype =
        op->parentNum() > 0
            ? ((TensorNode *)op->getParentNode(0))->getTensor()->getDataType()
            : DataType::Float_t;

    std::string dtype_flag;
    switch (dtype) {
    case DataType::Float_t:
        dtype_flag = "f";
        break;
    case DataType::Double_t:
        dtype_flag = "d";
        break;
    case DataType::Int8_t:
        dtype_flag = "i8";
        break;
    case DataType::Int32_t:
        dtype_flag = "i";
        break;
    default:
        dtype_flag = "f";
        SWLOG_ERROR << "UNKNOWN DataType " << op->name() << "'s parent "
            << op->getParentNode(0)->name() << " "<< static_cast<int>(dtype) << "\n";
    }
    auto name = op->name();
    Label *oplabel = op->getLabel();

    std::string mkldnn_engine = "mkldnn_eng";
    std::string mkldnn_stream = "mkldnn_s";

    SWLOG_DEBUG(2) << "begin genKernelCall for " << name << "\n";

    // TODO assert legal dimensions
    if ((oplabel->getTypeNameLabel()).compare("MatrixMatrixMul") == 0) {
        // TODO assert
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int k = B->getDim(0);
        int n = B->getDim(1);

        writer_ << "matrixMatrixMul_" << dtype_flag << "(" << m << ", " << n
                << ", " << k << ", " << tensors_name_map_[A] << ", " << k
                << ", " << tensors_name_map_[B] << ", " << n << ", "
                << tensors_name_map_[C] << ", " << n << ");\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("MatrixMatrixFCBias") == 0) {
        auto *in = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *w = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *b = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        // assert(in->getNDim()==2 && "FC does not support src dim=4 currently\n");


        if(config_.mkldnn) {
            /* for mkldnn
             * 1. w already tranposed to oCiC in Engine::transformForMKLDNN()
             * 2. in tensor may be nchw(caffe2, without reshape) | nhwc (swc) | nc
             *
             * */
            auto in_dims = name+"_in_dims";
            auto w_dims = name+"_w_dims";
            auto b_dims = name+"_b_dims";
            auto out_dims = name+"_out_dims";
            auto strides_dims = name + "_strides_dims"; 
            auto pads_dims = name + "_pads_dims"; 

            std::vector<size_t> idims;
            if(in->getNDim() == 4) {
                // in may be nhwc(swc defined), or nchw(caffe2 imported)
                auto dim2 = in->viewAs2D(1);    
                idims = {dim2.first, dim2.second};
            }else if(in->getNDim() ==2) {
                idims = in->getDims();
            }

            emit_mkldnn_memory_dims(in_dims, idims); 
            emit_mkldnn_memory_dims(w_dims, w->getDims()); 
            emit_mkldnn_memory_dims(b_dims, b->getDims()); 
            emit_mkldnn_memory_dims(out_dims, out->getDims()); 
            
            auto in_md = name+"_in_md";
            auto w_md = name+"_w_md";
            auto b_md = name+"_b_md";
            auto out_md = name+"_out_md";
            emit_mkldnn_memory_desc(in_md, in_dims, in, "any"); // true implies format_tag::any 
            emit_mkldnn_memory_desc(w_md, w_dims, w, "any"); 
            emit_mkldnn_memory_desc(b_md, b_dims, b, "any"); 
            emit_mkldnn_memory_desc(out_md, out_dims, out, "any"); 
            writer_ << "\n";

            auto fc_desc = name+"_desc";
            writer_ << "auto " << fc_desc << " = inner_product_forward::desc(prop_kind::forward_inference, "
                << in_md << ", "
                << w_md << ", "
                << b_md << ", "
                << out_md << ");\n";


            // inner_product primitive descriptor 
            auto fc_pd = name + "_pd";
            writer_ << "auto " << fc_pd << " = inner_product_forward::primitive_desc("
                << fc_desc << ", " << mkldnn_engine << ");\n";
            writer_ << "\n";

            // memory objects
            auto in_mem = name+"_in_mem";
            auto w_mem = name+"_w_mem";
            auto b_mem = name+"_b_mem";
            auto out_mem = name+"_out_mem";
            emit_mkldnn_memory(in_mem, in, in_dims, 
                mkldnn_engine, tensors_name_map_[in], "nc"); 
            emit_mkldnn_memory(w_mem, w, w_dims, 
                mkldnn_engine, tensors_name_map_[w], "nc"); // nc 
            emit_mkldnn_memory(b_mem, b, b_dims,
                mkldnn_engine, tensors_name_map_[b]); 
            emit_mkldnn_memory(out_mem, out, out_dims,
                mkldnn_engine, tensors_name_map_[out]); 
            writer_ << "\n";

            // actual src, weight, dst of mkl inner_product memory
            #define EMIT_NEED_REORDER(var, pd, obj, mem)    \
                writer_ << "auto " << var << " = "         \
                    << pd << "." << #obj << "_desc() != "  \
                    << mem << ".get_desc();\n";
            auto need_reorder_in = "need_reorder_" + name + "_in";
            auto need_reorder_w = "need_reorder_" + name + "_w";
            auto need_reorder_out = "need_reorder_" + name + "_out";
            EMIT_NEED_REORDER(need_reorder_in, fc_pd, src, in_mem);
            EMIT_NEED_REORDER(need_reorder_w, fc_pd, weights, w_mem);
            EMIT_NEED_REORDER(need_reorder_out, fc_pd, dst, out_mem);
            writer_ << "\n";


            #define CREATE_CONV_MEM(cmem, flag, pd, obj, mem)                       \
                writer_ << "auto " << cmem << " = " << flag << " ? "                \
                    << "memory(" << pd << "." << #obj << "_desc(), mkldnn_eng)"  \
                    << " : " << mem << ";\n";
            auto src_mem = name+"_src_mem";
            auto weight_mem = name+"_weights_mem";
            auto dst_mem = name+"_dst_mem";
            CREATE_CONV_MEM(src_mem, need_reorder_in, fc_pd, src, in_mem);
            CREATE_CONV_MEM(weight_mem, need_reorder_w, fc_pd, weights, w_mem);
            CREATE_CONV_MEM(dst_mem, need_reorder_out, fc_pd, dst, out_mem);
            writer_ << "\n";


            #define EMIT_REORDER(cmd, flag, from, to)                   \
                writer_ << "if(" << flag<< ") {\n"                      \
                    << "\t" << "auto "<< cmd << " = reorder("           \
                    << from << ", " << to << ");\n";                    \
                writer_ << "\t" << cmd << ".execute(mkldnn_s, {"        \
                    << "{MKLDNN_ARG_FROM, " << from << "},"             \
                    << "{MKLDNN_ARG_TO, " << to << "}});\n";           \
                writer_ << "\t" << "mkldnn_s.wait();\n";                \
                writer_ << "}\n";                                       \

            auto cmd_reorder_in = "reorder_" + name + "_in";
            auto cmd_reorder_w = "reorder_" + name + "_w";
            auto cmd_reorder_out = "reorder_" + name + "_out";
            EMIT_REORDER(cmd_reorder_in, need_reorder_in, in_mem, src_mem);
            EMIT_REORDER(cmd_reorder_w, need_reorder_w, w_mem, weight_mem);
            writer_ << "\n";

             
            // inner_product computation
            auto cmd_fc = name + "_forward";
            writer_ << "auto " << cmd_fc << " = "
                << "inner_product_forward(" << fc_pd << ");\n";
            writer_ << cmd_fc << ".execute(mkldnn_s, {\n";
            writer_.indentInc();
            writer_ << "{ MKLDNN_ARG_SRC," << src_mem << "},\n"
                << "{ MKLDNN_ARG_WEIGHTS," << weight_mem <<  "},\n"
                << "{ MKLDNN_ARG_BIAS,"<< b_mem <<  "},\n"
                << "{ MKLDNN_ARG_DST," << dst_mem << "} });\n";
            writer_.indentDec();
            writer_ << "mkldnn_s.wait();\n";
            
            EMIT_REORDER(cmd_reorder_out, need_reorder_out, dst_mem, out_mem);

            return;
        }

    }

    if ((oplabel->getTypeNameLabel()).compare("Reshape") == 0) {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();

        writer_ << tensors_name_map_[B] << " = " << tensors_name_map_[A] << ";\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("TensorAscend") == 0) {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();

        writer_ << tensors_name_map_[B] << " = " << tensors_name_map_[A] << ";\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("TensorDescend") == 0) {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();

        writer_ << tensors_name_map_[B] << " = " << tensors_name_map_[A] << ";\n";
    }

    if ((oplabel->getTypeNameLabel()) == "BatchedAdd"  || (oplabel->getTypeNameLabel()) == "MatrixVectorAdd") {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();

        size_t sliceNum, sliceSize;
        std::tie(sliceNum, sliceSize) = convertToDim2(A->getDims());
        auto bdim = B->size();
        (void)bdim;
        assert((sliceSize == bdim) &&
               "batch flattened dim.second != bias dim!");

        writer_ << "batchedadd_" << dtype_flag << "(" << tensors_name_map_[C]
                << ", " << tensors_name_map_[A] << ", " << tensors_name_map_[B]
                << ", " << sliceNum << ", " << sliceSize << ");\n";
    }
    if ((oplabel->getTypeNameLabel()) == "BatchedReduceAdd") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *output = ((TensorNode *)op->getChildNode(0))->getTensor();

        size_t sliceNum, sliceSize;
        std::tie(sliceNum, sliceSize) = convertToDim2(input->getDims());
        auto bdim = output->size();
        (void)bdim;
        assert((sliceSize == bdim) &&
               "batch flattened dim.second != bias dim!");

        writer_ << "batchedreduceadd_" << dtype_flag << "("
                << tensors_name_map_[output] << ", " << tensors_name_map_[input]
                << ", " << sliceNum << ", " << sliceSize << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "ElementAdd") {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();

        if(config_.mkldnn) {
            // mkldnn support more than 2 source operands with MKLDNN_ARG_MULTIPLE_SRC+i
            auto out_dims = name + "_out_dims";
            emit_mkldnn_memory_dims(out_dims, C->getDims()); 

            auto src0_md = name+"_src0_md";
            auto src1_md = name+"_src1_md";
            auto out_md = name+"_out_md";
            auto srcs_vec = name+"_srcs_vec";
            auto scales_vec = name+"_scales_vec";
            auto srcs_md = name+"_srcs_md";

            emit_mkldnn_memory_desc(src0_md, out_dims, A);
            emit_mkldnn_memory_desc(src1_md, out_dims, B);
            std::string dtype = getTypeString(C);
            std::string layout = C->getMemLayoutTag();
            writer_ << "std::vector<memory::desc> " << srcs_vec << " = {"
                << src0_md << ", " << src1_md << "};\n"; 
            writer_ << "std::vector<" << dtype << "> " << scales_vec << " = {"
                << "1.0f" << ", " << "1.0f" << "};\n"; 
            writer_ << "\n";


            /*
            writer_ << "auto " << srcs_md << " = memory::desc({{" << srcs_vec << "}, " 
                << dtype_mkldnn_datatype_map.at(dtype) << ", " 
                << layout_mkldnn_format_tag_map.at(layout)
                << "});\n";
            writer_ << "\n";
            */

            emit_mkldnn_memory_desc(out_md, out_dims, C); 
            writer_ << "\n";


            // BatchNormalizatio primitive descriptor 
            auto sum_pd = name + "_pd";
            writer_ << "auto " << sum_pd << " = sum::primitive_desc("
                << out_md << ", " 
                << scales_vec << ", " << srcs_vec << ", " 
                << mkldnn_engine << ");\n";
            writer_ << "\n";

            // memory objects: in(may skeep if exists)
            auto src0_mem = name+"_src0_mem";
            auto src1_mem = name+"_src1_mem";
            auto out_mem = name+"_out_mem";
            emit_mkldnn_memory(src0_mem, A, out_dims, mkldnn_engine, tensors_name_map_[A]); 
            emit_mkldnn_memory(src1_mem, B, out_dims, mkldnn_engine, tensors_name_map_[B]); 
            emit_mkldnn_memory(out_mem, C, out_dims,mkldnn_engine, tensors_name_map_[C]); 


            // sum computation
            auto cmd_sum = name + "_cmd";
            writer_ << "auto " << cmd_sum << " = "
                << "sum(" << sum_pd << ");\n";
            writer_ << cmd_sum << ".execute(mkldnn_s, {\n";
            writer_.indentInc();
            writer_ << "{ MKLDNN_ARG_MULTIPLE_SRC+0," << src0_mem << "},\n"
                << "{ MKLDNN_ARG_MULTIPLE_SRC+1," << src1_mem << "},\n"
                << "{ MKLDNN_ARG_DST," << out_mem << "} });\n";
            writer_.indentDec();
            writer_ << "mkldnn_s.wait();\n";
            writer_ << "\n";

            return;
        }

        auto num = A->size();

        writer_ << "vecAdd_" << dtype_flag << "(" << num << ", "
                << tensors_name_map_[A] << ", " << tensors_name_map_[B] << ", "
                << tensors_name_map_[C] << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "Conv2d") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *filter = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *bias = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *conv_op = (Conv2dOp *)op->getOp();
        auto kernels = conv_op->getKernels();
        auto strides = conv_op->getStrides();
        auto pads = conv_op->getPads();
        auto group = conv_op->getGroup();

        if(config_.mkldnn) {
            // if conv in is nhwc
            auto idims = (input->getMemLayoutTag()=="nhwc") ?
                input->getShuffledDims(NHWC2NCHW/*{0,3,1,2}*/) : input->getDims();
            
            auto in_dims = name+"_in_dims";
            auto w_dims = name+"_w_dims";
            auto b_dims = name+"_b_dims";
            auto out_dims = name+"_out_dims";
            auto strides_dims = name + "_strides_dims"; 
            auto pads_dims = name + "_pads_dims"; 
            emit_mkldnn_memory_dims(in_dims, idims); 
            emit_mkldnn_memory_dims(w_dims, filter->getDims()); 
            emit_mkldnn_memory_dims(b_dims, bias->getDims()); 
            emit_mkldnn_memory_dims(out_dims, out->getDims()); 
            emit_mkldnn_memory_dims(strides_dims, strides); 
            emit_mkldnn_memory_dims(pads_dims, pads); 

            auto in_md = name+"_in_md";
            auto w_md = name+"_w_md";
            auto b_md = name+"_b_md";
            auto out_md = name+"_out_md";
            emit_mkldnn_memory_desc(in_md, in_dims, input, "any"); // true implies format_tag::any 
            emit_mkldnn_memory_desc(w_md, w_dims, filter, "any"); 
            emit_mkldnn_memory_desc(b_md, b_dims, bias); 
            emit_mkldnn_memory_desc(out_md, out_dims, out, "any"); 
            writer_ << "\n";


            // convolution descriptor
            // typo
            /*
            mkldnn_convolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &padding_l[0],
                        &padding_r[0])
            */
            auto conv_desc = name+"_desc";
            writer_ << "auto " << conv_desc << " = convolution_forward::desc(prop_kind::forward_inference, "
                << "algorithm::convolution_direct, "
                << in_md << ", "
                << w_md << ", "
                << b_md << ", "
                << out_md << ", "
                << strides_dims <<", "
                << pads_dims << ", "
                << pads_dims << ");\n";


            // convolution primitive descriptor 
            auto conv_pd = name + "_pd";
            writer_ << "auto " << conv_pd << " = convolution_forward::primitive_desc("
                << conv_desc << ", " << mkldnn_engine << ");\n";
            writer_ << "\n";
            

            // memory objects
            auto in_mem = name+"_in_mem";
            auto w_mem = name+"_w_mem";
            auto b_mem = name+"_b_mem";
            auto out_mem = name+"_out_mem";
            emit_mkldnn_memory(in_mem, input, in_dims, 
                mkldnn_engine, tensors_name_map_[input]); 
            emit_mkldnn_memory(w_mem, filter, w_dims, 
                mkldnn_engine, tensors_name_map_[filter]); 
            emit_mkldnn_memory(b_mem, bias, b_dims,
                mkldnn_engine, tensors_name_map_[bias]); 
            emit_mkldnn_memory(out_mem, out, out_dims,
                mkldnn_engine, tensors_name_map_[out]); 
            writer_ << "\n";


            // actual src, weight, dst of mkl convolution memory
            #define EMIT_NEED_REORDER(var, pd, obj, mem)    \
                writer_ << "auto " << var << " = "         \
                    << pd << "." << #obj << "_desc() != "  \
                    << mem << ".get_desc();\n";
            auto need_reorder_in = "need_reorder_" + name + "_in";
            auto need_reorder_w = "need_reorder_" + name + "_w";
            auto need_reorder_out = "need_reorder_" + name + "_out";
            EMIT_NEED_REORDER(need_reorder_in, conv_pd, src, in_mem);
            EMIT_NEED_REORDER(need_reorder_w, conv_pd, weights, w_mem);
            EMIT_NEED_REORDER(need_reorder_out, conv_pd, dst, out_mem);
            writer_ << "\n";


            #define CREATE_CONV_MEM(cmem, flag, pd, obj, mem)                       \
                writer_ << "auto " << cmem << " = " << flag << " ? "                \
                    << "memory(" << pd << "." << #obj << "_desc(), mkldnn_eng)"  \
                    << " : " << mem << ";\n";
            auto src_mem = name+"_src_mem";
            auto weight_mem = name+"_weights_mem";
            auto dst_mem = name+"_dst_mem";
            CREATE_CONV_MEM(src_mem, need_reorder_in, conv_pd, src, in_mem);
            CREATE_CONV_MEM(weight_mem, need_reorder_w, conv_pd, weights, w_mem);
            CREATE_CONV_MEM(dst_mem, need_reorder_out, conv_pd, dst, out_mem);
            writer_ << "\n";


            #define EMIT_REORDER(cmd, flag, from, to)                   \
                writer_ << "if(" << flag<< ") {\n"                      \
                    << "\t" << "auto "<< cmd << " = reorder("           \
                    << from << ", " << to << ");\n";                    \
                writer_ << "\t" << cmd << ".execute(mkldnn_s, {"        \
                    << "{MKLDNN_ARG_FROM, " << from << "},"             \
                    << "{MKLDNN_ARG_TO, " << to << "}});\n";           \
                writer_ << "\t" << "mkldnn_s.wait();\n";                \
                writer_ << "}\n";                                       \

            auto cmd_reorder_in = "reorder_" + name + "_in";
            auto cmd_reorder_w = "reorder_" + name + "_w";
            auto cmd_reorder_out = "reorder_" + name + "_out";
            EMIT_REORDER(cmd_reorder_in, need_reorder_in, in_mem, src_mem);
            EMIT_REORDER(cmd_reorder_w, need_reorder_w, w_mem, weight_mem);
            writer_ << "\n";

             
            // convolution computation
            auto cmd_conv = name + "_forward";
            writer_ << "auto " << cmd_conv << " = "
                << "convolution_forward(" << conv_pd << ");\n";
            writer_ << cmd_conv << ".execute(mkldnn_s, {\n";
            writer_.indentInc();
            writer_ << "{ MKLDNN_ARG_SRC," << src_mem << "},\n"
                << "{ MKLDNN_ARG_WEIGHTS," << weight_mem <<  "},\n"
                << "{ MKLDNN_ARG_BIAS,"<< b_mem <<  "},\n"
                << "{ MKLDNN_ARG_DST," << dst_mem << "} });\n";
            writer_.indentDec();
            writer_ << "mkldnn_s.wait();\n";
            
            EMIT_REORDER(cmd_reorder_out, need_reorder_out, dst_mem, out_mem);

            return;
        }
        
        auto iDims = op->name() + "_inDims";
        auto oDims = op->name() + "_outDims";
        auto fDims = op->name() + "_filterDims";
        auto bDims = op->name() + "_biasDims";
        auto kernelsVar = op->name() + "_filterSizes";
        auto stridesVar = op->name() + "_strides";
        auto padsVar = op->name() + "_pads";

        writer_ << emitArrayDefAndInit(iDims, input->getDims());
        writer_ << emitArrayDefAndInit(oDims, out->getDims());
        writer_ << emitArrayDefAndInit(fDims, filter->getDims());
        writer_ << emitArrayDefAndInit(bDims, bias->getDims());
        writer_ << emitArrayDefAndInit(kernelsVar, kernels);
        writer_ << emitArrayDefAndInit(stridesVar, strides);
        writer_ << emitArrayDefAndInit(padsVar, pads);

        writer_ << "conv2d_" << dtype_flag << "(" << tensors_name_map_[out]
                << ", " << tensors_name_map_[input] << ", "
                << tensors_name_map_[filter] << ", " << tensors_name_map_[bias]
                << ", " << oDims << ", " << iDims << ", " << fDims << ", "
                << bDims << ", " << kernelsVar << ", " << stridesVar << ", "
                << padsVar << ", " << group << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "Conv2dGrad") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *filter = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *bias = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *out = ((TensorNode *)op->getParentNode(3))->getTensor();
        auto *outputG = ((TensorNode *)op->getParentNode(4))->getTensor();

        auto *inputG = ((TensorNode *)op->getChildNode(0))->getTensor();
        auto *filterG = ((TensorNode *)op->getChildNode(1))->getTensor();
        auto *biasG = ((TensorNode *)op->getChildNode(2))->getTensor();

        auto *conv_op = (Conv2dGradOp *)op->getOp();
        auto kernels = conv_op->getKernels();
        auto strides = conv_op->getStrides();
        auto pads = conv_op->getPads();
        auto group = conv_op->getGroup();

        auto iDims = op->name() + "_inDims";
        auto oDims = op->name() + "_outDims";
        auto fDims = op->name() + "_filterDims";
        auto bDims = op->name() + "_biasDims";
        auto kernelsVar = op->name() + "_filterSizes";
        auto stridesVar = op->name() + "_strides";
        auto padsVar = op->name() + "_pads";

        writer_ << emitArrayDefAndInit(iDims, input->getDims());
        writer_ << emitArrayDefAndInit(oDims, out->getDims());
        writer_ << emitArrayDefAndInit(fDims, filter->getDims());
        writer_ << emitArrayDefAndInit(bDims, bias->getDims());
        writer_ << emitArrayDefAndInit(kernelsVar, kernels);
        writer_ << emitArrayDefAndInit(stridesVar, strides);
        writer_ << emitArrayDefAndInit(padsVar, pads);

        writer_ << "conv2dGrad_" << dtype_flag << "(" << tensors_name_map_[inputG]
                << ", " << tensors_name_map_[filterG] << ", "
                << tensors_name_map_[biasG] << ", "
                << tensors_name_map_[outputG] << ", "
                << tensors_name_map_[input] << ", "
                << tensors_name_map_[filter] << ", "
                << oDims << ", " << iDims << ", " << fDims << ", "
                << bDims << ", " << kernelsVar << ", " << stridesVar << ", "
                << padsVar << ", " << group << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "BatchNormalization") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *scale = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *bias = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *mean = ((TensorNode *)op->getParentNode(3))->getTensor();
        auto *var = ((TensorNode *)op->getParentNode(4))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *bn_op = (BatchNormalizationOp *)op->getOp();
        float epsilon = bn_op->getEpsilon();

        if(config_.mkldnn) {
            auto channels = scale->getDims()[0];
            auto in_dims = name+"_in_dims";
            auto out_dims = name+"_out_dims";
            auto scaleshift_dims = name+"_scaleshift_dims";
            emit_mkldnn_memory_dims(in_dims, input->getDims()); 
            emit_mkldnn_memory_dims(out_dims, out->getDims()); 
            emit_mkldnn_memory_dims(scaleshift_dims, {2,channels}); 
            writer_ << "\n";

            auto in_md = name+"_in_md";
            auto out_md = name+"_out_md";
            emit_mkldnn_memory_desc(in_md, in_dims, input); 
            writer_ << "\n";


            // BatchNormalizatio primitive descriptor 
            // only md arguments is src_md, and dst layout is the same as src
            // so no need for format_tag::any and reorder
            auto bn_desc  = name + "_desc";
            writer_ << "auto " << bn_desc << " = batch_normalization_forward::desc("
                << "prop_kind::forward_inference, "
                << in_md << ", "
                << std::setprecision(12) << epsilon << ","
                << "normalization_flags::use_scale_shift" << ");\n";

            // BatchNormalizatio primitive descriptor 
            auto bn_pd = name + "_pd";
            writer_ << "auto " << bn_pd << " = batch_normalization_forward::primitive_desc("
                << bn_desc << ", " << mkldnn_engine << ");\n";
            writer_ << "\n";

            // memory objects: in(may skeep if exists)
            auto in_mem = name+"_in_mem";
            auto scaleshift_mem = name+"_scaleshift_mem";
            auto out_mem = name+"_out_mem";
            emit_mkldnn_memory(in_mem, input, in_dims, 
                mkldnn_engine, tensors_name_map_[input]); 
            // scaleshift_mem actually holds two tensors: scale and bias, so dims is a problem
            // this can be only right when scale and bias are continuous
            writer_ << "auto " << scaleshift_mem << " = memory({{" << scaleshift_dims << "}, "
                << dtype_mkldnn_datatype_map.at("float") << ", "
                << layout_mkldnn_format_tag_map.at("xy") << "}, "
                << mkldnn_engine << ", " << tensors_name_map_[scale]<< ");\n";
            emit_mkldnn_memory(out_mem, out, out_dims,
                mkldnn_engine, tensors_name_map_[out]); 
            writer_ << "\n";

            
            // bn computation
            auto cmd_bn = name + "_forward";
            writer_ << "auto " << cmd_bn << " = "
                << "batch_normalization_forward(" << bn_pd << ");\n";
            writer_ << cmd_bn << ".execute(mkldnn_s, {\n";
            writer_.indentInc();
            writer_ << "{ MKLDNN_ARG_SRC," << in_mem << "},\n"
                << "{ MKLDNN_ARG_SCALE_SHIFT,"<< scaleshift_mem <<  "},\n"
                << "{ MKLDNN_ARG_DST," << out_mem << "} });\n";
            writer_.indentDec();
            writer_ << "mkldnn_s.wait();\n";

            return;
        }

        auto iDims = op->name() + "_inDims";
        writer_ << emitArrayDefAndInit(iDims, input->getDims());

        writer_ << "batchnormalization_" << dtype_flag << "("
                << tensors_name_map_[out] << ", " << tensors_name_map_[input]
                << ", " << tensors_name_map_[mean] << ", "
                << tensors_name_map_[var] << ", " << tensors_name_map_[scale]
                << ", " << tensors_name_map_[bias] << ", " << iDims << ", "
                << std::setprecision(12) << epsilon << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "MaxPool" ||
        (oplabel->getTypeNameLabel()) == "AveragePool") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *pool_op = (MaxPoolOp *)op->getOp();
        auto kernels = pool_op->getKernels();
        auto strides = pool_op->getStrides();
        auto pads = pool_op->getPads();
        
        if(config_.mkldnn) {
            
            auto in_dims = name+"_in_dims";
            auto w_dims = name+"_w_dims";
            auto b_dims = name+"_b_dims";
            auto out_dims = name+"_out_dims";
            auto strides_dims = name + "_strides_dims"; 
            auto kernel_dims = name + "_kernel_dims"; 
            auto pads_dims = name + "_pads_dims"; 
            emit_mkldnn_memory_dims(in_dims, input->getDims()); 
            emit_mkldnn_memory_dims(out_dims, out->getDims()); 
            emit_mkldnn_memory_dims(strides_dims, strides); 
            emit_mkldnn_memory_dims(kernel_dims, kernels); 
            emit_mkldnn_memory_dims(pads_dims, pads); 

            auto in_md = name+"_in_md";
            auto out_md = name+"_out_md";
            emit_mkldnn_memory_desc(in_md, in_dims, input);
            //emit_mkldnn_memory_desc(out_md, out_dims, out, true); 
            emit_mkldnn_memory_desc(out_md, out_dims, out); 
            writer_ << "\n";

            std::string pool_type = ((oplabel->getTypeNameLabel()) == "MaxPool") ?
                "max" : "avg";
            
            auto pool_desc = name+"_desc";
            writer_ << "auto " << pool_desc << " = pooling_forward::desc(prop_kind::forward_inference, "
                // << "algorithm::pooling_max, " // todo pooling_avg
                << "algorithm::pooling_" << pool_type << ", "
                << in_md << ", "
                << out_md << ", "
                << strides_dims <<", "
                << kernel_dims <<", "
                << pads_dims << ", "
                << pads_dims << ");\n";

            // poololution primitive descriptor 
            auto pool_pd = name + "_pd";
            writer_ << "auto " << pool_pd << " = pooling_forward::primitive_desc("
                << pool_desc << ", " << mkldnn_engine << ");\n";
            writer_ << "\n";

            // memory objects
            auto in_mem = name+"_in_mem";
            auto out_mem = name+"_out_mem";
            emit_mkldnn_memory(in_mem, input, in_dims, 
                mkldnn_engine, tensors_name_map_[input]); 
            emit_mkldnn_memory(out_mem, out, out_dims,
                mkldnn_engine, tensors_name_map_[out]); 
            writer_ << "\n";

            /*
            // ok too
            auto cmd_pool = name + "_forward";
            writer_ << "auto " << cmd_pool << " = "
                << "pooling_forward(" << pool_pd << ");\n";
            writer_ << cmd_pool << ".execute(mkldnn_s, {\n";
            writer_.indentInc();
            writer_ << "{ MKLDNN_ARG_SRC," << in_mem << "},\n"
                << "{ MKLDNN_ARG_DST," << out_mem << "} });\n";
            writer_.indentDec();
            writer_ << "mkldnn_s.wait();\n";
            */

            auto need_reorder_out = "need_reorder_" + name + "_out";
            EMIT_NEED_REORDER(need_reorder_out, pool_pd, dst, out_mem);
            writer_ << "\n";

            auto dst_mem = name+"_dst_mem";
            CREATE_CONV_MEM(dst_mem, need_reorder_out, pool_pd, dst, out_mem);
            writer_ << "\n";

            auto cmd_reorder_out = "reorder_" + name + "_out";
            writer_ << "\n";

            // pooling computation
            auto cmd_pool = name + "_forward";
            writer_ << "auto " << cmd_pool << " = "
                << "pooling_forward(" << pool_pd << ");\n";
            writer_ << cmd_pool << ".execute(mkldnn_s, {\n";
            writer_.indentInc();
            writer_ << "{ MKLDNN_ARG_SRC," << in_mem << "},\n"
                << "{ MKLDNN_ARG_DST," << dst_mem << "} });\n";
            writer_.indentDec();
            writer_ << "mkldnn_s.wait();\n";
            
            EMIT_REORDER(cmd_reorder_out, need_reorder_out, dst_mem, out_mem);

            return;
        }
        

        auto iDims = op->name() + "_inDims";
        auto oDims = op->name() + "_outDims";
        auto kernelsVar = op->name() + "_filterSizes";
        auto stridesVar = op->name() + "_strides";
        auto padsVar = op->name() + "_pads";

        writer_ << emitArrayDefAndInit(iDims, input->getDims());
        writer_ << emitArrayDefAndInit(oDims, out->getDims());
        writer_ << emitArrayDefAndInit(kernelsVar, kernels);
        writer_ << emitArrayDefAndInit(stridesVar, strides);
        writer_ << emitArrayDefAndInit(padsVar, pads);

        if ((oplabel->getTypeNameLabel()) == "MaxPool") {
            writer_ << "maxpool_" << dtype_flag << "("
                    << tensors_name_map_[input] << ", "
                    << tensors_name_map_[out] << ", " << iDims << ", " << oDims
                    << ", " << kernelsVar << ", " << stridesVar << ", "
                    << padsVar << ");\n";
        } else {
            writer_ << "avgpool_" << dtype_flag << "("
                    << tensors_name_map_[input] << ", "
                    << tensors_name_map_[out] << ", " << iDims << ", " << oDims
                    << ", " << kernelsVar << ", " << stridesVar << ", "
                    << padsVar << ");\n";
        }
    }

    if ((oplabel->getTypeNameLabel()) == "MaxPoolGrad") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *output = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *outputG = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *inputG = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *pool_op = (MaxPoolGradOp *)op->getOp();
        auto kernels = pool_op->getKernels();
        auto strides = pool_op->getStrides();
        auto pads = pool_op->getPads();

        auto iDims = op->name() + "_inDims";
        auto oDims = op->name() + "_outDims";
        auto kernelsVar = op->name() + "_filterSizes";
        auto stridesVar = op->name() + "_strides";
        auto padsVar = op->name() + "_pads";

        writer_ << emitArrayDefAndInit(iDims, input->getDims());
        writer_ << emitArrayDefAndInit(oDims, output->getDims());
        writer_ << emitArrayDefAndInit(kernelsVar, kernels);
        writer_ << emitArrayDefAndInit(stridesVar, strides);
        writer_ << emitArrayDefAndInit(padsVar, pads);


        writer_ << "maxpoolGrad_" << dtype_flag << "("
                << tensors_name_map_[inputG] << ", "
                << tensors_name_map_[outputG] << ", "
                << tensors_name_map_[input] << ", "
                << iDims << ", " << oDims
                << ", " << kernelsVar << ", " << stridesVar << ", "
                << padsVar << ");\n";

    }

    if ((oplabel->getTypeNameLabel()) == "Relu") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        if(config_.mkldnn) {
            auto in_dims = name+"_in_dims";

            emit_mkldnn_memory_dims(in_dims, input->getDims()); 
            writer_ << "\n";
            auto in_md = name+"_in_md";
            auto out_md = name+"_out_md";
            emit_mkldnn_memory_desc(in_md, in_dims, input); 
            writer_ << "\n";
            
            auto relu_desc  = name + "_desc";
            // https://intel.github.io/mkl-dnn/dev_guide_eltwise.html
            writer_ << "auto " << relu_desc << " = eltwise_forward::desc("
                << "prop_kind::forward_inference, "
                << "algorithm::eltwise_relu, "
                << in_md << ", "
                << "0" << ");\n";


            // BatchNormalizatio primitive descriptor 
            auto relu_pd = name + "_pd";
            writer_ << "auto " << relu_pd << " = eltwise_forward::primitive_desc("
                << relu_desc << ", " << mkldnn_engine << ");\n";
            writer_ << "\n";

            // memory objects: in(may skeep if exists)
            auto in_mem = name+"_in_mem";
            auto out_mem = name+"_out_mem";
            emit_mkldnn_memory(in_mem, input, in_dims, 
                mkldnn_engine, tensors_name_map_[input]); 
            emit_mkldnn_memory(out_mem, out, in_dims,
                mkldnn_engine, tensors_name_map_[out]); 
            writer_ << "\n";

            
            // bn computation
            auto cmd_relu = name + "_forward";
            writer_ << "auto " << cmd_relu << " = "
                << "eltwise_forward(" << relu_pd << ");\n";
            writer_ << cmd_relu << ".execute(mkldnn_s, {\n";
            writer_.indentInc();
            writer_ << "{ MKLDNN_ARG_SRC," << in_mem << "},\n"
                << "{ MKLDNN_ARG_DST," << out_mem << "} });\n";
            writer_.indentDec();
            writer_ << "mkldnn_s.wait();\n";

            return;
        }
        size_t size = input->size();
        writer_ << "relu_" << dtype_flag << "(" << tensors_name_map_[input]
                << ", " << tensors_name_map_[out] << ", " << size << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "ReluGrad") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *outputG = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *inputG = ((TensorNode *)op->getChildNode(0))->getTensor();

        size_t size = input->size();
        writer_ << "reluGrad_" << dtype_flag << "(" << tensors_name_map_[inputG]
                << ", " << tensors_name_map_[input]
                << ", " << tensors_name_map_[outputG] << ", " << size << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "Transpose") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        if(config_.mkldnn) {
            auto idims = (input->getMemLayoutTag()=="nhwc") ?
                input->getShuffledDims(NHWC2NCHW/*{0,3,1,2}*/) : input->getDims();
            auto odims = (out->getMemLayoutTag()=="nhwc") ?
                out->getShuffledDims(NHWC2NCHW/*{0,3,1,2}*/) : out->getDims();

            if(out->getMemLayoutTag()=="nc") {
                input->setMemLayout(layout_cn);
                idims = odims;
            }

            auto in_dims = name+"_in_dims";
            auto out_dims = name+"_out_dims";
            emit_mkldnn_memory_dims(in_dims, odims); 
            emit_mkldnn_memory_dims(out_dims, odims); 
            writer_ << "\n";

            // memory objects
            auto in_mem = name+"_in_mem";
            auto out_mem = name+"_out_mem";
            emit_mkldnn_memory(in_mem, input, in_dims, 
                mkldnn_engine, tensors_name_map_[input]); 
            emit_mkldnn_memory(out_mem, out, out_dims,
                mkldnn_engine, tensors_name_map_[out]); 
            writer_ << "\n";

            
            // bn computation
            auto cmd_reorder = name + "_reorder";
            writer_ << "auto " << cmd_reorder << " = "
                << "reorder(" << in_mem << ", " << out_mem << ");\n";
            writer_ << cmd_reorder << ".execute(mkldnn_s, {\n";
            writer_.indentInc();
            writer_ << "{ MKLDNN_ARG_FROM," << in_mem << "},\n"
                << "{ MKLDNN_ARG_TO," << out_mem << "} });\n";
            writer_.indentDec();
            writer_ << "mkldnn_s.wait();\n";

            return;
        }
        auto *trans_op = (TransposeOp *)op->getOp();
        auto shuffle = trans_op->getShuffle();

        auto iDims = op->name() + "_inDims";
        auto oDims = op->name() + "_outDims";
        auto shuffleDims = op->name() + "_shuffle";

        writer_ << emitArrayDefAndInit(iDims, input->getDims());
        writer_ << emitArrayDefAndInit(oDims, out->getDims());
        writer_ << emitArrayDefAndInit(shuffleDims, shuffle);

        switch (input->getNDim()) {
        case 2:
            writer_ << "transpose2d_" << dtype_flag << "("
                    << tensors_name_map_[input] << ", "
                    << tensors_name_map_[out] << ", " << iDims << ", " << oDims
                    << ", " << shuffleDims << ");\n";
            break;
        case 4:
            writer_ << "transpose4d_" << dtype_flag << "("
                    << tensors_name_map_[input] << ", "
                    << tensors_name_map_[out] << ", " << iDims << ", " << oDims
                    << ", " << shuffleDims << ");\n";
        }
    }

    if ((oplabel->getTypeNameLabel()) == "MatrixTranspose") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        std::vector<size_t> shuffle = {1, 0};

        auto iDims = op->name() + "_inDims";
        auto oDims = op->name() + "_outDims";
        auto shuffleDims = op->name() + "_shuffle";

        writer_ << emitArrayDefAndInit(iDims, input->getDims());
        writer_ << emitArrayDefAndInit(oDims, out->getDims());
        writer_ << emitArrayDefAndInit(shuffleDims, shuffle);

        switch (input->getNDim()) {
        case 2:
            writer_ << "transpose2d_" << dtype_flag << "("
                    << tensors_name_map_[input] << ", "
                    << tensors_name_map_[out] << ", " << iDims << ", " << oDims
                    << ", " << shuffleDims << ");\n";
            break;
        case 4:
            writer_ << "transpose4d_" << dtype_flag << "("
                    << tensors_name_map_[input] << ", "
                    << tensors_name_map_[out] << ", " << iDims << ", " << oDims
                    << ", " << shuffleDims << ");\n";
        }
    }
    if ((oplabel->getTypeNameLabel()).compare("MatrixTanh") == 0) {
        // TODO assert
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        writer_ << "matrixTanh_" << dtype_flag << "(" << m << ", " << n << ", "
                << tensors_name_map_[A] << ", " << n << ", "
                << tensors_name_map_[B] << ", " << n << ");\n";
    }
    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmax") == 0) {
        // TODO assert
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        if(config_.mkldnn) {
            auto in_dims = name+"_in_dims";

            emit_mkldnn_memory_dims(in_dims, input->getDims()); 
            writer_ << "\n";
            auto in_md = name+"_in_md";
            auto out_md = name+"_out_md";
            emit_mkldnn_memory_desc(in_md, in_dims, input); 
            writer_ << "\n";
            
            auto softmax_desc  = name + "_desc";
            writer_ << "auto " << softmax_desc << " = softmax_forward::desc("
                << "prop_kind::forward_inference, "
                << in_md << ", "
                << "1" << ");\n"; // currently only support nc axis=1


            // softmax primitive descriptor 
            auto softmax_pd = name + "_pd";
            writer_ << "auto " << softmax_pd << " = softmax_forward::primitive_desc("
                << softmax_desc << ", " << mkldnn_engine << ");\n";
            writer_ << "\n";

            // memory objects: in(may skeep if exists)
            auto in_mem = name+"_in_mem";
            auto out_mem = name+"_out_mem";
            emit_mkldnn_memory(in_mem, input, in_dims, 
                mkldnn_engine, tensors_name_map_[input]); 
            emit_mkldnn_memory(out_mem, out, in_dims,
                mkldnn_engine, tensors_name_map_[out]); 
            writer_ << "\n";

            
            // bn computation
            auto cmd_softmax = name + "_fwd"; // softmax_forward may conflict with mkldnn::softmax_forward()
            writer_ << "auto " << cmd_softmax << " = "
                << "softmax_forward(" << softmax_pd << ");\n";

            writer_ << cmd_softmax << ".execute(mkldnn_s, {\n";
            writer_.indentInc();
            writer_ << "{ MKLDNN_ARG_SRC," << in_mem << "},\n"
                << "{ MKLDNN_ARG_DST," << out_mem << "} });\n";
            writer_.indentDec();
            writer_ << "mkldnn_s.wait();\n";

            return;
        }
        int m = input->getDim(0);
        int n = input->getDim(1);

        writer_ << "matrixSoftmax_" << dtype_flag << "(" << m << ", " << n
                << ", " << tensors_name_map_[input] << ", " << n << ", "
                << tensors_name_map_[out] << ", " << n << ");\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmaxWithLoss") == 0) {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *label = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *prob = ((TensorNode *)op->getChildNode(0))->getTensor();
        auto *loss = ((TensorNode *)op->getChildNode(1))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        writer_ << "matrixSoftmaxWithLoss_" << dtype_flag << "(" << m << ", " << n
                << ", " << tensors_name_map_[A] << ", " << n << ", "
                << tensors_name_map_[prob] << ", " << n << ", "
                << tensors_name_map_[label] << ", "
                << tensors_name_map_[loss] << ");\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("ArgMax") == 0) {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        auto *argmax_Op = (ArgMaxOp *)op->getOp();
        auto topK = argmax_Op->getTopK();

        assert((size_t)topK==B->getDim(1) && "ArgMax topK != output.dim(2)");

        writer_ << "argMax_" << dtype_flag << "(" << tensors_name_map_[A]
                << ", " << tensors_name_map_[B] << ", " << m << ", " << n
                << ", " << topK << ");\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("Accuracy") == 0) {
        auto *pred = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *label = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *accum = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = pred->getDim(0);
        int n = pred->getDim(1);

        writer_ << "accuracy_" << dtype_flag << "(" << tensors_name_map_[pred] << ", " 
                << tensors_name_map_[label] << ", " 
                << tensors_name_map_[accum] << ", " 
                << m << ", " << n << ");\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("Debug") == 0) {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();

        // assert(A->getNDim() == 2);
        int m = A->getDim(0);
        // int n = A->getDim(1);
        int n = (A->getNDim() == 2) ? A->getDim(1) : 1;

        writer_ << "printMatrix"
                << "(" << tensors_name_map_[A] << ", " << m << ", " << n
                << ");\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("MatrixTanhGrad") == 0) {
        // TODO assert
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *output = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *outputG = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *inputG = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = input->getDim(0);
        int n = input->getDim(1);

        writer_ << "matrixTanhGrad_" << dtype_flag << "(" << m << ", " << n
                << ", " << tensors_name_map_[inputG] << ", " << n << ", "
                << tensors_name_map_[output] << ", " << n << ", "
                << tensors_name_map_[outputG] << ", " << n << ");\n";
    }
    // TODO: depreciate this, do not link label to MatrixSoftmax
    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmaxGrad") == 0) {
        // TODO assert
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *label = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *output = ((TensorNode *)op->getParentNode(2))->getTensor();
        // auto *outputG = ((TensorNode *)op->getParentNode(3))->getTensor();
        auto *inputG = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = input->getDim(0);
        int n = input->getDim(1);

        writer_ << "matrixSoftmaxGrad_" << dtype_flag << "(" << m << ", " << n
                << ", " << tensors_name_map_[inputG] << ", " << n << ", "
                << tensors_name_map_[output] << ", " << n << ", "
                << tensors_name_map_[label] << ");\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmaxWithLossGrad") == 0) {
        // TODO assert
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *label = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *prob= ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *inputG = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = input->getDim(0);
        int n = input->getDim(1);

        writer_ << "matrixSoftmaxWithLossGrad_" << dtype_flag << "(" << m << ", " << n
                << ", " << tensors_name_map_[inputG] << ", " << n << ", "
                << tensors_name_map_[prob] << ", " << n << ", "
                << tensors_name_map_[label] << ");\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("SGD") == 0) {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *inputG = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *momen = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *input_mirror = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *sgdOp = (SGDOp *)op->getOp();
        float lr = sgdOp->getLR();
        float decay = sgdOp->getDecay();
        float momentum = sgdOp->getMomentum();
        size_t batch = sgdOp->getBatch();

        assert(input == input_mirror &&
               "SGD input and output ptr should refer to the same Tensor\n");
        size_t size = input->size();
        writer_ << "sgd_" << dtype_flag << "(" << size << ", "
                << tensors_name_map_[input_mirror] << ", "
                << tensors_name_map_[input] << ", " << tensors_name_map_[inputG]
                << ", " << tensors_name_map_[momen] << ", " << lr << ", "
                << decay << ", " << momentum << ", " << batch << ");\n";
    }
    SWLOG_DEBUG(2) << "end genKernelCall for " << op->name() << "\n";

    if(config_.compute_op_annotation) {
        writer_ << "*/\n";
    }
}

// TODO depreciate this function
std::string Codegen::dtype() { return "float"; }

void Codegen::emitMemFree() {
    SWLOG_DEBUG(4) << "genMemoryFree\n";

    std::string dtype = this->dtype();
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (size == 0)
            continue;

        switch (dev.type) {
        case DeviceType::CPU:
            writer_ << "free(" << base << ");\n";
            break;
        case DeviceType::GPU:
            writer_ << "\n";
            writer_ << "cudaSetDevice(" << dev.id << ");\n";
            writer_ << "cudaFree(" << base << ");\n";
            break;
        default:
            SWLOG_ERROR << "Unknown DeviceType\n";
            break;
        }
    }
    writer_ << "\n";
}
//----------------------------------------------------------
// MKLDNN

void Codegen::emit_mkldnn_memory_dims(std::string name, std::vector<size_t> dims) {
    std::ostringstream os;
    os << "memory::dims " << name << " = {";
    for (auto dim : dims)
        os << dim << ", ";

    std::string str = os.str();
    writer_ << str.substr(0, str.length() - 2) + "};\n";

}
// currently only supoort format_tag::any
void Codegen::emit_mkldnn_memory_desc(std::string &name, std::string mkldnn_dims, Tensor *tensor, std::string layout_tag) {
    std::string dtype = getTypeString(tensor);
    assert(dtype_mkldnn_datatype_map.count(dtype) && "mkldnn unsupported data type\n");

    std::string layout = layout_tag.length()==0 ?
        tensor->getMemLayoutTag() : layout_tag;

    auto tensor_layout = std::make_pair(tensor, layout);

    if(layout_tag == "any") {
        writer_ << "auto " << name << " = memory::desc({" << mkldnn_dims << "}, "
            << dtype_mkldnn_datatype_map.at(dtype) << ", "
            << "memory::format_tag::any);\n";
    }
    else if(tensors_mkldnn_mem_map_.count(tensor_layout)) {
        auto mkldnn_mem = tensors_mkldnn_mem_map_.at(tensor_layout);
        writer_ << "// " << name << " alias to " << mkldnn_mem << "\n"; 
        writer_ << "auto " << name << " = " << mkldnn_mem + ".get_desc();\n";
    }
    else {
        writer_ << "auto " << name << " = memory::desc({" << mkldnn_dims << "}, "
            << dtype_mkldnn_datatype_map.at(dtype) << ", "
            << layout_mkldnn_format_tag_map.at(layout) << ");\n";
    }
}


void Codegen::emit_mkldnn_memory(std::string &name, Tensor *tensor, std::string mkldnn_dims, std::string engine, std::string handle, std::string layout_tag) { 
    std::string dtype = getTypeString(tensor);
    assert(dtype_mkldnn_datatype_map.count(dtype) && "mkldnn unsupported data type\n");

    std::string layout = layout_tag.length()==0 ?
        tensor->getMemLayoutTag() : layout_tag;

    SWLOG_DEBUG(10) << "emit_mkldnn_memory for " << name << " " << layout << "\n";

    auto tensor_layout = std::make_pair(tensor, layout);

    if(tensors_mkldnn_mem_map_.count(tensor_layout)) {
        auto mkldnn_mem = tensors_mkldnn_mem_map_.at(tensor_layout);
        writer_ << "// " << name << " alis to " << mkldnn_mem << "\n"; 
        name = mkldnn_mem;
        return;
    }

    std::string format_tag = layout_tag.length()==0 ? 
        layout_mkldnn_format_tag_map.at(layout) : "memory::format_tag::" + layout_tag;

    writer_ << "auto " << name << " = memory({{" << mkldnn_dims<< "}, "
        << dtype_mkldnn_datatype_map.at(dtype) << ", "
        // << layout_mkldnn_format_tag_map.at(layout) << "}, "
        << format_tag << "}, "
        << engine << ", " << handle << ");\n";
    
    tensors_mkldnn_mem_map_[tensor_layout] = name;  
}
    
//----------------------------------------------------------

void Codegen::emitFuncCallCUDA(OpNode *op) {
    std::string dtype_flag = dtype();

    Label *oplabel = op->getLabel();
    SWLOG_DEBUG(2) << "genKernelCall for " << oplabel->getTypeNameLabel()
                   << std::endl;

    // TODO assert legal dimensions
    if ((oplabel->getTypeNameLabel()).compare("MatrixMatrixMul") == 0) {

        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = C->getDim(0);
        int k = A->getDim(1);
        int n = C->getDim(1);
        if (config_.cublas) {
            std::string alpha = UniqueName("alpha");
            std::string beta = UniqueName("beta");
            writer_ << "const float " << alpha << "=1, " << beta << "=0;\n";
            writer_ << "cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,\n";
            writer_.indentInc();
            writer_ << n << ", " << m << ", " << k << ",\n";
            writer_ << "&" << alpha << ",\n";
            writer_ << tensors_name_map_[B] << ", " << n << ",\n";
            writer_ << tensors_name_map_[A] << ", " << k << ",\n";
            writer_ << "&" << beta << ",\n";
            writer_ << tensors_name_map_[C] << ", " << n << ");\n";
            writer_.indentDec();
        }
    }
    if ((oplabel->getTypeNameLabel()).compare("MatrixTanh") == 0) {
        // TODO assert
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        writer_ << "matrixTanh_" << dtype_flag << "<<<1, " << m
                << ", 0, stream[" << oplabel->getDeviceLabel().id << "]>>>("
                << tensors_name_map_[A] << ", " << tensors_name_map_[B] << ", "
                << n << ");\n";
    }
    if ((oplabel->getTypeNameLabel()) == "BatchedAdd" || (oplabel->getTypeNameLabel()) == "MatrixVectorAdd") {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();

        size_t sliceNum, sliceSize;
        std::tie(sliceNum, sliceSize) = convertToDim2(A->getDims());
        auto bdim = B->size();
        (void)bdim;
        assert((sliceSize == bdim) &&
               "batch flattened dim.second != bias dim!");

        writer_ << "batchedadd_" << dtype_flag << "<<<1, " << sliceNum
                << ", 0, stream[" << oplabel->getDeviceLabel().id << "]>>>("
                << tensors_name_map_[C] << ", " << tensors_name_map_[A] << ", "
                << tensors_name_map_[B] << ", " << sliceSize << ");\n";
    }
    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmax") == 0) {
        // TODO assert
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        writer_ << "matrixSoftmax_" << dtype_flag << "<<<1, " << m
                << ", 0, stream[" << oplabel->getDeviceLabel().id << "]>>>("
                << tensors_name_map_[A] << ", " << tensors_name_map_[B] << ", "
                << n << ");\n";
    }
}


} // namespace codegen
} // namespace swc
