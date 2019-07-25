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
    if (d.type == DeviceType::CPU) {
        if(d.id == INT_MAX)
            os << "cpup";
        else
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

int Codegen::getMPISendRecvTag(Tensor *tensor) {
    int idx = 0;
    for (auto &t : mpi_sendRecv_tags_) {
        if (t == tensor)
            return idx;
        idx++;
    }
    mpi_sendRecv_tags_.push_back(tensor);
    return idx;
}

bool Codegen::delMPISendRecvTag(Tensor *tensor) {
    for (auto it = mpi_sendRecv_tags_.begin(); it != mpi_sendRecv_tags_.end();
         it++) {
        if (*it == tensor) {
            mpi_sendRecv_tags_.erase(it);
            return true;
        }
    }
    return false;
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

    Device cpu1;
    cpu1.id = 1;
    Device cpu2;
    cpu2.id = 2;

    Device cpup;
    cpup.id = INT_MAX;

    auto m_cpu0 = std::make_shared<MemoryAllocator>(cpu0, "cpu0", 0xFFFFFFFF);
    auto m_cpu1 = std::make_shared<MemoryAllocator>(cpu1, "cpu1", 0xFFFFFFFF);
    auto m_cpu2 = std::make_shared<MemoryAllocator>(cpu2, "cpu2", 0xFFFFFFFF);
    auto m_gpu0 = std::make_shared<MemoryAllocator>(gpu0, "gpu0", 0xFFFFFFFF);
    auto m_gpu1 = std::make_shared<MemoryAllocator>(gpu1, "gpu1", 0xFFFFFFFF);
    p_mem_alllocator_ = std::make_shared<MemoryAllocator>(cpup, "cpup", 0xFFFFFFFF);

    mem_allocators_.push_back(m_cpu0);
    mem_allocators_.push_back(m_cpu1);
    mem_allocators_.push_back(m_cpu2);
    mem_allocators_.push_back(m_gpu0);
    mem_allocators_.push_back(m_gpu1);

    dev_allocator_map_[cpu0] = m_cpu0.get();
    dev_allocator_map_[cpu1] = m_cpu1.get();
    dev_allocator_map_[cpu2] = m_cpu2.get();
    dev_allocator_map_[gpu0] = m_gpu0.get();
    dev_allocator_map_[gpu1] = m_gpu1.get();
    dev_allocator_map_[cpup] = p_mem_alllocator_.get();

    m_cpu0->setBasePtrName(UniqueName(deviceToStr(cpu0) + "_baseptr"));
    m_cpu1->setBasePtrName(UniqueName(deviceToStr(cpu1) + "_baseptr"));
    m_cpu2->setBasePtrName(UniqueName(deviceToStr(cpu2) + "_baseptr"));
    p_mem_alllocator_->setBasePtrName(UniqueName(deviceToStr(cpup) + "_baseptr"));

    m_gpu0->setBasePtrName(UniqueName(deviceToStr(gpu0) + "_baseptr"));
    m_gpu1->setBasePtrName(UniqueName(deviceToStr(gpu1) + "_baseptr"));
}
void Codegen::codeGenInit() {
    initMemoryAllocators();
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

void Codegen::emitMPIInit() {
    if (config_.mpi) {
        writer_
            << "// ========================================================\n";
        writer_ << "// MPI INIT\n";
        writer_ << "int rank, nprocs;\n";
        writer_ << "char proc_name[MPI_MAX_PROCESSOR_NAME];\n";
        writer_ << "int proc_name_len;\n";
        writer_ << "MPI_Status status;\n";

        writer_ << "MPI_Init(&argc, &argv);\n";
        writer_ << "MPI_Comm_size(MPI_COMM_WORLD, &nprocs);\n";
        writer_ << "MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n";
        writer_ << "MPI_Get_processor_name(proc_name,&proc_name_len);\n";
        writer_ << "std::cout << \"process \" << rank << \" of \" << nprocs \
			<< \" run on \" << proc_name << std::endl;\n";
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

std::string Codegen::generate() {
    codeGenInit();

    std::ostringstream ss;
    ss << "/*******************************************************************"
          "******\n"
       << "  > File Name: graph.cpp\n"
       // << "  > Author: none\n"
       // << "  > Mail:  \n"
       // << "  > Created Time: "
       << "  > IRGraph\n"
       << "  > |-TensorNode " << graph_->tensorNodeNum() << "\n"
       << "  > |-opNode     " << graph_->opNodeNum() << "\n"
       << " *******************************************************************"
          "*****/\n";

    writer_ << "#include <iostream>\n"
            << "#include <random>\n"
            << "#include <stdlib.h>\n"
            << "#include <math.h>\n"
            << "#include \"utils/image.h\"\n";

    if (config_.mpi) {
        writer_ << "#include <mpi.h>\n";
    }

    if (config_.cuda) {
        writer_ << "#include <cuda.h>\n"
                << "#include <cublas_v2.h>\n";
        // #include "utils/cuda_kernels.cu"
        // writer_ << CUDA_CODE;
        writer_ << "#include \"utils/cuda_kernels.h\"\n";
    }

    // #include "kernels.h"
    // writer_ << KERNELS_CODE;
    writer_ << "#include \"utils/kernels.h\"\n"
            << "#include \"utils/DataLoader.h\"\n"
            << "#include \"utils/utils.h\"\n";

    if (config_.train_mode) {
        writer_ << "#include \"gflags/gflags.h\"\n"
                << "#include <google/protobuf/io/coded_stream.h>\n"
                << "#include <google/protobuf/io/zero_copy_stream_impl.h>\n\n";

        emitGflagsDef();
    }

    writer_ << "\n\n"
            << "int main(int argc, char** argv) {\n";
    writer_.indentInc();

    if (config_.train_mode) {
        writer_ << "gflags::ParseCommandLineFlags(&argc, &argv, true);\n";
    }

    emitCUDAInit();
    emitMPIInit();

    writer_ << "\n// variable declaration and initiation\n";
    emitMemAllocs();

    writer_ << "\n// call op routine functions\n";
    emitExecute();
    // emitFuncCalls();

    writer_ << "\n// free memory\n";
    emitMemFree();

    emitMPIFinalize();

    writer_ << "return 0;\n";

    writer_.indentDec();
    writer_ << "}\n";

    std::ofstream fout("Graph.cpp", std::fstream::out);
    fout << ss.str() + writer_.get_code();
    fout.close();

    fout.flush();
    // TODO : NVCC
    if (config_.mpi) {
        makefile_builder_.setCXXCompiler("mpic++");
    } else {
        makefile_builder_.setCXXCompiler("/usr/bin/c++");
    }

    makefile_builder_.addCXXSrc("Graph.cpp");
    makefile_builder_.addCXXSrc("utils/DataLoader.cpp");
    makefile_builder_.addCXXSrc("caffe2.pb.cc");
    makefile_builder_.addIncDir("/usr/local/include");

    makefile_builder_.addLibDir("/usr/local/lib");
    makefile_builder_.addLib("protobuf");
    makefile_builder_.addLib("gflags");

    fout.open("G_Makefile", std::fstream::out);
    fout << makefile_builder_.generate();
    fout.close();

    return ss.str() + writer_.get_code();
}

void Codegen::emitMemAllocs() {
    SWLOG_DEBUG(4) << "genMemAllocs \n";

    allocateMemAddr();

    emitVarDeclarations();

    emitMemAllocations();

    emitTensorAddresses();

    if (config_.train_mode)
        emitDataLoaderInit();

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

    for (int i = 0; i < graph->tensorNodeNum(); i++) {
        TensorNode *tnode = graph->getTensorNode(i);
        Tensor *tensor = tnode->getTensor();

        if (tensors_name_map_.count(tensor))
            continue;

        std::string bufferName = UniqueName(tnode->name());

        size_t size = tensor->getSizeInBytes();

        Label *label = tnode->getLabel();
        Device dev = label->getDeviceLabel();

        SWLOG_DEBUG(1) << "allocateMemAddr " << tnode->name() << " " << size
                       << " on dev(" << static_cast<int>(dev.type) << ", "
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
        // tensors_offset_map_[tensor] = std::make_pair(base, addr /
        // sizeof(float));
        tensors_offset_map_[tensor] = std::make_pair(base, addr);
        // tensors_base_map_[tensor] = base;
    }
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
    // std::string dtype = this->dtype();
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (size == 0)
            continue;

        if (config_.mpi) {
            writer_ << "if(rank == 0) {\n";
            writer_.indentInc();
        }

        emitMemAllocation(base, size, dev);

        if (config_.mpi) {
            writer_.indentDec();
            writer_ << "} // if rank\n";
        }
    }
    
    if(p_mem_alllocator_->getMemAllocated()) {

        auto dev = p_mem_alllocator_->getDevice();
        std::string base = p_mem_alllocator_->getBasePtrName();
        uint64_t size = p_mem_alllocator_->getMemAllocated();

        emitMemAllocation(base, size, dev);
    }
    writer_ << "\n";
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

/// if config_.mpi=true this func deal with
/// the MASTER(0) process
void Codegen::emitTensorAddresses() {
    SWLOG_DEBUG(4) << "begin emitTensorAddresse...\n";

    std::set<Tensor *> visited_tensors;

    if (config_.mpi) {
        writer_ << "if(rank == 0) {\n";
        writer_.indentInc();
    }

    emitTensorAddresses(graph_, &visited_tensors);

    if (config_.mpi) {
        writer_.indentDec();
        writer_ << "} // if rank\n";
    }

    for (int i = 0; i < graph_->opNodeNum(); i++) {
        OpNode *opnode = graph_->getOpNode(i);
        if (auto graphOp = dynamic_cast<SubGraphOp *>(opnode->getOp())) {
            if (auto ngraph = graphOp->getGraph()) {
                switchTo(ngraph);
                Device dev = ngraph->getDeviceLabel();
                if (config_.mpi && dev.type == DeviceType::CPU) {
                    writer_ << "if(rank ==" << dev.id << ") {\n";
                    writer_.indentInc();
                }
                emitTensorAddresses(ngraph, &visited_tensors);
                if (config_.mpi && dev.type == DeviceType::CPU) {
                    writer_.indentDec();
                    writer_ << "} // if rank\n";
                }
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

    if (config_.mpi) {
        writer_ << "if(rank == 0) {\n";
        writer_.indentInc();
    }

    if (config_.train_mode) {
        emitTensorInitFromSnapshot(graph_, &visited_tensors);
    } else {
        emitTensorInitializations(graph_, &visited_tensors);
    }

    if (config_.mpi) {
        writer_.indentDec();
        writer_ << "} // if rank\n";
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
        // writer_ << name << " = reinterpret_cast<" << dtype
        //        << "*>(" << base << " + " << offset << ");\n";

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
            dispathOpNode(op);
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
            dispathOpNode(op);
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
    writer_ << "if(iter % " << config_.train_config.display<< " == 0) {\n";
    writer_.indentInc();

    writer_ << R"(std::cout << "iterations " << iter << "\n";)"
            << "\n";
    for(int i=0; i<graph_->outNodeNum(); i++) {
        TensorNode * outnode = graph_->getOutNode(i);
        Tensor* out = outnode->getTensor();
        int m = out->getDim(0);
        int n = out->getNDim()==2 ? out->getDim(1) : 1;
        writer_ << "// OutNode " << i << ": " << outnode->name() << "\n";
        writer_ << "std::cout << \"" << outnode->name() <<":\\n\";" << "\n";
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

static std::string getBytesProtoString(BytesProto proto) {
    switch (proto) {
    case ONE_BYTE_AS_INT:
        return "ONE_BYTE_AS_INT";
    case FOUR_BYTES_AS_FLOAT:
        return "FOUR_BYTES_AS_FLOAT";
    default:
        return "ONE_BYTE_AS_INT";
    }
}

static std::string getInitialLizerString(const std::vector<size_t> &dims) {
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
    // DataLoader loader(filename, BytesProto::ONE_BYTE_AS_INT,
    // BytesProto::FOUR_BYTES_AS_FLOAT, 1, 60000, {8u}, {8u, 28u, 28u, 1u});
    //
    writer_ << "std::string train_data_file = \""
            << config_.train_config.train_data_file << "\";\n";
    writer_ << "DataLoader loader(";
    // writer_ << "\"" << config_.train_config.train_data_file << "\", ";
    writer_ << "train_data_file, ";
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

void Codegen::emitExecute() {
    if (config_.train_mode) {
        TensorNode *label = graph_->getTrainLabelNode();
        TensorNode *data = graph_->getTrainDataNode();

        std::string label_var = tensors_name_map_.at(label->getTensor());
        std::string data_var = tensors_name_map_.at(data->getTensor());
        /*
        std::string label_var = label->name();
        std::string data_var = data->name();
        */
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
    }
}

void Codegen::emitFuncCalls() {
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
                    dispathOpNode(opnode);
                }
            }
        }
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
                } else {
                    dispathOpNode(opnode);
                }
            }
        }
}

void Codegen::switchTo(IRGraph *ngraph) {
    // Device host = this->graph_->getDeviceLabel();
    Device dev = ngraph->getDeviceLabel();
    if (dev.type == DeviceType::CPU) {
        // TODO MPI

    } else if (dev.type == DeviceType::GPU) {
        writer_ << "cudaSetDevice(" << dev.id << ");\n";
    }
}

void Codegen::switchFrom(IRGraph *ngraph) { (void)ngraph; }

void Codegen::dispathOpNode(OpNode *op) {
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
        if (config_.mpi) {
            writer_ << "if(rank == " << dev.id << ") {\n";
            writer_.indentInc();
        }

        switch (dev.type) {
        case DeviceType::CPU:
            emitFuncCall(op);
            break;
        case DeviceType::GPU:
            emitFuncCallCUDA(op);
            break;
        default:
            SWLOG_ERROR << "unknown device type in dispathOpNode\n";
        }

        if (config_.mpi) {
            writer_.indentDec();
            writer_ << "} // if rank\n";
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

    if (from_dev.type == DeviceType::CPU && to_dev.type == DeviceType::CPU &&
        from_dev.id != to_dev.id) {
        if (!config_.mpi)
            return;

        int tag = getMPISendRecvTag(to);
        writer_ << "if(rank == " << from_dev.id << ") {\n";
        writer_.indentInc();
        writer_ << "MPI_Send(" << fname << "+" << from_offset << ", " << size
                << ", "
                << "MPI_CHAR, " << to_dev.id << ", " << tag
                << ",  MPI_COMM_WORLD);\n";
        writer_.indentDec();
        writer_ << "} // if rank\n";

        writer_ << "if(rank == " << to_dev.id << ") {\n";
        writer_.indentInc();
        writer_ << "MPI_Recv(" << tname << "+" << to_offset << ", " << size
                << ", "
                << "MPI_CHAR, " << from_dev.id << ", " << tag
                << ",  MPI_COMM_WORLD, &status);\n";
        writer_.indentDec();
        writer_ << "} // if rank\n";
    }
}

void Codegen::emitFuncCall(OpNode *op) {
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
        SWLOG_ERROR << "UNKNOWN DataType\n";
    }

    Label *oplabel = op->getLabel();
    SWLOG_DEBUG(2) << "genKernelCall for " << oplabel->getTypeNameLabel()
                   << std::endl;

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

    if ((oplabel->getTypeNameLabel()).compare("Reshape") == 0) {
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
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        writer_ << "matrixSoftmax_" << dtype_flag << "(" << m << ", " << n
                << ", " << tensors_name_map_[A] << ", " << n << ", "
                << tensors_name_map_[B] << ", " << n << ");\n";
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

        writer_ << "argMax_" << dtype_flag << "(" << tensors_name_map_[A]
                << ", " << tensors_name_map_[B] << ", " << m << ", " << n
                << ", " << topK << ");\n";
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
                << decay << ", " << momentum << ", " << batch << ");\n ";
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

        if (config_.mpi) {
            writer_ << "if(rank == " << dev.id << ") {\n";
            writer_.indentInc();
        }

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

        if (config_.mpi) {
            writer_.indentDec();
            writer_ << "} // if rank\n";
        }
    }

    writer_ << "\n";
}

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

void Codegen::emitMPIFinalize() {
    if (config_.mpi)
        writer_ << "MPI_Finalize();\n";
}

} // namespace codegen
} // namespace swc
