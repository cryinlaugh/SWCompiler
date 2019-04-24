/*************************************************************************
    > File Name: Codegen.cpp
    > Author: wayne
    > Mail:
    > Created Time: 二  1/22 10:32:13 2019
 ************************************************************************/

#include "Codegen.h"
#include "SWC.h"
#include <cassert>
#include <fstream>
#include <iomanip>
#include <string>

using namespace swc::op;

namespace swc {
namespace codegen {

#define stream                                                                 \
    genIndent();                                                               \
    stream_

static std::string deviceToStr(const Device &d) {
    std::ostringstream os;
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

    auto m_cpu0 = std::make_shared<MemoryAllocator>(cpu0, "cpu0", 0xFFFFFFFF);
    auto m_gpu0 = std::make_shared<MemoryAllocator>(gpu0, "gpu0", 0xFFFFFFFF);
    auto m_gpu1 = std::make_shared<MemoryAllocator>(gpu1, "gpu1", 0xFFFFFFFF);

    mem_allocators_.push_back(m_cpu0);
    mem_allocators_.push_back(m_gpu0);
    mem_allocators_.push_back(m_gpu1);

    dev_allocator_map_[cpu0] = m_cpu0.get();
    dev_allocator_map_[gpu0] = m_gpu0.get();
    dev_allocator_map_[gpu1] = m_gpu1.get();

    m_cpu0->setBasePtrName(UniqueName(deviceToStr(cpu0) + "_baseptr"));
    m_gpu0->setBasePtrName(UniqueName(deviceToStr(gpu0) + "_baseptr"));
    m_gpu1->setBasePtrName(UniqueName(deviceToStr(gpu1) + "_baseptr"));
}
void Codegen::codeGenInit() {
    // todo clear of new vector and map
    // names_map_.clear();
    // tensors_name_map_.clear();
    // tensors_offset_map_.clear();
    this->active_graph_ = graph_;
    indent_ = 0;
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

    if (flag_use_cublas) {
        stream << "cublasStatus_t stat;\n";
        stream << "cublasHandle_t handle;\n";
        stream << "stat = cublasCreate(&handle);\n";
        stream << "if (stat != CUBLAS_STATUS_SUCCESS) {\n";
        stream << "    printf (\"CUBLAS initialization failed\\n\");\n";
        stream << "    return EXIT_FAILURE;\n";
        stream << "}\n\n";
    }

    stream << "cudaStream_t stream[" << N << "];\n";
    stream << "for(int i=0; i<" << N << "; i++)\n";
    this->indent_++;
    stream << "cudaStreamCreate(&stream[i]);\n\n";
    this->indent_--;
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

    stream << "#include <iostream>\n"
           << "#include <random>\n"
           << "#include <stdlib.h>\n"
           << "#include <math.h>\n"
           << "#include \"image.h\"\n";

    if (flag_use_cublas) {
        stream << "#include <cuda.h>\n"
               << "#include <cublas_v2.h>\n";
#include "cuda_kernels.cu"
        stream << CUDA_CODE;
    }

#include "kernels.h"
    stream << KERNELS_CODE;

    stream << "\n"
           << "int main(){\n";
    indent_++;

    emitCUDAInit();

    stream << "\n";
    stream << "// variable declaration and initiation\n";
    emitMemAllocs();

    stream << "\n";
    stream << "// call op routine functions\n";
    emitFuncCalls();

    stream << "\n";
    stream << "// free memory\n";
    emitMemFree();

    stream << "return 0;\n";

    indent_--;
    stream << "}\n";

    std::ofstream fout("Graph.cpp", std::fstream::out);
    fout << ss.str() + stream_.str();
    fout.close();

    return ss.str() + stream_.str();
}

void Codegen::emitMemAllocs() {
    SWLOG_INFO << "genMemAllocs \n";

    allocateMemAddr();

    emitVarDeclarations();

    emitMemAllocations();

    emitTensorInitializations();
}
void Codegen::allocateMemAddr() {
    SWLOG_INFO << "begin allocateMemAddr...\n";

    allocateMemAddr(graph_);
    for (int i = 0; i < graph_->opNodeNum(); i++) {
        OpNode *opnode = graph_->getOpNode(i);
        if (auto graphOp = dynamic_cast<SubGraphOp *>(opnode->getOp())) {
            if (graphOp->getGraph())
                SWLOG_INFO << "allocateMemAddr on subG: " << opnode->name()
                           << "\n";
            allocateMemAddr(graphOp->getGraph());
        }
    }
    SWLOG_INFO << "end allocateMemAddr...\n";
}
void Codegen::allocateMemAddr(IRGraph *graph_) {

    for (int i = 0; i < graph_->tensorNodeNum(); i++) {
        TensorNode *tnode = graph_->getTensorNode(i);
        Tensor *tensor = tnode->getTensor();

        if (tensors_name_map_.count(tensor))
            continue;

        std::string bufferName = UniqueName(tnode->name());

        size_t size = tensor->getSizeInBytes();

        Label *label = tnode->getLabel();
        Device dev = label->getDeviceLabel();

        SWLOG_INFO << "allocateMemAddr " << tnode->name() << " " << size
                   << " on dev(" << static_cast<int>(dev.type) << ", " << dev.id
                   << ")."
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
    SWLOG_INFO << "begin emitVarDeclarations...\n";

    // std::string dtype = this->dtype();
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        std::string base = allocator->getBasePtrName();
        // stream << dtype << " *" << base << ";\n";
        stream << "char *" << base << ";\n";
    }

    /*
    for(int i=0 ; i<graph_->tensorNodeNum(); i++){
        auto *tensor = graph_->getTensorNode(i)->getTensor();
        std::string name = tensors_name_map_[tensor];
        stream << dtype << " *" << name << ";\n";
    }
    */
    // for (auto it : tensors_name_map_) {
    //     stream << dtype << " *" << it.second << ";\n";
    // }

    stream << "\n";

    SWLOG_INFO << "end emitVarDeclarations...\n";
}

void Codegen::emitMemAllocations() {
    // std::string dtype = this->dtype();
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (size == 0)
            continue;
        switch (dev.type) {
        case DeviceType::CPU:
            // stream << base << " = (" << dtype << "*)malloc(" << size <<
            // ");\n";
            stream << base << " = (char*)malloc(" << size << ");\n";
            break;
        case DeviceType::GPU:
            stream << "\n";
            stream << "cudaSetDevice(" << dev.id << ");\n";
            stream << "cudaMalloc(&" << base << ", " << size << ");\n";
            break;
        default:
            SWLOG_ERROR << "Unknown DeviceType\n";
            break;
        }
    }
    stream << "\n";
}

void Codegen::emitTensorInitializations() {
    SWLOG_INFO << "begin emitTensorInitializations...\n";

    std::set<Tensor *> visited_tensors;

    emitTensorInitializations(graph_, &visited_tensors);
    for (int i = 0; i < graph_->opNodeNum(); i++) {
        OpNode *opnode = graph_->getOpNode(i);
        if (auto graphOp = dynamic_cast<SubGraphOp *>(opnode->getOp())) {
            if (auto ngraph = graphOp->getGraph()) {
                switchTo(ngraph);
                emitTensorInitializations(ngraph, &visited_tensors);
                stream << "\n";
            }
        }
    }

    SWLOG_INFO << "end emitTensorInitializations...\n";
}

void Codegen::emitTensorInitializations(IRGraph *graph_,
                                        std::set<Tensor *> *visited_tensors) {
    for (int i = 0; i < graph_->tensorNodeNum(); i++) {
        auto *tnode = graph_->getTensorNode(i);
        auto *tensor = tnode->getTensor();

        if (visited_tensors->count(tensor))
            continue;
        visited_tensors->insert(tensor);

        std::string dtype;
        switch (tnode->getDataType()) {
        case DataType::Float_t:
            dtype = "float";
            break;
        case DataType::Double_t:
            dtype = "double";
            break;
        case DataType::Int32_t:
            dtype = "int";
            break;
        default:
            SWLOG_ERROR << "UNKNOWN DataType\n";
        }

        std::string name = tensors_name_map_[tensor];
        uint64_t size = tensor->size();
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];
        stream << dtype << "* " << name << " = reinterpret_cast<" << dtype
               << "*>(" << base << " + " << offset << ");\n";

        TensorInitInfo info = tensor->getTensorInitInfo();
        switch (tensor->getTensorInitType()) {
        case TensorInitType::NONE:
            break;
        case TensorInitType::XAVIER: {
            // TODO
            stream << "initTensorXavier(" << name << ", " << size << ", "
                   << info.getFilterSize() << ");\n";
            break;
        }
        case TensorInitType::CONSTANT: {
            stream << "initTensorConstant(" << name << ", " << size << ", "
                   << info.getConstant() << ");\n";
            break;
        }
        case TensorInitType::ZERO: {
            stream << "initTensorZero(" << name << ", " << size << ");\n";
            break;
        }
        case TensorInitType::FILE: {
            stream << "load(" << name << ", " << size << ", "
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
            SWLOG_INFO << name << " TensorInitType= NONE\n";
            break;

        } // switch
    }     // tensor loop

    stream << "\n";
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
        stream << "float *" << bufferName
               << " = (float *)malloc(sizeof(float) * " << size << ");\n";
        break;
    case DataType::Double_t:
        stream << "double *" << bufferName
               << " = (double *)malloc(sizeof(double) * " << size << ");\n";
        break;
    default:
        SWLOG_ERROR << "UNKNOWN DataType\n";
    }

    tensors_name_map_[tnode->getTensor()] = bufferName;
    return bufferName;
}

void Codegen::emitFuncCalls() {
    for (int i = 0; i < graph_->topologyNum(); i++)
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto node = graph_->getNodeInTopo(i, j);
            if (node->nodeType() == OP_NODE) {
                auto opnode = (OpNode *)node;
                stream << "\n";
                stream << "// topology(" << i << ", " << j
                       << "): " << opnode->name() << " : "
                       << opnode->getOpName() << "\n";
                if (auto graphOp =
                        dynamic_cast<SubGraphOp *>(opnode->getOp())) {
                    if (auto ngraph = graphOp->getGraph()) {
                        switchTo(ngraph);
                        emitFuncCalls(ngraph);
                        stream << "\n";
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
                stream << "// topology(" << i << ", " << j
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
        stream << "cudaSetDevice(" << dev.id << ");\n";
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

        emitMemcpyFromTo(from_tensor, from_dev, offset, size, to_tensor, dev);
    } else if (auto gather = dynamic_cast<GatherOp *>(op->getOp())) {
        auto *from = ((TensorNode *)op->getParentNode(0));
        auto *from_tensor = from->getTensor();
        auto *to = ((TensorNode *)op->getChildNode(0));
        auto *to_tensor = to->getTensor();
        Device to_dev = to->getLabel()->getDeviceLabel();

        size_t offset = gather->getOffset();
        size_t size = from_tensor->getSizeInBytes();

        emitMemcpyFromTo(from_tensor, dev, offset, size, to_tensor, to_dev);
    } else {
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
    }
}

void Codegen::emitMemcpyFromTo(Tensor *from, Device from_dev, size_t offset,
                               size_t size, Tensor *to, Device to_dev) {
    std::string fname = tensors_name_map_[from];
    std::string tname = tensors_name_map_[to];
    if (from_dev.type == DeviceType::CPU && to_dev.type == DeviceType::GPU) {
        if (flag_multiStream) {
            stream << "cudaMemcpyAsync(" << tname << ", " << fname << "+"
                   << offset << ", " << size << ", "
                   << "cudaMemcpyHostToDevice, stream[" << to_dev.id << "]);\n";
        } else {
            stream << "cudaMemcpy(" << tname << ", " << fname << "+" << offset
                   << ", " << size << ", "
                   << "cudaMemcpyHostToDevice);\n";
        }
    }

    if (from_dev.type == DeviceType::GPU && to_dev.type == DeviceType::CPU) {
        if (flag_multiStream) {
            stream << "cudaMemcpyAsync(" << tname << "+" << offset << ", "
                   << fname << ", " << size << ", "
                   << "cudaMemcpyDeviceToHost, stream[" << from_dev.id
                   << "]);\n";

        } else {
            stream << "cudaMemcpy(" << tname << "+" << offset << ", " << fname
                   << ", " << size << ", "
                   << "cudaMemcpyDeviceToHost);\n";
        }
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
    SWLOG_INFO << "genKernelCall for " << oplabel->getTypeNameLabel()
               << std::endl;

    // TODO assert legal dimensions
    if ((oplabel->getTypeNameLabel()).compare("MatrixMatrixMul") == 0) {
        // TODO assert
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = C->getDim(0);
        int k = B->getDim(0);
        int n = C->getDim(1);

        stream << "matrixMatrixMul_" << dtype_flag << "(" << m << ", " << n
               << ", " << k << ", " << tensors_name_map_[A] << ", " << k << ", "
               << tensors_name_map_[B] << ", " << n << ", "
               << tensors_name_map_[C] << ", " << n << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "BatchedAdd") {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();

        size_t sliceNum, sliceSize;
        std::tie(sliceNum, sliceSize) = convertToDim2(A->getDims());
        auto bdim = B->size();
        (void)bdim;
        assert((sliceSize == bdim) &&
               "batch flattened dim.second != bias dim!");

        stream << "batchedadd_" << dtype_flag << "(" << tensors_name_map_[C]
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

        stream << "batchedreduceadd_" << dtype_flag << "("
               << tensors_name_map_[output] << ", " << tensors_name_map_[input]
               << ", " << sliceNum << ", " << sliceSize << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "ElementAdd") {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto num = A->size();

        stream << "vecAdd_" << dtype_flag << "(" << num << ", "
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

        stream << emitArrayDefAndInit(iDims, input->getDims());
        stream << emitArrayDefAndInit(oDims, out->getDims());
        stream << emitArrayDefAndInit(fDims, filter->getDims());
        stream << emitArrayDefAndInit(bDims, bias->getDims());
        stream << emitArrayDefAndInit(kernelsVar, kernels);
        stream << emitArrayDefAndInit(stridesVar, strides);
        stream << emitArrayDefAndInit(padsVar, pads);

        stream << "conv2d_" << dtype_flag << "(" << tensors_name_map_[out]
               << ", " << tensors_name_map_[input] << ", "
               << tensors_name_map_[filter] << ", " << tensors_name_map_[bias]
               << ", " << oDims << ", " << iDims << ", " << fDims << ", "
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
        stream << emitArrayDefAndInit(iDims, input->getDims());

        stream << "batchnormalization_" << dtype_flag << "("
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

        stream << emitArrayDefAndInit(iDims, input->getDims());
        stream << emitArrayDefAndInit(oDims, out->getDims());
        stream << emitArrayDefAndInit(kernelsVar, kernels);
        stream << emitArrayDefAndInit(stridesVar, strides);
        stream << emitArrayDefAndInit(padsVar, pads);

        if ((oplabel->getTypeNameLabel()) == "MaxPool") {
            stream << "maxpool_" << dtype_flag << "("
                   << tensors_name_map_[input] << ", " << tensors_name_map_[out]
                   << ", " << iDims << ", " << oDims << ", " << kernelsVar
                   << ", " << stridesVar << ", " << padsVar << ");\n";
        } else {
            stream << "avgpool_" << dtype_flag << "("
                   << tensors_name_map_[input] << ", " << tensors_name_map_[out]
                   << ", " << iDims << ", " << oDims << ", " << kernelsVar
                   << ", " << stridesVar << ", " << padsVar << ");\n";
        }
    }

    if ((oplabel->getTypeNameLabel()) == "Relu") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        size_t size = input->size();
        stream << "relu_" << dtype_flag << "(" << tensors_name_map_[input]
               << ", " << tensors_name_map_[out] << ", " << size << ");\n";
    }

    if ((oplabel->getTypeNameLabel()) == "Transpose") {
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *trans_op = (TransposeOp *)op->getOp();
        auto shuffle = trans_op->getShuffle();

        auto iDims = op->name() + "_inDims";
        auto oDims = op->name() + "_outDims";
        auto shuffleDims = op->name() + "_shuffle";

        stream << emitArrayDefAndInit(iDims, input->getDims());
        stream << emitArrayDefAndInit(oDims, out->getDims());
        stream << emitArrayDefAndInit(shuffleDims, shuffle);

        switch (input->getNDim()) {
        case 2:
            stream << "transpose2d_" << dtype_flag << "("
                   << tensors_name_map_[input] << ", " << tensors_name_map_[out]
                   << ", " << iDims << ", " << oDims << ", " << shuffleDims
                   << ");\n";
            break;
        case 4:
            stream << "transpose4d_" << dtype_flag << "("
                   << tensors_name_map_[input] << ", " << tensors_name_map_[out]
                   << ", " << iDims << ", " << oDims << ", " << shuffleDims
                   << ");\n";
        }
    }

    if ((oplabel->getTypeNameLabel()).compare("MatrixTanh") == 0) {
        // TODO assert
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        stream << "matrixTanh_" << dtype_flag << "(" << m << ", " << n << ", "
               << tensors_name_map_[A] << ", " << n << ", "
               << tensors_name_map_[B] << ", " << n << ");\n";
    }
    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmax") == 0) {
        // TODO assert
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        stream << "matrixSoftmax_" << dtype_flag << "(" << m << ", " << n
               << ", " << tensors_name_map_[A] << ", " << n << ", "
               << tensors_name_map_[B] << ", " << n << ");\n";
    }
    if ((oplabel->getTypeNameLabel()).compare("MatrixTanhGrad") == 0) {
        // TODO assert
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *output = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *outputG = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *inputG = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = input->getDim(0);
        int n = input->getDim(1);

        stream << "matrixTanhGrad_" << dtype_flag << "(" << m << ", " << n
               << ", " << tensors_name_map_[inputG] << ", " << n << ", "
               << tensors_name_map_[output] << ", " << n << ", "
               << tensors_name_map_[outputG] << ", " << n << ");\n";
    }
    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmaxGrad") == 0) {
        // TODO assert
        auto *input = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *label = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *output = ((TensorNode *)op->getParentNode(2))->getTensor();
        // auto *outputG = ((TensorNode *)op->getParentNode(3))->getTensor();
        auto *inputG = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = input->getDim(0);
        int n = input->getDim(1);

        stream << "matrixSoftmaxGrad_" << dtype_flag << "(" << m << ", " << n
               << ", " << tensors_name_map_[inputG] << ", " << n << ", "
               << tensors_name_map_[output] << ", " << n << ", "
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
        stream << "sgd_" << dtype_flag << "(" << size << ", "
               << tensors_name_map_[input_mirror] << ", "
               << tensors_name_map_[input] << ", " << tensors_name_map_[inputG]
               << ", " << tensors_name_map_[momen] << ", " << lr << ", "
               << decay << ", " << momentum << ", " << batch << ");\n ";
    }
}

// TODO depreciate this function
std::string Codegen::dtype() { return "float"; }

void Codegen::emitMemFree() {
    SWLOG_INFO << "genMemoryFree\n";

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
            stream << "free(" << base << ");\n";
            break;
        case DeviceType::GPU:
            stream << "\n";
            stream << "cudaSetDevice(" << dev.id << ");\n";
            stream << "cudaFree(" << base << ");\n";
            break;
        default:
            SWLOG_ERROR << "Unknown DeviceType\n";
            break;
        }
    }

    stream << "\n";
}

void Codegen::emitFuncCallCUDA(OpNode *op) {
    std::string dtype_flag = dtype();

    Label *oplabel = op->getLabel();
    SWLOG_INFO << "genKernelCall for " << oplabel->getTypeNameLabel()
               << std::endl;

    // TODO assert legal dimensions
    if ((oplabel->getTypeNameLabel()).compare("MatrixMatrixMul") == 0) {

        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = C->getDim(0);
        int k = A->getDim(1);
        int n = C->getDim(1);
        if (flag_use_cublas) {
            std::string alpha = UniqueName("alpha");
            std::string beta = UniqueName("beta");
            stream << "const float " << alpha << "=1, " << beta << "=0;\n";
            stream << "cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,\n";
            indent_++;
            stream << n << ", " << m << ", " << k << ",\n";
            stream << "&" << alpha << ",\n";
            stream << tensors_name_map_[B] << ", " << n << ",\n";
            stream << tensors_name_map_[A] << ", " << k << ",\n";
            stream << "&" << beta << ",\n";
            stream << tensors_name_map_[C] << ", " << n << ");\n";
            indent_--;
        }
    }
    if ((oplabel->getTypeNameLabel()).compare("MatrixTanh") == 0) {
        // TODO assert
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        stream << "matrixTanh_" << dtype_flag << "<<<1, " << m << ", 0, stream["
               << oplabel->getDeviceLabel().id << "]>>>("
               << tensors_name_map_[A] << ", " << tensors_name_map_[B] << ", "
               << n << ");\n";
    }
    if ((oplabel->getTypeNameLabel()) == "BatchedAdd") {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();

        size_t sliceNum, sliceSize;
        std::tie(sliceNum, sliceSize) = convertToDim2(A->getDims());
        auto bdim = B->size();
        (void)bdim;
        assert((sliceSize == bdim) &&
               "batch flattened dim.second != bias dim!");

        stream << "batchedadd_" << dtype_flag << "<<<1, " << sliceNum
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

        stream << "matrixSoftmax_" << dtype_flag << "<<<1, " << m
               << ", 0, stream[" << oplabel->getDeviceLabel().id << "]>>>("
               << tensors_name_map_[A] << ", " << tensors_name_map_[B] << ", "
               << n << ");\n";
    }
}

} // namespace codegen
} // namespace swc
