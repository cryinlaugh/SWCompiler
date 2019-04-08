/*************************************************************************
    > File Name: Codegen.cpp
    > Author: wayne
    > Mail: 
    > Created Time: 二  1/22 10:32:13 2019
 ************************************************************************/

#include "SWC.h"
#include "Codegen.h"
#include <fstream>
#include <string>
#include <cassert>

namespace swc{
namespace codegen{

#define stream \
    genIndent(); \
    stream_

static std::string deviceToStr(const Device& d){
    std::ostringstream os;
    if(d.type == DeviceType::CPU){
        os << "cpu" << d.id;
    }else if(d.type == DeviceType::GPU){
        os << "gpu" << d.id;
    }
    return os.str();
}

static std::pair<size_t, size_t> convertToDim2(const std::vector<size_t> &dims){
    size_t second = 1;
    for(size_t i=1; i<dims.size(); i++)
        second *= dims[i];

    return std::make_pair(dims[0], second);
}

static std::string emitArrayDefAndInit(std::string name, const std::vector<size_t> &dims){
    std::ostringstream os;
    os << "const size_t " << name << "[] = {";
    for(auto dim : dims)
        os << dim << ", ";
    
    std::string str = os.str();
    return str.substr(0, str.length()-2)+"};\n"; 
} 

template <typename Dtype>
void Codegen<Dtype>::destroy(){
    graph_ = nullptr;
    names_map_.clear();
    tensors_name_map_.clear();
    tensors_offset_map_.clear();
}
template <typename Dtype>
void Codegen<Dtype>::initMemoryAllocators(){
    Device cpu0;
    Device gpu0; gpu0.type=DeviceType::GPU; gpu0.id=0;
    Device gpu1; gpu1.type=DeviceType::GPU; gpu1.id=1;  

    auto m_cpu0 = std::make_shared<MemoryAllocator>(cpu0, "cpu0", 0xFFFFFFFF);
    auto m_gpu0 = std::make_shared<MemoryAllocator>(gpu0, "gpu0", 0xFFFFFFFF);
    auto m_gpu1 = std::make_shared<MemoryAllocator>(gpu1, "gpu1", 0xFFFFFFFF); 
    
    mem_allocators_.push_back(m_cpu0);
    // mem_allocators_.push_back(m_gpu0);
    // mem_allocators_.push_back(m_gpu1);
    
    dev_allocator_map_[cpu0] = m_cpu0.get();
    dev_allocator_map_[gpu0] = m_gpu0.get();
    dev_allocator_map_[gpu1] = m_gpu1.get();

    m_cpu0->setBasePtrName(UniqueName(deviceToStr(cpu0)+"_baseptr"));
    m_gpu0->setBasePtrName(UniqueName(deviceToStr(gpu0)+"_baseptr"));
    m_gpu1->setBasePtrName(UniqueName(deviceToStr(gpu1)+"_baseptr"));
}

template <typename Dtype>
void Codegen<Dtype>::emitCUDAInit(){
    //TODO create stream depending on number of device or config
    // one stream per device 
    int N = 0;
    for(auto allocator : mem_allocators_){
        Device dev = allocator->getDevice(); 
        if(dev.type == DeviceType::GPU)
            N++; 
    }
    if(N == 0)
        return;

    if(flag_use_cublas){
        stream << "cublasStatus_t stat;\n";
        stream << "cublasHandle_t handle;\n";
        stream << "stat = cublasCreate(&handle);\n";
        stream << "if (stat != CUBLAS_STATUS_SUCCESS) {\n";
        stream << "    printf (\"CUBLAS initialization failed\\n\");\n";
        stream << "    return EXIT_FAILURE;\n";
        stream << "}\n\n";
    }

    stream << "cudaStream_t stream[" << N << "];\n";
    stream << "for(int i=0; i<" << N <<"; i++)\n";
    this->indent_++;
    stream << "cudaStreamCreate(&stream[i]);\n\n";
    this->indent_--;
}

template <typename Dtype>
void Codegen<Dtype>::codeGenInit(){
    // todo clear of new vector and map
    // names_map_.clear();
    // tensors_name_map_.clear();
    // tensors_offset_map_.clear();
    this->active_graph_= graph_; 
    indent_ = 0;
    initMemoryAllocators();
}

template<typename Dtype>
std::string Codegen<Dtype>::UniqueName(std::string name){
    auto iter  = names_map_.find(name);
    if(iter != names_map_.end()){
        std::string uname = name;
        std::ostringstream os;
        while(names_map_.count(uname) != 0){
            os << name << (++iter->second);
            uname = os.str();
        }
        name = uname;
    }
    names_map_[name] = 0;
    return name;
}

template<typename Dtype>
std::string Codegen<Dtype>::generate(){
    codeGenInit();
    
    std::ostringstream ss;
    ss << "/*************************************************************************\n"
        << "  > File Name: graph.cpp\n"
        // << "  > Author: none\n"
        // << "  > Mail:  \n"
        // << "  > Created Time: "
        << "  > IRGraph\n"
        << "  > |-TensorNode " << graph_->tensorNodeNum() << "\n"
        << "  > |-opNode     " << graph_->opNodeNum() << "\n"
        << " ************************************************************************/\n"; 

    stream << "#include <iostream>\n"
            << "#include <random>\n"
            << "#include <stdlib.h>\n"
            << "#include <math.h>\n"
            << "#include \"image.h\"\n";

    if(flag_use_cublas){
        stream << "#include <cuda.h>\n"
            << "#include <cublas_v2.h>\n";
#include "cuda_kernels.cu"
        stream << CUDA_CODE;
    }

#include "kernels.h"
    stream << KERNELS_CODE;

    stream << "\n" << "int main(){\n";
    indent_++;

    stream << "\n";
    stream << "// allocate memory\n";
    genMemAllocs();

    stream<< "\n";
    genFuncCalls();

    stream << "\n";
    stream << "// free memory\n";
    genMemFree();

    stream << "return 0;\n";

    indent_--;
    stream<< "}\n";
 
    std::ofstream fout("Graph.cpp", std::fstream::out);
    fout << ss.str() + stream_.str();
    fout.close();

    return ss.str() + stream_.str();
}

template<typename Dtype>
void Codegen<Dtype>::genFuncCalls(){
    for (int i = 0; i < graph_->topologyNum(); i++)  
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto node = graph_->getNodeInTopo(i, j);
            if (node->nodeType() == OP_NODE){
                auto opnode = (OpNode<Dtype>*)node;
                stream << "\n";
                stream << "// topology(" << i << ", " << j << "): " << opnode->name() 
                        << " : " << opnode->getOpName() <<"\n";
                if(auto graphOp = dynamic_cast<SubGraphOp<Dtype>*>(opnode->getOp())){
                    if(auto ngraph = graphOp->getGraph()){
                        switchTo(ngraph);
                        genFuncCalls(ngraph);
                        stream << "\n";
                    }
                }else {
                    dispathOpNode(opnode);
                }
            }
        }
}
template<typename Dtype>
void Codegen<Dtype>::switchTo(IRGraph<Dtype>* ngraph){
    //Device host = this->graph_->getDeviceLabel();
    Device dev = ngraph->getDeviceLabel();
    if(dev.type == DeviceType::CPU){
        //TODO MPI
    }
    else if(dev.type == DeviceType::GPU){
        stream <<  "cudaSetDevice(" << dev.id << ");\n";
    }
}
template<typename Dtype>
void Codegen<Dtype>::switchFrom(IRGraph<Dtype>* ngraph){

}

template<typename Dtype>
void Codegen<Dtype>::genFuncCalls(IRGraph<Dtype>* graph_){
    for (int i = 0; i < graph_->topologyNum(); i++)  
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto node = graph_->getNodeInTopo(i, j);
            if (node->nodeType() == OP_NODE){
                stream << "// topology(" << i << ", " << j << "): " << node->name() << "\n";
                auto opnode = (OpNode<Dtype>*)node;
                if(auto graphOp = dynamic_cast<SubGraphOp<Dtype>*>(opnode->getOp())){
                }else {
                    dispathOpNode(opnode);
                }
            }
        }
}

template<typename Dtype>
void Codegen<Dtype>::dispathOpNode(OpNode<Dtype>* op){
    if(!op->runable())
        return;

    Label *label = op->getLabel();
    Device dev = label->getDeviceLabel();
    if(auto scatter = dynamic_cast<ScatterOp<Dtype>*>(op->getOp())){
        auto* from = ((TensorNode<Dtype>*)op->getParentNode(0));  
        auto* from_tensor = from->getTensor();
        Device from_dev = from->getLabel()->getDeviceLabel();
        auto* to = ((TensorNode<Dtype>*)op->getChildNode(0));  
        auto* to_tensor = to->getTensor();

        size_t offset  = scatter->getOffset();
        size_t size = from_tensor->getSizeInBytes();
        
        emitMemcpyFromTo(from_tensor, from_dev, offset, size,
            to_tensor, dev); 
    }
    else if(auto gather = dynamic_cast<GatherOp<Dtype>*>(op->getOp())){
        auto* from = ((TensorNode<Dtype>*)op->getParentNode(0));  
        auto* from_tensor = from->getTensor();
        auto* to = ((TensorNode<Dtype>*)op->getChildNode(0));  
        auto* to_tensor = to->getTensor();
        Device to_dev= to->getLabel()->getDeviceLabel();

        size_t offset  = gather->getOffset();
        size_t size = from_tensor->getSizeInBytes();
        
        emitMemcpyFromTo(from_tensor, dev, offset, size,
            to_tensor, to_dev); 
    }else{
        switch(dev.type){
        case DeviceType::CPU:
            genFuncCall(op); 
            break;
        case DeviceType::GPU:
            genFuncCallCUDA(op); 
            break;
        default:
            SWLOG_ERROR << "unknown device type in dispathOpNode\n";
        }
    }
}

template<typename Dtype>
void Codegen<Dtype>::emitMemcpyFromTo(Tensor<Dtype>* from, Device from_dev, size_t offset, size_t size,
            Tensor<Dtype>* to, Device to_dev){
    std::string fname = tensors_name_map_[from];
    std::string tname = tensors_name_map_[to];
    if(from_dev.type==DeviceType::CPU && to_dev.type==DeviceType::GPU){
        if(flag_multiStream){
            stream << "cudaMemcpyAsync(" <<  tname << ", " << fname << "+" << offset << ", "
                << size << ", " << "cudaMemcpyHostToDevice, stream[" << to_dev.id << "]);\n";
        }else{
            stream << "cudaMemcpy(" <<  tname << ", " << fname << "+" << offset << ", "
                << size << ", " << "cudaMemcpyHostToDevice);\n";
        }
    }  

    if(from_dev.type==DeviceType::GPU && to_dev.type==DeviceType::CPU){
        if(flag_multiStream){
            stream << "cudaMemcpyAsync(" <<  fname << ", " << tname << "+" << offset << ", "
                << size << ", " << "cudaMemcpyDeviceToHost, stream[" << to_dev.id << "]);\n";

        }else{
            stream << "cudaMemcpy(" <<  fname << ", " << tname << "+" << offset << ", "
                << size << ", " << "cudaMemcpyDeviceToHost);\n";
        }
    }  
}

template<typename Dtype>
void Codegen<Dtype>::genFuncCall(OpNode<Dtype>* op){
    std::string dtype_flag;
    if(auto node = dynamic_cast<OpNode<float>*>(op)){
        dtype_flag = "f";
    }else if(auto node = dynamic_cast<OpNode<double>*>(op)){
        dtype_flag = "d";
    }else{
        SWLOG_ERROR << "Unknown Dtype\n";
    }
    
    Label* oplabel = op->getLabel();
    SWLOG_INFO << "genKernelCall for " << oplabel->getTypeNameLabel() << std::endl;

    // TODO assert legal dimensions
    if ((oplabel->getTypeNameLabel()).compare("MatrixMatrixMul") == 0) {
        //TODO assert
        auto* A = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* B = ((TensorNode<Dtype>*)op->getParentNode(1))->getTensor();  
        auto* C = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  
        int m = C->getDim(0);
        int k = B->getDim(0);
        int n = C->getDim(1);

        stream << "matrixMatrixMul_"<< dtype_flag 
                << "(" << m << ", " << n << ", " << k << ", "
                << tensors_name_map_[A] << ", " << k << ", "
                << tensors_name_map_[B] << ", " << n << ", "
                << tensors_name_map_[C] << ", " << n << ");\n";    
        
    }
    
    if((oplabel->getTypeNameLabel()) == "BatchedAdd"){
        auto* A = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* B = ((TensorNode<Dtype>*)op->getParentNode(1))->getTensor();  
        auto* C = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  
    
        size_t sliceNum, sliceSize;
        std::tie(sliceNum, sliceSize) = convertToDim2(A->getDims());
        auto bdim = B->size();
        (void)bdim;
        assert((sliceSize==bdim) && "batch flattened dim.second != bias dim!"); 

        stream << "batchedadd_" << dtype_flag
                << "(" << tensors_name_map_[C] << ", " 
                << tensors_name_map_[A] << ", " 
                << tensors_name_map_[B] << ", " 
                << sliceNum << ", " << sliceSize << ");\n";
    }

    if((oplabel->getTypeNameLabel()) == "Conv2d"){
        auto* input = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* filter = ((TensorNode<Dtype>*)op->getParentNode(1))->getTensor();  
        auto* bias = ((TensorNode<Dtype>*)op->getParentNode(2))->getTensor();  
        auto* out = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  

        auto *conv_op = (Conv2dOp<Dtype>*)op->getOp();
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

        stream << "conv2d_"  << dtype_flag << "("        
                << tensors_name_map_[out] << ", "
                << tensors_name_map_[input] << ", "
                << tensors_name_map_[filter] << ", "
                << tensors_name_map_[bias] << ", "
                << oDims << ", "
                << iDims << ", "
                << fDims << ", "
                << bDims << ", "
                << kernelsVar << ", "
                << stridesVar << ", "
                << padsVar << ", "
                << group << ");\n";
    }

    if((oplabel->getTypeNameLabel()) == "MaxPool"){
        auto* input = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* out = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  

        auto *pool_op= (MaxPoolOp<Dtype>*)op->getOp();
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

        stream << "maxpool_"  << dtype_flag << "("        
                << tensors_name_map_[input] << ", "
                << tensors_name_map_[out] << ", "
                << iDims << ", "
                << oDims << ", "
                << kernelsVar << ", "
                << stridesVar << ", "
                << padsVar << ");\n";
    }

    if((oplabel->getTypeNameLabel()) == "Relu"){
        auto* input = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* out = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  
        
        size_t size = input->size();
        stream << "relu_" << dtype_flag << "("
                << tensors_name_map_[input] << ", "
                << tensors_name_map_[out] << ", "
                << size << ");\n";
            
    }

    if((oplabel->getTypeNameLabel()) == "Transpose"){
        auto* input = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* out = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  

        auto *trans_op= (TranposeOp<Dtype>*)op->getOp();
        auto shuffle = trans_op->getShuffle();

        auto iDims = op->name() + "_inDims";
        auto oDims = op->name() + "_outDims";
        auto shuffleDims = op->name() + "_shuffle";

        stream << emitArrayDefAndInit(iDims, input->getDims()); 
        stream << emitArrayDefAndInit(oDims, out->getDims()); 
        stream << emitArrayDefAndInit(shuffleDims, shuffle); 
        
        switch(input->getNDim()){
        case 2:
            stream << "transpose2d_" << dtype_flag << "("
                << tensors_name_map_[input] << ", "
                << tensors_name_map_[out] << ", "
                << iDims << ", "
                << oDims << ", "
                << shuffleDims << ");\n";
            break;
        case 4:
            stream << "transpose4d_" << dtype_flag << "("
                << tensors_name_map_[input] << ", "
                << tensors_name_map_[out] << ", "
                << iDims << ", "
                << oDims << ", "
                << shuffleDims << ");\n";
            
        }
            
    }

    if ((oplabel->getTypeNameLabel()).compare("MatrixTanh") == 0) {
        //TODO assert
        auto* A = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* B = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  
        int m = A->getDim(0);
        int n = A->getDim(1);
        
        stream << "matrixTanh_"<< dtype_flag 
                << "(" << m << ", " << n << ", "
                << tensors_name_map_[A] << ", " << n << ", "
                << tensors_name_map_[B] << ", " << n << ");\n";    
        
    } 
    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmax") == 0) {
        //TODO assert
        auto* A = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* B = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  
        int m = A->getDim(0);
        int n = A->getDim(1);
        
        stream << "matrixSoftmax_"<< dtype_flag 
                << "(" << m << ", " << n << ", "
                << tensors_name_map_[A] << ", " << n << ", "
                << tensors_name_map_[B] << ", " << n << ");\n";    
        
    }      
}

// TODO depreciate this function 
template<typename Dtype>
void Codegen<Dtype>::genMemAllocs(){
    SWLOG_INFO << "genMemAllocs \n";

    allocateMemAddr();

    emitVarDeclarations();

    emitCUDAInit();
    emitMemAllocations();

    emitTensorInitializations();

}

template<typename Dtype>
std::string Codegen<Dtype>::genTensorMemAlloc(TensorNode<Dtype>* tnode){
    std::string bufferName = UniqueName(tnode->name()); 
    int dims = tnode->getTensor()->getNDim(); 
    size_t size = 1;
    for(int dim=0; dim<dims; dim++)
        size *= tnode->getTensor()->getDim(dim);

    if(auto node = dynamic_cast<TensorNode<float>*>(tnode)){
        stream << "float *" << bufferName
               << " = (float *)malloc(sizeof(float) * " << size << ");\n";
    }
    else if(auto node = dynamic_cast<TensorNode<double>*>(tnode)){
        stream << "double *" << bufferName
               << " = (double *)malloc(sizeof(double) * " << size << ");\n";
    }

    tensors_name_map_[tnode->getTensor()] = bufferName;
    return bufferName;
}

template <typename Dtype>
void Codegen<Dtype>::allocateMemAddr(){
    SWLOG_INFO << "---- " << "begin allocateMemAddr...\n";

    allocateMemAddr(graph_);
    for(int i=0; i<graph_->opNodeNum(); i++){
        OpNode<Dtype>* opnode= graph_->getOpNode(i);
        if(auto graphOp = dynamic_cast<SubGraphOp<Dtype>*>(opnode->getOp())){
            if(graphOp->getGraph())
                allocateMemAddr(graphOp->getGraph()); 
        }
    }

    SWLOG_INFO << "---- " << "end allocateMemAddr...\n";
}

template <typename Dtype>
void Codegen<Dtype>::allocateMemAddr(IRGraph<Dtype>* graph_){
    
    for(int i=0 ; i<graph_->tensorNodeNum(); i++){
        TensorNode<Dtype>* tnode= graph_->getTensorNode(i);
        Tensor<Dtype>* tensor = tnode->getTensor();

        if(tensors_name_map_.count(tensor))
            continue;

        std::string bufferName = UniqueName(tnode->name()); 

        size_t size = tensor->getSizeInBytes();

        SWLOG_INFO << "---- " << tnode->name() << " " << size << "\n";
        
        Label* label = tnode->getLabel();
        Device dev = label->getDeviceLabel();

        auto *allocator = dev_allocator_map_[dev];
        if(!allocator){
            SWLOG_ERROR << "allocator" << static_cast<int>(dev.type) << " " << dev.id << " not found\n";
        }
        uint64_t addr = allocator->allocate(tensor, size);
        std::string base = allocator->getBasePtrName();

        tensors_name_map_[tensor] = bufferName;
        tensors_offset_map_[tensor] = std::make_pair(base, addr/sizeof(Dtype));
        // tensors_base_map_[tensor] = base;
    }
}

template<> std::string Codegen<float>::dtype(){ return "float"; }
template<> std::string Codegen<double>::dtype(){ return "double"; }

template <typename Dtype>
void Codegen<Dtype>::emitVarDeclarations(){
    SWLOG_INFO << "---- " << "begin emitVarDeclarations...\n";

    std::string dtype = this->dtype();
    for(auto m : mem_allocators_){
        MemoryAllocator *allocator = m.get();
        std::string base = allocator->getBasePtrName();    
        stream << dtype << " *" << base << ";\n";    
    }

    
    /*
    for(int i=0 ; i<graph_->tensorNodeNum(); i++){
        auto *tensor = graph_->getTensorNode(i)->getTensor();
        std::string name = tensors_name_map_[tensor];
        stream << dtype << " *" << name << ";\n";
    }
    */
    for(auto it : tensors_name_map_){
        stream << dtype << " *" << it.second << ";\n";
    }

    stream << "\n";

    SWLOG_INFO << "---- " << "end emitVarDeclarations...\n";
}

template <typename Dtype>
void Codegen<Dtype>::emitMemAllocations(){
    std::string dtype = this->dtype();
    for(auto m : mem_allocators_){
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if(size ==0 )
            continue;
        switch(dev.type){
        case DeviceType::CPU:
            stream << base << " = (" << dtype << "*)malloc(" << size << ");\n";
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

template <typename Dtype>
void Codegen<Dtype>::emitTensorInitializations(){
    SWLOG_INFO << "---- " << "begin emitTensorInitializations...\n";

    std::set<Tensor<Dtype>*> visited_tensors;

    emitTensorInitializations(graph_, &visited_tensors);
    for(int i=0; i<graph_->opNodeNum(); i++){
        OpNode<Dtype>* opnode= graph_->getOpNode(i);
        if(auto graphOp = dynamic_cast<SubGraphOp<Dtype>*>(opnode->getOp())){
            if(auto ngraph = graphOp->getGraph()){
                switchTo(ngraph);
                emitTensorInitializations(ngraph, &visited_tensors); 
                stream << "\n";
            }
        }
    }

    SWLOG_INFO << "---- " << "end emitTensorInitializations...\n";
}

template <typename Dtype>
void Codegen<Dtype>::emitTensorInitializations(IRGraph<Dtype>* graph_,
        std::set<Tensor<Dtype>*> *visited_tensors){
    for(int i=0 ; i<graph_->tensorNodeNum(); i++){
        auto *tnode = graph_->getTensorNode(i);
        auto *tensor = tnode->getTensor();

        if(visited_tensors->count(tensor))
            continue;
        visited_tensors->insert(tensor);

        std::string dtype = this->dtype();
        std::string name = tensors_name_map_[tensor];
        uint64_t size = tensor->size();
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];
        stream << name << " = " << base << " + " << offset << ";\n";

        TensorInitInfo<Dtype> info = tensor->getTensorInitInfo();
        switch(tensor->getTensorInitType()) {
            case TensorInitType::NONE:
                break;
            case TensorInitType::XAVIER: {
                //TODO 
                stream << "initTensorXavier(" << name << ", "
                        << size << ", " << info.getFilterSize() <<");\n";
                break;
            }
            case TensorInitType::CONSTANT: {
                stream << "initTensorConstant(" << name << ", "
                        << size << ", " << "1.0f);\n"; 
                break;
            }
            case TensorInitType::ZERO: {
                stream << "initTensorZero(" << name << ", "
                        << size << ");\n"; 
                break;
            }
            case TensorInitType::FILE: {
                stream << "load(" << name << ", "
                        << size << ", 0, \"" << info.getFilePath() << "\");\n";
                break;
            }
            case TensorInitType::PARENTOP: {
                auto *op = (OpNode<Dtype>*)tnode->getParentNode(0); 
                dispathOpNode(op);
                break;
            }
            default:
                SWLOG_INFO << name << " TensorInitType= NONE\n"; 
                break;

        } // switch
    } // tensor loop

    stream << "\n";
}

template<typename Dtype>
void Codegen<Dtype>::genMemFree(){
    SWLOG_INFO << "genMemoryFree\n";

    std::string dtype = this->dtype();
    for(auto m : mem_allocators_){
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if(size ==0 )
            continue;
        switch(dev.type){
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
template<typename Dtype>
void Codegen<Dtype>::genFuncCallCUDA(OpNode<Dtype>* op){
    std::string dtype_flag = dtype();
    
    Label* oplabel = op->getLabel();
    SWLOG_INFO << "genKernelCall for " << oplabel->getTypeNameLabel() << std::endl;

    // TODO assert legal dimensions
    if ((oplabel->getTypeNameLabel()).compare("MatrixMatrixMul") == 0) {

        auto* A = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* B = ((TensorNode<Dtype>*)op->getParentNode(1))->getTensor();  
        auto* C = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  
        int m = C->getDim(0); 
        int k = A->getDim(1);
        int n = C->getDim(1);
        if(flag_use_cublas){
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
        //TODO assert
        auto* A = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* B = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  
        int m = A->getDim(0);
        int n = A->getDim(1);
        
        stream << "matrixTanh_"<< dtype_flag 
                << "<<<1, " << m << ", 0, stream[" 
                << oplabel->getDeviceLabel().id << "]>>>"  
                << tensors_name_map_[A] << ", " 
                << tensors_name_map_[B] << ", " << n << ");\n";    
        
    } 
    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmax") == 0) {
        //TODO assert
        auto* A = ((TensorNode<Dtype>*)op->getParentNode(0))->getTensor();  
        auto* B = ((TensorNode<Dtype>*)op->getChildNode(0))->getTensor();  
        int m = A->getDim(0);
        int n = A->getDim(1);
        
        stream << "matrixSoftmax_"<< dtype_flag 
                << "<<<1, " << m << ", 0, stream[" 
                << oplabel->getDeviceLabel().id << "]>>>"  
                << tensors_name_map_[A] << ", " 
                << tensors_name_map_[B] << ", " << n << ");\n";    
        
    }      
}

INSTANTIATE_CLASS(Codegen);

} //namespace codegen
} // namespace swc



