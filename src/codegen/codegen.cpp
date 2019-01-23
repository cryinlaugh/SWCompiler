/*************************************************************************
	> File Name: codegen.cpp
	> Author: wayne
	> Mail: singleon11@gmail.com 
	> Created Time: 二  1/22 10:32:13 2019
 ************************************************************************/

#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/IRGraph.h"
#include "common.h"
#include "SWLOG.h"
#include "pass/Label.h"

#include "codegen.h"
#include <fstream>
#include <string>

namespace swc{
namespace codegen{

#define stream \
    genIndent(); \
    stream_

template <typename Dtype>
void Codegen<Dtype>::destroy(){
    graph_ = nullptr;
    names_map_.clear();
    tensors_malloc_map_.clear();
}

template <typename Dtype>
void Codegen<Dtype>::codeGenInit(){
    names_map_.clear();
    tensors_malloc_map_.clear();
    indent_ = 0;
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
            << "#include <math.h>\n";

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
    // not topology-ordered
    // for(int i=0; i<graph_->opNodeNum(); i++){
    //     genKernel(graph_->getOpNode(i));
    // }

    for (int i = 0; i < graph_->topologyNum(); i++)  
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            if (graph_->getNodeInTopo(i, j)->nodeType() == OP_NODE){
                genFuncCall((OpNode<Dtype>*)graph_->getNodeInTopo(i, j));                 
            }
        }
}

template<typename Dtype>
void Codegen<Dtype>::genFuncCall(OpNode<Dtype>* op){
    //op->genKernel(stream); 
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
        TensorNode<Dtype>* A = (TensorNode<Dtype>*)op->getParentNode(0);  
        TensorNode<Dtype>* B = (TensorNode<Dtype>*)op->getParentNode(1);  
        TensorNode<Dtype>* C = (TensorNode<Dtype>*)op->getChildNode(0);  
        int m = C->getTensor()->getDim(0);
        int k = A->getTensor()->getDim(1);
        int n = C->getTensor()->getDim(1);

        stream << "matrixMatrixMul_"<< dtype_flag 
                << "(" << m << ", " << n << ", " << k << ", "
                << tensors_malloc_map_[A] << ", " << m << ", "
                << tensors_malloc_map_[B] << ", " << k << ", "
                << tensors_malloc_map_[C] << ", " << m << ");\n";    
        
    }
    if ((oplabel->getTypeNameLabel()).compare("MatrixTanh") == 0) {
        //TODO assert
        TensorNode<Dtype>* A = (TensorNode<Dtype>*)op->getParentNode(0);  
        TensorNode<Dtype>* B = (TensorNode<Dtype>*)op->getChildNode(0);  
        int m = A->getTensor()->getDim(0);
        int n = A->getTensor()->getDim(1);
        
        stream << "matrixTanh_"<< dtype_flag 
                << "(" << m << ", " << n << ", "
                << tensors_malloc_map_[A] << ", " << m << ", "
                << tensors_malloc_map_[B] << ", " << m << ");\n";    
        
    }      
}

// TODO depreciate this function 
template<typename Dtype>
void Codegen<Dtype>::genMemAllocs(){
    SWLOG_INFO << "genMemAllocs \n";
    for(int i=0 ; i< graph_->tensorNodeNum(); i++){
        TensorNode<Dtype>* tnode = graph_->getTensorNode(i);
        // generate malloc statment
        std::string bufferName = genTensorMemAlloc(tnode);

        Label *tlabel  = tnode->getLabel();

        //TODO init tensor if needed
    }
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

    tensors_malloc_map_[tnode] = bufferName;
    return bufferName;
}



template<typename Dtype>
void Codegen<Dtype>::genMemFree(){
    SWLOG_INFO << "genMemoryFree\n";

    auto iter  = tensors_malloc_map_.begin();
    while(iter != tensors_malloc_map_.end()){
        stream << "free(" << iter->second << ");\n";   

        iter++;     
    }
}


INSTANTIATE_CLASS(Codegen);

} //namespace codegen
} // namespace swc



