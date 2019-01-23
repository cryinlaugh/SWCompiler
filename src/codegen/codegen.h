/*************************************************************************
	> File Name: codegen.h
	> Author: wayne
	> Mail: singleon11@gmail.com 
	> Created Time: 二  1/22 10:18:36 2019
 ************************************************************************/
#ifndef _CODEGEN_H_
#define _CODEGEN_H_

#include <unordered_map>
#include <sstream>

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
    void genFuncCall(OpNode<Dtype>* op);

    // generate malloc for tensor data
    void genMemAllocs();
    void genMemFree();

    std::string genTensorMemAlloc(TensorNode<Dtype>* tnode);
    
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
    std::unordered_map<std::string, int> names_map_;
    std::unordered_map<TensorNode<Dtype>*, std::string> tensors_malloc_map_;
};

} // namespace codegen
} // namespace swc

#endif
