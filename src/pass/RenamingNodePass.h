/*************************************************************************
	> File Name: RenamingNodePass.h
	> Author: wayne
	> Mail:  
	> Created Time: 一  4/ 8 16:32:26 2019
 ************************************************************************/
#ifndef _RENAMINGNODEPASS_H_
#define _RENAMINGNODEPASS_H_

#include "OptimizePass.h"
#include <unordered_map>

namespace swc {

class UniqueName{
    std::unordered_map<std::string, int> names_map_;
public:
    std::string operator()(const std::string  &inputName) { 
        std::cout << "add " << inputName << "\n";
        assert(!inputName.empty() && "inputName empty");
        std::string name;
        for(const char c : inputName){
            if(c=='/' || c=='.' || c=='-')
                name.push_back('_');
            else
                name.push_back(c);
        }

        auto iter  = names_map_.find(name);
        if(iter != names_map_.end()){
            std::string uname = name;
            while(names_map_.count(uname) != 0){
                std::ostringstream os;
                os << name << (++iter->second);
                uname = os.str();
            }
            name = uname;
        }
        std::cout << "get " << name << "\n\n";
        names_map_[name] = 0;
        return name; 
    }
};

template<typename Dtype>
class RenamingNodePass: public OptimizePass<Dtype> {
    using OptimizePass<Dtype>::_graph;
    UniqueName uniqueName;
public:
    RenamingNodePass(IRGraph<Dtype> * graph): OptimizePass<Dtype>(graph) {};
    ~RenamingNodePass(){} 

    void run();        
};

template<typename Dtype>
void RenamingNodePass<Dtype>::run(){
    int nTensorNodes = _graph->tensorNodeNum();
    int nOpNodes = _graph->opNodeNum();
    for (int i = 0; i < nTensorNodes; i++) {
        TensorNode<Dtype>* node = _graph->getTensorNode(i);

        std::cout << "tensor ";
        std::string uname = uniqueName(node->name());
        node->setName(uname);

    }

    for (int i = 0; i < nOpNodes; i++) {
        OpNode<Dtype>* node = _graph->getOpNode(i);
        std::cout << "tensor ";
        std::string uname = uniqueName(node->name());
        node->setName(uname);
    }

}

}
#endif
