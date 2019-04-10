/*************************************************************************
	> File Name: RenamingNodePass.h
	> Author: wayne
	> Mail:  
	> Created Time: 一  4/ 8 16:32:26 2019
 ************************************************************************/
#ifndef _RENAMINGNODEPASS_H_
#define _RENAMINGNODEPASS_H_

#include "SWLOG.h"
#include "OptimizePass.h"

#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"

#include <unordered_map>
#include <sstream>

namespace swc {

class UniqueName{
    std::unordered_map<std::string, int> names_map_;
public:
    std::string operator()(const std::string  &inputName) { 
        SWLOG_INFO << "originalName " << inputName << "\n";
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
        SWLOG_INFO << "uniqueName " << name << "\n\n";
        names_map_[name] = 0;
        return name; 
    }
};


class RenamingNodePass: public OptimizePass {
    using OptimizePass::_graph;
    UniqueName uniqueName;
public:
    RenamingNodePass(IRGraph * graph): OptimizePass(graph) {};
    ~RenamingNodePass(){} 

    void run(){
        int nTensorNodes = _graph->tensorNodeNum();
        int nOpNodes = _graph->opNodeNum();
        for (int i = 0; i < nTensorNodes; i++) {
            TensorNode* node = _graph->getTensorNode(i);
            std::string uname = uniqueName(node->name());
            node->setName(uname);
        }

        for (int i = 0; i < nOpNodes; i++) {
            OpNode* node = _graph->getOpNode(i);
            std::string uname = uniqueName(node->name());
            node->setName(uname);
        }
}
};



}
#endif
