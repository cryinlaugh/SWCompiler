/***********************************************
#
#      Filename: src/pass/subgraphpass.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-07-31 15:11:36
# Last Modified: 2019-07-31 15:11:36
***********************************************/
#include "SWLOG.h"
#include "OptimizePass.h"
#include "graphIR/IRGraph.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "op/dlOp/dlOp.h"
#include "parallel/TilingLabel.h"
#include <queue>
#include <unordered_set>
#include <map>
#include "tool/dotGen.h"
namespace swc {

namespace pass {
class SubGraphPass;
}

class swc::pass::SubGraphPass : public swc::pass::OptimizePass {

    using OptimizePass::_graph;
private:

    std::map<IRNode*, int> _visit;
    std::unordered_set<IRNode *> _curComponent;
    //std::vector<std::vector<IRNode *> > _component;
    //int count;
public:
    SubGraphPass(IRGraph * graph ): OptimizePass(graph) {};
    ~SubGraphPass() {
    };

    void DFS(IRNode* irnode) {
        _visit[irnode] = 1;
        _curComponent.insert(irnode);
        for(auto child : irnode->getChildNodes()) {
            if(_visit[child] == 0)
                DFS(child);
        }
        for(auto parent : irnode->getParentNodes()) {
            if(_visit[parent] == 0)
                DFS(parent);
        }

    }

    void run() {

        genSubgraph(2);

    }
    void genSubgraph(int num) {
        //std::vector<IRNode * > parallelNodes;

        std::unordered_set<IRNode *> parallelNodes;

        for (int i = 0; i < _graph->topologyNum(); i++) {
            for (int j = 0; j < _graph->getNumInTopoLevel(i); j++) {
                IRNode * irnode = _graph->getNodeInTopo(i, j);
                parallelNodes.insert(irnode);
                _visit[irnode] = 0;


            }
        }
        int count = 0;

        std::vector<Device> devices;
        for(int k = 0; k < num; ++k) {
            //devices.push_back()
            Device device;
            device.id = k+1;
            devices.push_back(device);
        }

        for(std::unordered_set<IRNode *>::iterator i = parallelNodes.begin(); i != parallelNodes.end(); i++) {
            if(_visit[*i] == 0) {
                DFS(*i);
                count ++;


                std::vector<IRGraph* > subGs;
                std::vector<OpNode* > subGNodes;
                for(int k = 0; k < num; ++k) {
                    IRGraph * subG = new IRGraph();
                    subGs.push_back(subG);
                    SubGraphOp * subG_Op = new SubGraphOp();
                    subG_Op->setGraph(subG);
                    OpNode * subGNode = new OpNode("subG", subG_Op);
                    subGNodes.push_back(subGNode);
                    //subG->setDeviceLabel(devices[k]);
                }

                std::cout << "Nodes in curComponent: " << _curComponent.size() << std::endl;
                for(std::unordered_set<IRNode *>::iterator j = _curComponent.begin(); j != _curComponent.end(); j++) {
                    if ((*j)->nodeType() == OP_NODE) {
                        auto *node = (OpNode *)(*j);
                        for(int k = 0; k < num; ++k) {
                            subGs[k]->pushOpNode(node);
                        }
                        _graph->delOpNode(node);

                        //std::cout << "OpNode :" << node->name() << std::endl;

                    } else if ((*j)->nodeType() == TENSOR_NODE) {
                        auto *node = (TensorNode*)(*j);
                        //subG->pushTensorNode(node);


                        TilingLabel * tlabel = node->getTilingLabel();
                        if(tlabel->isApplied() == 1) {
                            TensorNode *node_mirror = node->clone();
                            node_mirror->setExternal(true);
                            for(int k = 0; k < num; ++k) {
                                subGs[k]->pushTensorNode(node_mirror);
                            }
                            for (auto c : node->getChildNodes()) {
                                if (_curComponent.count(c)) {
                                    c->destroyUpperNode(node);
                                    c->exlinkUpperNode(node_mirror);
                                    //std::cout<<"test:"<<c->name()<<","<<node_mirror->name()<<std::endl;

                                }
                            }

                            //std::cout << "in:" << node->name() << std::endl;
                            for(int k = 0; k < num; k++) {
                                subGNodes[k] -> exlinkUpperNode(node);
                            }
                        } else if(tlabel->isApplied() == 2) {

                            TensorNode *node_mirror = node->clone();
                            node_mirror->setExternal(true);
                            for(int k = 0; k < num; ++k) {
                                subGs[k]->pushTensorNode(node_mirror);
                            }
                            for(auto p : node->getParentNodes()) {
                                if(_curComponent.count(p)) {
                                    node->destroyUpperNode(p);
                                    node_mirror->exlinkUpperNode(p);
                                    //std::cout<<"test:"<<node_mirror->nodeType()<<std::endl;
                                    //c->destroyUpperNode(node);
                                    //c->exlinkUpperNode(node_mirror);
                                }
                            }
                            //std::cout << "test:" << node->name() <<"-"<<subGNode->name()<< std::endl;
                            for(int k = 0; k < num; ++k) {
                                node->exlinkUpperNode(subGNodes[k]);
                            }
                        } else {
                            // std::cout << "del:" << node->name() << std::endl;
                            for(int k = 0; k < num; ++k) {
                                subGs[k]->pushTensorNode(node);
                            }
                            _graph->delTensorNode(node);
                        }
                    }

                }
                for(int k = 0; k < num; k++) {
                    _graph->pushOpNode(subGNodes[k]);
                    subGs[k]->findInOut();
                    subGs[k]->updateTopology();
                
                    subGs[k]->setDeviceLabel(devices[k]);
                    dotGen(subGs[k], "test.dot");
                }
                _graph->findInOut();
                _graph->updateTopology();
                std::cout << "ComponentNum:" << count << std::endl;
            }

            //std::cout << "ComponentNum:" << count << std::endl;

        }
    }

};

}  //namespace pass  //namespace swc
