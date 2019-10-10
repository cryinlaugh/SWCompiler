/*************************************************************************
    > File Name: SearchSpace.h
    > Author: wayne
    > Mail:
    > Created Time: 六  9/28 00:01:32 2019
 ************************************************************************/
#ifndef _SEARCHSPACE_H
#define _SEARCHSPACE_H

#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>
#include <random>
#include <cmath>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <iomanip>
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "graphIR/IRGraph.h"
#include "parallel/parallelGen.h"
#include "parallel/TilingLabel.h"
#include "pass/ParallelLoweringPass.h"
#include "pass/EliminationPass.h"
#include "op/dlOp/dlOp.h"

using namespace std;

namespace swc{
class OpStrategy{
public:
    OpStrategy(OpNode * opNode, int p){
        
        _opNode = opNode;
        std::vector<std::vector<int> > strategies = ParallelGen::generateStgy(opNode);

        //check legal 
        // channel % degree may not be zero
        // finalstrategy = strategies[0];
        int nInputs = opNode->parentNum();
        int nOutputs = opNode->childNum();
        for(auto strategy : strategies) {
            bool legal = true;
            int idx = 0;
            for(auto tensor_dim: strategy) {
                
                Tensor* tensor;
                if(idx < nInputs) {
                    tensor = ((TensorNode*)opNode->getParentNode(idx))->getTensor();
                } else if(idx < (nInputs+nOutputs)) {
                    tensor = ((TensorNode*)opNode->getChildNode(idx-nInputs))->getTensor();
                } else {
                    legal = false;
                    break;
                }

                if(tensor_dim >= 0) {
                    if(tensor->getDim(tensor_dim) % p) {
                        legal = false;
                        break;
                    }
                }
                idx++;
            } // for parallel dim in this strategy
            
            if(legal)
                _op_strategies.push_back(strategy);
        }

        std::ostringstream oss;
        std::ostream *os = &std::cout; 
        *os << "-----" << opNode->name() << " legal strategies------\n";
        for(auto sgy : _op_strategies){
            for(auto s: sgy)
                *os << s <<" ";
            *os<<"\n";
        }
        
    }

    ~OpStrategy(){}
    OpNode* getOpNode(){
        return _opNode;
    }

    std::vector<int> getOpStrategy(int index){
    
        return _op_strategies.at(index);
    }

    size_t getSize() { return _op_strategies.size(); }

private:
    std::vector<std::vector<int>> _op_strategies;
    OpNode * _opNode;
};


class StrategySearchSpace{
public:
    StrategySearchSpace(IRGraph * graph){
        _irgraph=graph;
        _p = _irgraph->getConfig().mpi_size;
    
    }
    ~StrategySearchSpace();
    

    size_t getOpNum() { return _graph_strategies.size(); }
    void addOpStrategyIfExist(OpNode *opnode) {
        OpStrategy *op_strategy = new OpStrategy(opnode, _p);           
        if(op_strategy->getSize() > 0) {
            SWLOG_DEBUG(8) << opnode->name() << " get legal strategies "
                << op_strategy->getSize() << "\n";
           _graph_strategies.push_back(op_strategy); 
        }
    }

    void printStrategySpace() {
        SWLOG_DEBUG(8) << "print StrategySearchSpace\n";
        double total = 1;
        for(auto &op_strategy : _graph_strategies) {
            auto opnode = op_strategy->getOpNode();
            auto size = op_strategy->getSize();
            std::cout << opnode->name() << " : " << size << "\n";
            total *= size;
        }
        std::cout << "[Summary of Graph Strategy Space]" << "nodes: " << 
            _graph_strategies.size() << " space= " << total << "\n";
    }
    
    OpNode* getOpNodeByIndex(int opIndex){
        return _graph_strategies.at(opIndex)->getOpNode();
    }

    std::vector<int> getOpStrategyByIndex(int opIndex, int tilingIndex){
        //std::cout << _graph_strategies.at(opIndex)->getOpNode()->name()<< " " << tilingIndex << "\n";
        return _graph_strategies.at(opIndex)->getOpStrategy(tilingIndex);
    }

    std::vector<int> getGeneSpace() {
        std::vector<int> geneSpace;
        for(auto _op_strategies: _graph_strategies) {
            auto size = _op_strategies->getSize(); 
            geneSpace.push_back(size);
        } 
        return geneSpace;
    }

    float getFitness(std::vector<int> identity) {
        //assert(genne.size()==_OpStrategys->size() && "illegal gene"); 
        //return  getCommunicationCost(identity);
        float communicationCost=0.0;
        int opIndex = 0;
        for(auto op_strategy_idx : identity) {
            std::vector<int> opStrategy = getOpStrategyByIndex(opIndex, op_strategy_idx);
            OpNode* opNode = getOpNodeByIndex(opIndex);
            //Performance opNode
            communicationCost+=getCommunicationCost(opNode, opStrategy);
            opIndex++;
        }
        return communicationCost;
    }

    float getFitnessByGraphTransform(std::vector<int> identity) {
        SWLOG_DEBUG(2) << "StrategySearchSpace getFitnessByGraphTransform begin\n";
        float communicationCost=0.0;

        IRGraph *graph = _irgraph->clone(); 
        //因为还没添加graph->destroy()方法，实际上指针还会指向同样的起始地址
        //而相应的tiling label还没有释放
        // std::cout << "copied graph address " << graph << std::endl;

        SWLOG_DEBUG(2) << "StrategySearchSpace getFitnessByGraphTransform labeling copied graph\n";
        int opIndex = 0;
        for(auto op_strategy_idx : identity) {
            std::vector<int> opStrategy = getOpStrategyByIndex(opIndex, op_strategy_idx);
            OpNode* opNode = getOpNodeByIndex(opIndex);

            auto *copied_opnode = (OpNode*)graph->getNodeByName(opNode->name());
            assert(copied_opnode && "cannot find opnode with same name in cloned graph");
            copied_opnode->setStrategyLabel(new StrategyLabel(opStrategy));

            opIndex++;
        }

        SWLOG_DEBUG(2) << "StrategySearchSpace getFitnessByGraphTransform begin ParallelLoweringPass\n";
        pass::ParallelLoweringPass *par_lowering_pass = new pass::ParallelLoweringPass(graph);
        par_lowering_pass->run();
        pass::EliminationPass *elimpass = new pass::EliminationPass(graph);
        elimpass->run();

        SWLOG_DEBUG(2) << "StrategySearchSpace getFitnessByGraphTransform getComm\n";
        communicationCost = graph->getCommCost();

        // std::cout << "\n" << graph->getCommTrace() << "\n";
        // std::cout << graph->getCommCost() << "\n";

        delete graph; 

        return communicationCost;
    }

    void addStrategyToGraph(std::vector<int> identity) {
        std::cout << "----------selected strategy by op out---------------------\n";
        int opIndex = 0;
        for(auto op_strategy_idx : identity) {
            std::vector<int> opStrategy = getOpStrategyByIndex(opIndex, op_strategy_idx);
            OpNode* opNode = getOpNodeByIndex(opIndex);

            opNode->setStrategyLabel(new StrategyLabel(opStrategy));

            opIndex++;

            std::cout << std::left << std::setw(3) << opIndex << " " 
                << std::left << std::setw(3) 
                << opStrategy.at(opNode->parentNum()) << " " 
                << std::left << std::setw(15) 
                << opNode->getChildNode(0)->name() << " " 
                << std::left << std::setw(15) 
                << opNode->name() << " children " << opNode->childNum() << "\n";
        }

    }

    float getCommunicationCost(OpNode * opNode, std::vector<int> opStrategy){
        float  communicateCost =0.0;
        auto config = _irgraph->getConfig();

        for(unsigned long i=0;i<opNode->getParentNodes().size();i++){
            int curTiling = opStrategy[i];          
            TensorNode * curTensorNode = dynamic_cast<TensorNode*>(opNode->getParentNode(i)); 
            std::map<TensorNode* ,std::set<int>>::iterator iter =  _inTensorStrategiesMap.find(curTensorNode); 
            if(iter!=_inTensorStrategiesMap.end()){
                std::set<int>  preTilings= iter->second;    
               if(preTilings.find(curTiling)!=preTilings.end()){//find the same tiling as curtiling in preTilings    
                   communicateCost += 0.0;
                   _inTensorStrategiesMap[curTensorNode].insert(curTiling);

               }else{
                   int smallestTiling = *std::min_element(preTilings.begin(),preTilings.end());
                   //we think the smallest communicatecost comes from the smallest tiling number 
                    communicateCost+=TransformOp::getSimCost(curTensorNode->getTensor()->getSizeInBytes(), config, smallestTiling,curTiling);
               } 
            }else{
                communicateCost+=ScatterOp::getSimCost(curTensorNode->getTensor()->getSizeInBytes(), config, curTiling);
                std::set<int> preTilings;
                preTilings.insert(curTiling);
                _inTensorStrategiesMap[curTensorNode]=preTilings;

            }       
        } 

        
        for(unsigned long i=0;i<opNode->getChildNodes().size();i++){
            int curTiling = opStrategy[i];     
            TensorNode * curTensorNode = dynamic_cast<TensorNode*>(opNode->getParentNode(i)); 
            std::map<TensorNode* ,std::set<int>>::iterator iter =  _outTensorStrategiesMap.find(curTensorNode);
                
            //TBC
            (void)curTiling;
            (void)iter;

        }    

        return communicateCost;
}
 
private:
    IRGraph * _irgraph;
    int _p; // parallel size = _irgraph->getConfig().mpi_size();
    std::vector<OpStrategy*> _graph_strategies;

    std::map<TensorNode*,std::set<int>> _inTensorStrategiesMap;
    std::map<TensorNode*,std::set<int>> _outTensorStrategiesMap;
      
};
           
 
class GeneticSearch{
private:
    mutable std::mt19937_64 rng{random_device{}()};
    std::vector<int> _geneSpace;    
    using IdentityWithFit = std::pair<std::vector<int>, float>;
    //using Population = std::vector<std::vector<int>>;
    using Population = std::vector<IdentityWithFit>;
    Population _population;
    size_t _populationSize;
    double _crossOverRate;
    double _mutationRate;
    size_t _numberElites;
    size_t _numGenerations;

    StrategySearchSpace* _sss;

    std::vector<int> randomIdentity() {
        // number of gene per identity
        size_t num = _geneSpace.size();
        vector<int> identity(num);
        for(size_t i=0; i<num; i++) {
            randomGene(identity, i);
        }

        return identity;
    }

    void randomGene(std::vector<int> &identity, int idx) {
        // closed interval [0, geneSpace-1]
        std::uniform_int_distribution<size_t> dist(0, _geneSpace[idx]-1);
        identity.at(idx) = dist(rng);
    }

    std::vector<double> getNormAccumFitness();

    bool isValid(std::vector<int> &identity) {
        for(size_t idx=0; idx<identity.size(); idx++) {
            auto gene = identity.at(idx);
            if(gene > _geneSpace.at(idx) || gene <0)
                return false;
        }
        return true;
    }

    std::vector<int> crossover(std::vector<int>& p1, std::vector<int>& p2);
    void mutate(std::vector<int>& identity);
    void breed();

    double getFitness(const std::vector<int>& identity) {
        // return _sss->getFitness(identity); 
        std::ostringstream os;
        for(auto s : identity)
            os << s << " ";
        SWLOG_DEBUG(2) << "[GA getFitness] " << os.str() << "\n"; 
        return _sss->getFitnessByGraphTransform(identity); 
    }
    
public:
    GeneticSearch(std::vector<int> geneSpace, 
        std::vector<std::vector<int>> &identities,
        size_t populationSize,
        double crossOverRate,
        double mutationRate,
        size_t numberElites,
        size_t numGenerations,
        StrategySearchSpace* sss):
        _geneSpace(geneSpace),
        _populationSize(populationSize),
        _crossOverRate(crossOverRate),
        _mutationRate(mutationRate),
        _numberElites(numberElites),
        _numGenerations(numGenerations),
        _sss(sss)
    {
        _population.reserve(_populationSize);
        assert(identities.size() < _populationSize && "init identities num > populationSize"); 
        size_t idx = 0;
        for(auto identity : identities) {
            if(identity.size() != geneSpace.size())
                continue;

            _population.push_back(std::make_pair(identity, getFitness(identity))); 
            idx++;
        }

        for(; idx<_populationSize; idx++) {
            auto identity = randomIdentity(); 
            while(!isValid(identity)) {
                identity = randomIdentity();
            }
            _population.push_back(std::make_pair(identity, getFitness(identity)));
        }

        size_t top = populationSize < 5 ? populationSize : 5;
        std::cout << "generation0" << " top" << top << " of " << populationSize << "\n";
        printTopKIdentity(top);

    }

    void run() {
        size_t top = _populationSize < 5 ? _populationSize : 5;

        for(size_t i=0; i<_numGenerations; i++) {
            breed();

            if(_numGenerations<500 || i%10==0) {
                std::cout << "generation" << i << " top" << top << " of " << _populationSize << "\n";
                printTopKIdentity(top);
            }
        }
    }

    std::vector<int> getBestIdentity() {
        return _population.at(0).first;
    }

    void printTopKIdentity(size_t k) {
        // _population should be ordered
        for(size_t i=0; i<k; i++) {
            auto &identity = _population.at(i).first;
            for(auto gene : identity)
                std::cout << gene << " ";
            std::cout << "(" << (size_t)_population.at(i).second << ")\n";
        }
    }
};

}
#endif
