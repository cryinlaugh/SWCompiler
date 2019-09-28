/*************************************************************************
    > File Name: SearchSpace.h
    > Author: wayne
    > Mail:
    > Created Time: 六  9/28 00:01:32 2019
 ************************************************************************/
#ifndef _SEARCHSPACE_H
#define _SEARCHSPACE_H

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <set>
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "graphIR/IRGraph.h"
#include "parallel/parallelGen.h"

using namespace std;

namespace swc{
class OpTiling{
public:
    OpTiling(OpNode * opNode){
        
        std::vector<std::vector<int> > originTilings= ParallelGen::generateStgy(opNode->getOp());
        //check legal 
        _tiling = originTilings;
    }
    ~OpTiling(){}
    OpNode* getOpNode(){
        return _opNode;
    }
    std::vector<int> getOpTiling(int index){
    
        return _tiling[index];
    }

    size_t getSize() { return _tiling.size(); }
private:
    std::vector<std::vector<int>> _tiling;
    OpNode * _opNode;
};


class StrategySearchSpace{
public:
    StrategySearchSpace(IRGraph * graph){
        _irgraph=graph;
    
    }
    ~StrategySearchSpace();
    
    
    OpNode* getOpNodeByIndex(int opIndex){
        return _OpTilings[opIndex]->getOpNode();
    }

    std::vector<int> getOpTilingByIndex(int opIndex, int tilingIndex){
       

        return _OpTilings[opIndex]->getOpTiling(tilingIndex);
    }

    std::vector<int> getGeneSpace() {
        std::vector<int> geneSpace;
        for(auto op_tiling : _OpTilings) {
            auto size = op_tiling->getSize(); 
            geneSpace.push_back(size);
        } 
        return geneSpace;
    }

    float getFitness(std::vector<int> identity) {
        //assert(genne.size()==_OpTilings->size() && "illegal gene"); 
        //return  getCommunicationCost(identity);
        float communicationCost=0.0;
        int opIndex = 0;
        for(auto op_strategy_idx : identity) {
            std::vector<int> opStrategy = getOpTilingByIndex(opIndex, op_strategy_idx);
            OpNode* opNode = getOpNodeByIndex(opIndex);
            //Performance opNode
            communicationCost+=getCommunicationCost(opNode, opStrategy);
            opIndex++;
        }
        return communicationCost;
    }
    float getCommunicationCost(OpNode * opNode, std::vector<int> opStrategy){
        
        //std::map<TensorNode*, int> candidate;
        float  communicateCost =0.0;    
        for(unsigned long i=0;i<opNode->getParentNodes().size();i++){
            int curTiling = opStrategy[i];
            
            TensorNode * curTensorNode = dynamic_cast<TensorNode*>(opNode->getParentNode(i)); 
            

            //_tensorNodeMap.find(curTensorNode);
            std::map<TensorNode* ,std::set<int>>::iterator iter =  _tensorNodeMap.find(curTensorNode); 
            if(iter!=_tensorNodeMap.end()){
                std::set<int>  preTilings= iter->second;    
               if(preTilings.find(curTiling)!=preTilings.end()){
                
                   communicateCost += 0.0;
               }else{
                   //find best
                    float best;
                    for(std::set<int>::iterator it=preTilings.begin() ;it!=preTilings.end();it++)
                    {
                        best=0.0;
                        //cout<<*it<<" occurs "<<endl;
                    }
                    communicateCost+=best;
               } 
                
                
            }else{
                communicateCost+=1.0;//getfork cost
                std::set<int> preTilings;
                preTilings.insert(curTiling);
                _tensorNodeMap[curTensorNode]=preTilings;   
            }       
        }       
        return communicateCost;
    }
 
private:
    IRGraph * _irgraph;
    std::vector<OpTiling*> _OpTilings;
    std::map<TensorNode*,std::set<int>> _tensorNodeMap;
};
           
 
class GeneticSearch{
private:
    mutable std::mt19937_64 rng{random_device{}()};
    std::vector<int> _geneSpace;    
    using Population = std::vector<std::vector<int>>;
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
        return _sss->getFitness(identity); 
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
            _population.push_back(identity); 
            idx++;
        }

        for(; idx<_populationSize; idx++) {
            auto identity = randomIdentity(); 
            while(!isValid(identity)) {
                identity = randomIdentity();
            }
            _population.push_back(identity);
        }

    }

    void run() {

        for(size_t i=0; i<_numGenerations; i++) {
            breed();
            if(_numGenerations<100 || i%10==0) {
                std::cout << "generation" << i << " top5\n";
                printTopKIdentity(5);
            }
        }
    }

    void printTopKIdentity(size_t k) {
        // _population should be ordered
        for(size_t i=0; i<k; i++) {
            auto &identity = _population.at(i);
            for(auto gene : identity)
                std::cout << gene << " ";
            std::cout << "\n";
        }
    }
};

std::vector<double> GeneticSearch::getNormAccumFitness() {
    // _population must be desc-ordered by fitness before call this function
    std::vector<double> fit(_populationSize);
    for(size_t i=0; i<_populationSize; i++) {
        auto identity = _population[i];
        fit[i] = getFitness(identity);
    }

    // normalize fitness 
    auto sum = std::accumulate(fit.begin(), fit.end(), 0.0);
    for(size_t i=0; i<_populationSize; i++) {
        fit[i] /= sum;
    }

    std::vector<double> accum_fit;
    // fit[0]
    // fit[0] + fit[1]
    // ...
    // 1.0
    std::partial_sum(fit.begin(), fit.end(), std::back_inserter(accum_fit));
    return accum_fit;
}


std::vector<int> GeneticSearch::crossover(std::vector<int>& p0, std::vector<int>& p1) {
    vector<int> child(p0.size());
    // ! closed interval [a, b]
    std::uniform_int_distribution<size_t> dist(0,1);
    for(size_t i=0; i<p1.size(); i++) {
        size_t idx = dist(rng); 
        child[i] = idx==0 ? p0.at(i) :p1.at(i);
    }

    return child;
}

void GeneticSearch::mutate(std::vector<int> & identity) {
    auto dist = std::discrete_distribution<int>{(1 - _mutationRate), _mutationRate};

    for(size_t idx=0; idx<identity.size(); idx++) {
        // decide if should mutate for each gene piece (!!! not identity)
        // so mutattionRate should be small
        bool shouldMutate = dist(rng);
        if(shouldMutate) {
            randomGene(identity, idx);
        }
    }
}

void GeneticSearch::breed() {
    std::sort(
      _population.begin(),
      _population.end(),
      [this](const std::vector<int>& a,
         const std::vector<int>& b) {
        
        return this->getFitness(a) > this->getFitness(b);
    });

    // elites directly enter next generation
    Population children(_population.begin(), _population.begin()+_numberElites);

    auto accum_fitness = getNormAccumFitness();

    // default uniform distribution between [0, 1)
    auto uniform_dist = std::uniform_real_distribution<double>{};
    // generate 0 or 1 with their weight as possibility
    // 0 means not crossover, 1 means shoudlCrossover
    auto disc_dist = std::discrete_distribution<int>{(1 - _crossOverRate), _crossOverRate};
    
    // select parents to crossover -> child -> mutate 
    // roulette method (descend-ordered Population, accumulated fitness, nomal distribution [0,1)
    while(children.size() < _populationSize) {
        auto limit = uniform_dist(rng);
        auto lb1 = std::lower_bound(accum_fitness.begin(), accum_fitness.end(), limit);
        limit = uniform_dist(rng);
        auto lb2 = std::lower_bound(accum_fitness.begin(), accum_fitness.end(), limit);

        if(lb1 == lb2)
            continue;

        auto p1 = _population.at(std::distance(accum_fitness.begin(), lb1));
        auto p2 = _population.at(std::distance(accum_fitness.begin(), lb2));

        bool shouldCross = disc_dist(rng);
        if(shouldCross) {

            auto child = crossover(p1, p2);

            mutate(child);
            
            children.push_back(child);
        }
    }
    _population = children;
}

class OpTiling{
public:
    OpTiling(OpNode * opNode){
        
        std::vector<std::vector<int> > originTilings= ParallelGen::generateStgy(opNode->getOp());
        //check legal 
        _tiling = originTilings;
    }
    ~OpTiling(){}
    OpNode* getOpNode(){
        return _opNode;
    }
    std::vector<int> getOpTiling(int index){
    
        return _tiling[index];
    }

    size_t getSize() { return _tiling.size(); }
private:
    std::vector<std::vector<int>> _tiling;
    OpNode * _opNode;
};


class StrategySearchSpace{
public:
    StrategySearchSpace(IRGraph * graph){
        _irgraph=graph;
    
    }
    ~StrategySearchSpace();
    
    
    OpNode* getOpNodeByIndex(int opIndex){
        return _OpTilings[opIndex]->getOpNode();
    }

    std::vector<int> getOpTilingByIndex(int opIndex, int tilingIndex){
       

        return _OpTilings[opIndex]->getOpTiling(tilingIndex);
    }

    std::vector<int> getGeneSpace() {
        std::vector<int> geneSpace;
        for(auto op_tiling : _OpTilings) {
            auto size = op_tiling->getSize(); 
            geneSpace.push_back(size);
        } 
        return geneSpace;
    }

    float getFitness(std::vector<int> identity) {
        //assert(genne.size()==_OpTilings->size() && "illegal gene"); 
        //return  getCommunicationCost(identity);
        float communicationCost=0.0;
        int opIndex = 0;
        for(auto op_strategy_idx : identity) {
            std::vector<int> opStrategy = getOpTilingByIndex(opIndex, op_strategy_idx);
            OpNode* opNode = getOpNodeByIndex(opIndex);
            //Performance opNode
            communicationCost+=getCommunicationCost(opNode, opStrategy);
            opIndex++;
        }
        return communicationCost;
    }
    float getCommunicationCost(OpNode * opNode, std::vector<int> opStrategy){
        
        //std::map<TensorNode*, int> candidate;
        float  communicateCost =0.0;    
        for(unsigned long i=0;i<opNode->getParentNodes().size();i++){
            int curTiling = opStrategy[i];
            
            TensorNode * curTensorNode = dynamic_cast<TensorNode*>(opNode->getParentNode(i)); 
            

            //_tensorNodeMap.find(curTensorNode);
            std::map<TensorNode* ,std::set<int>>::iterator iter =  _tensorNodeMap.find(curTensorNode); 
            if(iter!=_tensorNodeMap.end()){
                std::set<int>  preTilings= iter->second;    
               if(preTilings.find(curTiling)!=preTilings.end()){
                
                   communicateCost += 0.0;
               }else{
                   //find best
                    float best;
                    for(std::set<int>::iterator it=preTilings.begin() ;it!=preTilings.end();it++)
                    {
                        best=0.0;
                        //find smallest                       //cout<<*it<<" occurs "<<endl;
                    }
                    communicateCost+=best;
               } 
                
                
            }else{
                communicateCost+=1.0;//getfork cost
                std::set<int> preTilings;
                preTilings.insert(curTiling);
                _tensorNodeMap[curTensorNode]=preTilings;   
            }       
        }       
        return communicateCost;
    }
 
private:
IRGraph * _irgraph;
    std::vector<OpTiling*> _OpTilings;
    std::map<TensorNode*,std::set<int>> _tensorNodeMap;
    
};
           
    



}
#endif
