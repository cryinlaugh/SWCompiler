/*************************************************************************
	> File Name: src/parallel/SearchSpace.cpp
	> Author: wayne
	> Mail:  
	> Created Time: Wed 02 Oct 2019 04:00:25 AM UTC
 ************************************************************************/
#include "SearchSpace.h"

namespace swc{

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
        
        // when fitnees return comm, <
        // when fitnees return time, >
        return this->getFitness(a) < this->getFitness(b);
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

} // swc
