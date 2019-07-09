/*************************************************************************
	> File Name: parallelGen.cpp
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue Jul  9 07:20:55 2019
 ************************************************************************/

#include "parallelGen.h"
#include <string>
#include <map>
#include <vector>
#include <assert.h>
#include "op/Op.h"

using namespace std;

vector<vector<int> > swc::ParallelGen::generateStgy (swc::op::Op* testOp) {
    assert( (testOp->_einOp == 1) && "Not a parallelizable Op!");

    std::map<char, int> comDim;
    for(int iterT = 0; iterT < (testOp->_nInputTensor+testOp->_nOutputTensor); iterT ++){
        for(int dimIdx = 0; dimIdx < testOp->_einRep[iterT].size(); dimIdx ++){
            if (comDim.find(testOp->_einRep[iterT][dimIdx]) == comDim.end()){
                comDim[testOp->_einRep[iterT][dimIdx]] = 1;
            }else{
                comDim[testOp->_einRep[iterT][dimIdx]] = comDim[testOp->_einRep[iterT][dimIdx]] + 1;
            }
        }
    }

    map<char,int>::iterator it;
    it = comDim.begin();
    vector<vector<int> > strategies;
    while (it != comDim.end()) {
        if (!(it->second > 1)) {
            it++;
            continue;
        }
        vector<int> stgy;
        for (int iterT = 0; iterT < (testOp->_nInputTensor); iterT++){
            int dimIdx=0;
            for(dimIdx = 0; dimIdx < testOp->_einRep[iterT].size(); dimIdx ++){
                if( testOp->_einRep[iterT][dimIdx] == it->first) break;
            }
            if (dimIdx < testOp->_einRep[iterT].size()) stgy.push_back(dimIdx);
            else stgy.push_back(-1);
        }
        for (int iterT = testOp->_nInputTensor; iterT < (testOp->_nInputTensor+testOp->_nOutputTensor); iterT++){
            int dimIdx=0;
            for(dimIdx = 0; dimIdx < testOp->_einRep[iterT].size(); dimIdx ++){
                if( testOp->_einRep[iterT][dimIdx] == it->first) break;
            }
            if (dimIdx < testOp->_einRep[iterT].size()) stgy.push_back(dimIdx);
            else stgy.push_back(-2);
        }
        strategies.push_back(stgy);

        it++;
    }
    return strategies;
}
