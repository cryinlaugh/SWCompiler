/*************************************************************************
	> File Name: parallelGen.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue Jul  9 07:15:52 2019
 ************************************************************************/

#ifndef _PARALLELGEN_H
#define _PARALLELGEN_H

#include <string>
#include <map>
#include <vector>
#include <assert.h>
#include "op/Op.h"
namespace swc{

    class ParallelGen{

        public:

        std::vector<std::vector<int> > generateStgy(op::Op* testOp);
    };

}

#endif
