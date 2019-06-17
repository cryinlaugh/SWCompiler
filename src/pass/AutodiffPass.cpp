/*
 * AutodiffPass.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-05-23
 */

#include "AutodiffPass.h"

#include <stdlib.h>
#include <string>

#include "SWLOG.h"
#include "graphIR/IRGraph.h"

using namespace std;

namespace swc {
namespace pass {

METHOD_TYPE AutodiffPass::string2method(std::string& s)
{
    for (unsigned long i = 0; i < s.size(); i++)
       s[i] = tolower(s[i]);
    
    if (s == "sgd") return SGD_METHOD;
    if (s == "adam") return ADAM_METHOD;
    return -1;
}

void AutodiffPass::getMethods() 
{
    SWLOG_INFO<<"No method determinated..."<<std::endl;
    SWLOG_INFO<<"Please choose a solver: SGD, ADAM..."<<std::endl;
    abort();
}

void AutodiffPass::getSGDParameters()
{
    ((SGD_PARAMETERS*)_parameters)->lr = 0.001;
    ((SGD_PARAMETERS*)_parameters)->decay = 0.001;
    ((SGD_PARAMETERS*)_parameters)->momentum = 0.9;
}
void AutodiffPass::getSGDParameters(float lr,
                                    float decay,
                                    float momentum)
{
    ((SGD_PARAMETERS*)_parameters)->lr = lr;
    ((SGD_PARAMETERS*)_parameters)->decay = decay;
    ((SGD_PARAMETERS*)_parameters)->momentum = momentum;
}

void AutodiffPass::getADAMParameters()
{
    ((ADAM_PARAMETERS*)_parameters)->lr = 0.01;
}
void AutodiffPass::getADAMParameters(float lr)
{
    ((ADAM_PARAMETERS*)_parameters)->lr = lr;
}

void AutodiffPass::run(IRGraph* graph_train)
{
    SWLOG_INFO << "AutodiffPassi Run" << endl;
    graph_train->copyFrom(_graph);

    graph_train->updateTopology();

}

void AutodiffPass::destroy()
{
    if (_parameters != NULL) free(_parameters);
    SWLOG_INFO << "free AutodiffPass parameters" << endl;
}

void AutodiffPass::show()
{
    SWLOG_INFO << "Show Methods:" << endl;
    switch(_method)
    {
        case SGD_METHOD:
            SWLOG_INFO << "----SGD" << endl;
            SWLOG_INFO << "----learning rate:" 
                << ((SGD_PARAMETERS*)_parameters)->lr << endl;
            SWLOG_INFO<< "----decay:"
                << ((SGD_PARAMETERS*)_parameters)->decay << endl;
            SWLOG_INFO << "----momentum:"
                << ((SGD_PARAMETERS*)_parameters)->momentum << endl;
            break;
        case ADAM_METHOD:
            SWLOG_INFO << "----ADAM" << endl;
            SWLOG_INFO << "----learning rate:" 
                << ((ADAM_PARAMETERS*)_parameters)->lr << endl;
            break;
        default:
            break;
    }
}


} //namespace pass
} //namespace swc

