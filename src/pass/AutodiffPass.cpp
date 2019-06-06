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
}

void AutodiffPass::getSGDParameters()
{
    ((SGD_PARAMETERS*)_parameters)->lr = 0.01;
}
void AutodiffPass::getSGDParameters(double lr)
{
    ((SGD_PARAMETERS*)_parameters)->lr = lr;
}

void AutodiffPass::getADAMParameters()
{
    ((ADAM_PARAMETERS*)_parameters)->lr = 0.01;
}
void AutodiffPass::getADAMParameters(double lr)
{
    ((ADAM_PARAMETERS*)_parameters)->lr = lr;
}

void AutodiffPass::run()
{
    SWLOG_INFO << "autodiff-run" << endl;
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

