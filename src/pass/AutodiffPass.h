/*
 * AutodiffPass.h
 * Copyright (C) 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef AUTODIFFPASS_H
#define AUTODIFFPASS_H

#include "SWLOG.h"


// to handle string ignored-case-senstive compare 
inline int stricmp(char *s, char *t)
{
    if (s == NULL || t == NULL) {
        printf("compared strings are not exist");
        abort();
    }
    while (*s != '\0') {
        if((*s >= 'A') && (*s <= 'Z'))
            *s += 0x20;
        if((*t >= 'A') && (*t <= 'Z'))
            *t += 0x20;
        if (*s != *t)
            return 0;
    }
    if (*t != '\0') return 0;
    return 1;
}

namespace swc {

// Forward declarations
class IRGraph;
    
namespace pass {

#define SGD_METHOD 0
#define ADMA_METHOD 1

//define sgd parameters
typedef struct
{
    double _lr;
} SGD_PARAMETRS;


/**
 * @breif AutodiffPass to do the auto-differential of the
 * original net and generate a training network.
 */
class AutodiffPass {
  private:
    int _method;
    void* _parameters;
    IRGraph *_graph;


  public:
    AutodiffPass(IRGraph *graph){ _graph = graph; };
    ~AutodiffPass(){};

    //Method determination
    //default by SGD
    void getMethods(){
            
        SWLOG_INFO<<"Default SGD method..."<<std::endl;
        _method = SGD_METHOD;
        _parameters = (SGD_PARAMETRS*)malloc(sizeof(SGD_PARAMETRS));
    };
    template <typename T, typename... Types>
    void getMethods(T &firstArg, const Types &... args) {
        if(strcmp((char*)firstArg, "SGD") == 0)
        {
            
            SWLOG_INFO<<"Detect SGD method..."<<std::endl;
            _method = SGD_METHOD;
            _parameters = (SGD_PARAMETRS*)malloc(sizeof(SGD_PARAMETRS));
        }
    }

    // Parameters read-in
    void getParameters(){};
    template <typename T, typename... Types>
    void getParameters(const T &firsrArg, const Types &... args) {
    }

    void run();
};

}  //namespace pass 
}  //namespace swc 



#endif /* !AUTODIFFPASS_H */
