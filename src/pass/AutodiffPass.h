/*
 * AutodiffPass.h
 * Copyright (C) 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef AUTODIFFPASS_H
#define AUTODIFFPASS_H

#include "SWLOG.h"

namespace swc {

// Forward declarations
class IRGraph;

namespace pass {

#define SGD_METHOD 0
#define ADAM_METHOD 1

//define sgd parameters
typedef struct
{
    double lr;
} SGD_PARAMETERS;

//define adam parameters
typedef struct
{
    double lr;
} ADAM_PARAMETERS;

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
    ~AutodiffPass(){ destroy(); };

    int string2method(char* s);
    
    //Method determination
    void getMethods();
    
    template <typename T, typename... Types>
    void getMethods(T &firstArg, const Types &... args) {
        if(string2method((char*)firstArg) == SGD_METHOD) {
            SWLOG_INFO<<"Detect SGD method..."<<std::endl;
            _method = SGD_METHOD;
            _parameters = (SGD_PARAMETERS*)malloc(sizeof(SGD_PARAMETERS));
            getSGDParameters(args...);
        } 
        else if(string2method((char*)firstArg) == ADAM_METHOD) {
            SWLOG_INFO<<"Detect ADAM method..."<<std::endl;
            _method = ADAM_METHOD;
            _parameters = (ADAM_PARAMETERS*)malloc(sizeof(ADAM_PARAMETERS));
            getADAMParameters(args...);
        } else {
            SWLOG_INFO<<"Unidentified method..."<<std::endl;
            abort();
        }

    };


    // SGD Parameters read-in
    void getSGDParameters();
    void getSGDParameters(double lr);

    // ADAM Parameters read-in
    void getADAMParameters();
    void getADAMParameters(double lr);
    
    
    void show();

    void run();
    void destroy();

};

}  //namespace pass 
}  //namespace swc 



#endif /* !AUTODIFFPASS_H */
