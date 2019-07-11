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

typedef int METHOD_TYPE;

//define sgd parameters
typedef struct
{
    float lr;
    float decay;
    float momentum;
    size_t batch;
} SGD_PARAMETERS;

//define adam parameters
typedef struct
{
    float lr;
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

    int string2method(std::string& s);
    
    //Method determination
    void getMethods();
    
    template <typename T, typename... Types>
    void getMethods(const T &firstArg, const Types &... args) {
        std::string method_type_s(firstArg);
        METHOD_TYPE method_type = string2method(method_type_s);
        if(method_type == SGD_METHOD) {
            SWLOG_INFO<<"Detect SGD method..."<<std::endl;
            _method = SGD_METHOD;
            _parameters = (SGD_PARAMETERS*)malloc(sizeof(SGD_PARAMETERS));
            getSGDParameters(args...);
        } 
        else if(method_type == ADAM_METHOD) {
            SWLOG_INFO<<"Detect ADAM method..."<<std::endl;
            _method = ADAM_METHOD;
            _parameters = (ADAM_PARAMETERS*)malloc(sizeof(ADAM_PARAMETERS));
            getADAMParameters(args...);
        } else {
            SWLOG_INFO<<"Undefined method..."<<std::endl;
            abort();
        }

    };


    // SGD Parameters read-in
    void getSGDParameters();
    void getSGDParameters(float lr,
                          float decay,
                          float momentum,
                          size_t batch);

    // ADAM Parameters read-in
    void getADAMParameters();
    void getADAMParameters(float lr); 
    // temporary functon to avoid compilation error
    void getADAMParameters(float lr,
                           float decay,
                           float momentum,
                           size_t batch) {} 
    void show();

    void run(IRGraph* graph_train);
    void destroy();

};

}  //namespace pass 
}  //namespace swc 



#endif /* !AUTODIFFPASS_H */
