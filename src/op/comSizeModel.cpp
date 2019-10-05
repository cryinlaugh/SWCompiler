/*
 * comSizeModel.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-10-04
 */


#include "comSizeModel.h"
#include <cmath>
#include <cassert>

//The first type of cost model:
size_t  comSizeModel1(size_t size, COMM_TYPE type, Config& config)
{
    int degree = config.mpi_size; 
    
    switch(type) {
        case RECV_SEND: return size;
        case REDUCE_SEND: return size * 2;
        case RECV_BCAST: return size * 2;
        case REDUCE_BCAST: return size;
        case SELF_CP: return size/degree;
        case SCATTER: return size;
        case GATHER: return size;
        case REDUCE: return size;
        case BCAST: return size;
        default: assert("invalid communication type\n"); 
    }
}


//The second type of cost model:
//  relative communication time with mpi-utilization model
size_t  comSizeModel2(size_t size, COMM_TYPE type, Config& config)
{
    int degree = config.mpi_size;


    switch(type) {
        case RECV_SEND: return size;
        case REDUCE_SEND: return size * log(degree)/log(2) + size;
        case RECV_BCAST: return size * log(degree)/log(2) + size;
        case REDUCE_BCAST: return (config.net_topo == RING_NET ? size : size * log(degree)/log(2));
        case SELF_CP: return 0;
        case SCATTER: return size;
        case GATHER: return size;
        case REDUCE: return size * log(degree)/log(2);
        case BCAST: return size * log(degree)/log(2);
        default: assert("invalid communication type\n"); 
    }
}

//The third type of cost model:
//  relative communication time with mpi-utilization model and optimization
size_t  comSizeModel3(size_t size, COMM_TYPE type, Config& config)
{
    int degree = config.mpi_size;


    switch(type) {
        case RECV_SEND: return size / degree / degree * (degree - 1) * 2;
        case REDUCE_SEND: return size * log(degree)/log(2) + size;
        case RECV_BCAST: return size * log(degree)/log(2) + size;
        case REDUCE_BCAST: return (config.net_topo == RING_NET ? size : 2 * size * log(degree)/log(2));
        case SELF_CP: return 0;
        case SCATTER: return size;
        case GATHER: return size;
        case REDUCE: return size * log(degree)/log(2);
        case BCAST: return size * log(degree)/log(2);
        default: assert("invalid communication type\n"); 
    }
}

