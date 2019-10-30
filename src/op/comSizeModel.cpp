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
    return -1;
}


//The second type of cost model:
//  relative communication time with mpi-utilization model
size_t  comSizeModel2(size_t size, COMM_TYPE type, Config& config)
{
    int degree = config.mpi_size;


    switch(type) {
        // caution size is serial size of data except for recv/send
        case RECV_SEND: return 2*size * (degree-1) / degree;
        case REDUCE_SEND: return size * log2(degree) + size*(degree-1)/degree;
        case RECV_BCAST: return size * log2(degree) + size*(degree-1)/degree;
        case REDUCE_BCAST: return (config.net_topo == RING_NET ? size : 2*size * log2(degree));
        case SELF_CP: return 0;
        case SCATTER: return size * (degree-1) / degree;
        case GATHER: return size * (degree-1) / degree;
        case REDUCE: return size * log2(degree);
        case BCAST: return size * log2(degree);
        default: assert("invalid communication type\n"); 
    }
    return -1;
}

//The third type of cost model:
//  relative communication time with mpi-utilization model and optimization
size_t  comSizeModel3(size_t size, COMM_TYPE type, Config& config)
{
    int degree = config.mpi_size;


    switch(type) {
        case RECV_SEND: return size / degree / degree * (degree - 1) * 2;
        case REDUCE_SEND: return size * log2(degree) + size;
        case RECV_BCAST: return size * log2(degree) + size;
        case REDUCE_BCAST: return (config.net_topo == RING_NET ? size : 2 * size * log2(degree));
        case SELF_CP: return 0;
        case SCATTER: return size;
        case GATHER: return size;
        case REDUCE: return size * log2(degree);
        case BCAST: return size * log2(degree);
        default: assert("invalid communication type\n"); 
    }
    return -1;
}

