/*
 * comSizeModel.h
 * Copyright (C) 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef COMSIZEMODEL_H
#define COMSIZEMODEL_H


#include "common.h"


enum COMM_TYPE { RECV_SEND, REDUCE_SEND, RECV_BCAST, REDUCE_BCAST, SELF_CP,
                SCATTER, GATHER, REDUCE, BCAST};


//The first type of cost model:
//  only data size of communication with single node
size_t  comSizeModel1(size_t size, COMM_TYPE type, Config& config);

//The second type of cost model:
//  relative communication time with mpi-utilization model
size_t  comSizeModel2(size_t size, COMM_TYPE type, Config& config);



#endif /* !COMSIZEMODEL_H */
