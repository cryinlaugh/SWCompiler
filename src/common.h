/*************************************************************************
	> File Name: common.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 04 Dec 2018 08:09:21 AM UTC
 ************************************************************************/

#ifndef _COMMON_H
#define _COMMON_H

#include <memory>
#include <vector>

#define NODETYPE int
#define OPTYPE int
#define TENSORTYPE int

enum OpType {
  TENSOR_OP,
  BASIC_OP,
  DL_OP
};

enum NodeType {
  TENSOR_NODE,
  OP_NODE
};

enum TensorType {
    D2=2,
    D1=1,
    D0=0,
    UNKNOWN=-1
};

template <typename Dtype>
class SWMem {
    
private:
    
    size_t _len;
    Dtype* _data;
    
public:
    
    SWMem(size_t len, Dtype* data);
    ~SWMem();
    
    Dtype* data();
    Dtype* mutable_data();
    
};

#endif
