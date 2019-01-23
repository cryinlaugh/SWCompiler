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

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

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
    D5=5,
    D4=4,
    D3=3,
    D2=2,
    D1=1,
    D0=0,
    UNKNOWN=-1
};

enum class TensorInitType {
    XAVIER,
    CONSTANT,
    ZERO,
    NONE
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


template<typename U, typename V>
int delVecMember(std::vector<U>& vec, V& del) {
  int delDone = 0;
  for (typename std::vector<U>::iterator it = vec.begin(); it != vec.end(); ) {
    if (*it == del) {
      it = vec.erase(it);
      delDone = 1;
      break;
    } else {
      ++it;
    }
  }
  return delDone;
}

#endif
