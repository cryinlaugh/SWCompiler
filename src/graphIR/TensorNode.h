/*
 * TensorNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef TENSORNODE_H
#define TENSORNODE_H

#include "IRNode.h"
#include "../tensor/tensor.h"

namespace swc {

template <typename Dtype>
class TensorNode : public IRNode {

public:
	TensorNode(){};
	~TensorNode(){};

	void setTensor(Tensor<Dtype>* tensor) {
		_tensor = tensor; 
	}

	Tensor<Dtype>* getTensor() {
		return _tensor;
	}

private:
	Tensor<Dtype>* _tensor; 

};

} //namespace swc


#endif /* !TENSORNODE_H */
