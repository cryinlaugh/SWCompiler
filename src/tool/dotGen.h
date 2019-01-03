/*
 * dotGen.h_
 * Copyright (C) 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef DOTGEN_H_
#define DOTGEN_H_


namespace swc {

template<typename Dtype> class IRGraph;

template<typename Dtype>
void dotGen(IRGraph<Dtype>* graph); 

} //namespace swc

#endif /* !DOTGEN_H_ */
