#ifndef DOTGEN_H_
#define DOTGEN_H_

#include <string>

namespace swc {

class IRGraph;

void dotGen(IRGraph *graph, std::string file="IRGraph.dot");

} // namespace swc

#endif /* !DOTGEN_H_ */
