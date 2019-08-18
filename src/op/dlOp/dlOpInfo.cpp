/*************************************************************************
    > File Name: src/op/dlOp/dlOpInfo.cpp
    > Author: wayne
    > Mail:
    > Created Time: 二  8/ 6 10:13:18 2019
 ************************************************************************/

#include "dlOp.h"
#include <sstream>
#include <iterator>

template <typename T>
static std::string dumpVector(std::vector<T> vec) {
    if(vec.size() == 0)
        return "[]";
    std::ostringstream stream;
    stream << "[";
    for(auto i : vec) {
        stream << i << ", ";
    }
    std::string str = stream.str();
    return str.substr(0, str.length() - 2) + "]";
}

namespace swc {
namespace op {
std::string ScatterOp::getOpInfo() {
    std::ostringstream stream;
    /*
    stream << "Operation: "+ _opClassName + "\\n"
        << "_nInput: " << _nInput << "\\n"
        << "nOutput: " << _nOutput << "\\n";
    */
    stream << "axis: "  << axis_ << "\\n"
        << "degree: " << degree_ << "\\n";
    return Op::getOpInfo() + stream.str();
}

std::string GatherOp::getOpInfo() {
    std::ostringstream stream;
    stream << "axis: "  << axis_ << "\\n"
        << "degree: " << degree_ << "\\n";
    return Op::getOpInfo() + stream.str();
}

std::string TransformOp::getOpInfo() {
    std::ostringstream stream;
    stream << "pre_axis: "  << preAxis_<< "\\n"
           << "post_axis: "  << postAxis_ << "\\n"
           << "degree: " << degree_ << "\\n";
    return Op::getOpInfo() + stream.str();
}

std::string Conv2dOp::getOpInfo() {
    std::ostringstream stream;
    stream << "kernels: " << dumpVector(kernels_) << "\\n";
    // std::copy(kernels_.begin(), kernels_.end(), std::ostream_iterator<size_t>(stream, ", ")); 
    stream << "strides: " << dumpVector(strides_) << "\\n";
    stream << "pads: " << dumpVector(pads_) << "\\n";
    return Op::getOpInfo() + stream.str();
}

std::string MaxPoolOp::getOpInfo() {
    std::ostringstream stream;
    stream << "kernels: " << dumpVector(kernels_) << "\\n";
    // std::copy(kernels_.begin(), kernels_.end(), std::ostream_iterator<size_t>(stream, ", ")); 
    stream << "strides: " << dumpVector(strides_) << "\\n";
    stream << "pads: " << dumpVector(pads_) << "\\n";
    return Op::getOpInfo() + stream.str();
}

} // namespace op
} // namespace swc
