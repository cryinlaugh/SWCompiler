/*************************************************************************
    > File Name: MakefileBuilder.h
    > Author: wayne
    > Mail:
    > Created Time: 三  5/29 11:06:34 2019
 ************************************************************************/
#pragma once

#include "CodeWriter.h"
#include <vector>

namespace swc {
namespace codegen {
class MakefileBuilder {
  public:
    MakefileBuilder() {}
    ~MakefileBuilder() {}
    std::string generate();
    void setCXXCompiler(std::string cxx) { cxx_ = cxx; }
    void setNVCompiler(std::string nv) { nvcc_ = nv; }
    void addCXXSrc(std::string src) { cxx_srcs_.push_back(src); }
    void addIncDir(std::string dir) { include_dirs_.push_back(dir); }
    void addLibDir(std::string dir) { library_dirs_.push_back(dir); }
    void addLib(std::string lib) { libraries_.push_back(lib); }

  private:
    CodeWriter writer_;
    std::string cxx_{"g++"};
    std::string nvcc_{"nvcc"};
    std::string object_{"net.bin"};
    std::vector<std::string> cxx_srcs_;
    std::string cxx_flags_{"-std=c++11 -O3"};
    std::string ld_flags_;
    std::vector<std::string> include_dirs_;
    std::vector<std::string> library_dirs_;
    std::vector<std::string> libraries_;
};

} // namespace codegen
} // namespace swc
