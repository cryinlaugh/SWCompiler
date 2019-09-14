/*************************************************************************
    > File Name: ../src/codegen/MakefileBuilder.cpp
    > Author: wayne
    > Mail:
    > Created Time: 三  5/29 13:41:33 2019
 ************************************************************************/
#include "MakefileBuilder.h"

using namespace swc::codegen;
std::string MakefileBuilder::generate() {
    writer_ << "all: " + object_ << "\n";
    writer_ << "CXX := " << cxx_ << "\n";

    for (auto include_dir : include_dirs_)
        cxx_flags_ += (" -I" + include_dir);
    writer_ << "CXXFLAGS := " << cxx_flags_ << "\n";

    for (auto library_dir : library_dirs_)
        ld_flags_ += (" -L" + library_dir);
    for (auto lib : libraries_)
        ld_flags_ += (" -l" + lib);

    writer_ << "LDFLAGS := " << ld_flags_ << "\n";

    writer_ << object_ << ":";
    for (auto src : cxx_srcs_)
        writer_ << " " << src;
    writer_ << "\n";

    writer_ << "\t$(CXX)"
            << " $(CXXFLAGS)"
            << " $^ "
            << " $(LDFLAGS)"
            << " -o $@"
            << "\n";

    return writer_.get_code();
}
