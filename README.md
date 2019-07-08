# SWCompiler

## Definition

SWCompiler is a end-to-end multi-domain application specific compiler targeting HPC platforms. 

SW~ represents the original purpose of SWCompiler is to support automatic parallelization targeting Sunway-series supercomputers, among which is the currently worlds leading Sunway TaihuLight supercomputer.

## Major Features

Front-end Features:

Generally, SWC is aiming to support any applications that can be described as a Tensor Computing Flow (TCF).

Deep Learning, which is one of the well known tensor computing applications, is the first domain that is supported by SWC.

Besides, 3-D biomocular reconstruction and most stencil computations in classical scientific computation applications are also taken into consideration as the supporting domain of SWC.

## Prerequisites
In order to build SWCompiler, it is necessary to install a few packages.
* protobuf
* graphviz

__Ubuntu__

`sudo apt-get install libprotobuf-dev protobuf-compiler graphviz`

__macOS__

`brew install protobuf graphviz`

## Compilation
To build in Debug mode, the following command maybe useful.
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DLEVELDEBUG=4 ..
make all
```

## SWCompiler C++ API
The SWCompiler C++ API provides high level C++ interface for neural network specification, optimization pass, visualization and compilation.

More details can be found in [C++ API](doc/cxxapi.md)
