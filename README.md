# SWCompiler

## Definition

SWCompiler is a end-to-end multi-domain application specific compiler targeting HPC platforms. 

SW~ represents the original purpose of SWCompiler is to support automatic parallelization targeting Sunway-series supercomputers, among which is the currently worlds leading Sunway TaihuLight supercomputer.

## Major Features

Front-end Features:

Generally, SWC is aiming to support any applications that can be described as a Tensor Computing Flow (TCF). 

Deep Learning, which is one of the well known tensor computing applications, is the first domain that is supported by SWC.

Besides, 3-D biomocular reconstruction and most stencil computations in classical scientific computation applications are also taken into consideration as the supporting domain of SWC.


## SWCompiler C++ API
The SWCompiler C++ API provides high level C++ interface for neural network specification, optimization pass, visualization and compilation.

More details can be found in [C++ API](doc/cxxapi.md)