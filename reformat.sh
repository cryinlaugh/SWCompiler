#!/bin/bash
set -eu

# pss --cpp -f include src test| xargs clang-format -style=LLVM -i
pss --cpp -f include src test| xargs clang-format -assume-filename=".clang-format" -i
