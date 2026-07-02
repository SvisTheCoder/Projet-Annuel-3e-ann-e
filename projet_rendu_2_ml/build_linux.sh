#!/usr/bin/env sh
set -e
cmake -S . -B build
cmake --build build
printf '\nCompilation terminee. Lancez : ./build/demo\n'
