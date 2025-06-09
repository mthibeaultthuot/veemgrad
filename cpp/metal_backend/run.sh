#!/bin/bash




echo "build project for : $1"
cd build
make

./$1



