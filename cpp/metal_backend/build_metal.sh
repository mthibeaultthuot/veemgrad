#!/bin/bash


cd src/metal

rm metal_matmul.air 
rm metal_matmul.metallib



xcrun -sdk macosx metal -c metal_matmul.metal -o metal_matmul.air 
xcrun -sdk macosx metallib metal_matmul.air -o metal_matmul.metallib


mv metal_matmul.air ../../build/
mv metal_matmul.metallib ../../build/

