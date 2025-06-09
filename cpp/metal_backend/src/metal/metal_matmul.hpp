#pragma once

#ifndef METAL_MATMUL_HPP
#define METAL_MATMUL_HPP

#include <vector>
#include <Metal/Metal.hpp>
#include <string>




class MatMulKernel {
public:
    MatMulKernel(const std::string& metallibPath, const std::string& functionName, size_t maxElements);
    ~MatMulKernel();

    bool run(const std::vector<float>& matrixA,
             const std::vector<float>& matrixB,
             std::vector<float>& result,
             size_t rowsA, size_t colsA,
             size_t rowsB, size_t colsB);

    void warmUp();
private:

    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::ComputePipelineState* pipelineState;

    MTL::Buffer* bufferA;
    MTL::Buffer* bufferB;
    MTL::Buffer* bufferC;
    MTL::Buffer* paramsBuffer;

    size_t maxBufferSize;
};


std::unique_ptr<MatMulKernel> new_matmul_kernel(
    const std::string& metallibPath,
    const std::string& functionName,
    size_t maxElements
);

#endif 
