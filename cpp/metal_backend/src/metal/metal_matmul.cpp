#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION  
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <Metal/Metal.hpp>

#include "metal_wrapper.h"
#include "metal_matmul.hpp"
#include "shader_params.h"

#include <fstream>
#include <memory>
#include <cstring>
#include <iostream>
#include <vector>
#include <cassert>

struct Params {
    uint32_t M, N, K;
};


MatMulKernel::MatMulKernel(const std::string& metallibPath, const std::string& functionName, size_t maxElements) 
    : device(nullptr), commandQueue(nullptr), pipelineState(nullptr), 
      bufferA(nullptr), bufferB(nullptr), bufferC(nullptr), paramsBuffer(nullptr) {
    
    device = MTL::CreateSystemDefaultDevice();
   
    commandQueue = device->newCommandQueue();
  
    NS::String* path = NS::String::string(metallibPath.c_str(), NS::UTF8StringEncoding);
    NS::URL* fileURL = NS::URL::fileURLWithPath(path);
    NS::Error* error = nullptr;
    MTL::Library* library = device->newLibrary(fileURL, &error);
    
    if (!library) {
        if (error) {
            std::string errorStr = error->localizedDescription()->utf8String();
        }
    }
  
    MTL::Function* function = library->newFunction(NS::String::string(functionName.c_str(), NS::UTF8StringEncoding));
    if (!function) {
        library->release();
    }

    pipelineState = device->newComputePipelineState(function, &error);
    if (!pipelineState) {
        if (error) {
            std::string errorStr = error->localizedDescription()->utf8String();
            function->release();
            library->release();
        }
        function->release();
        library->release();
    }

    maxBufferSize = maxElements * sizeof(float);
    bufferA = device->newBuffer(maxBufferSize, MTL::StorageModeShared);
    bufferB = device->newBuffer(maxBufferSize, MTL::StorageModeShared);
    bufferC = device->newBuffer(maxBufferSize, MTL::StorageModeShared);
    paramsBuffer = device->newBuffer(sizeof(Params), MTL::StorageModeShared);

    if (!bufferA || !bufferB || !bufferC || !paramsBuffer) {
        if (bufferA) bufferA->release();
        if (bufferB) bufferB->release();
        if (bufferC) bufferC->release();
        if (paramsBuffer) paramsBuffer->release();
        pipelineState->release();
        function->release();
        library->release();
    }

    memset(bufferA->contents(), 0, maxBufferSize);
    memset(bufferB->contents(), 0, maxBufferSize);
    memset(bufferC->contents(), 0, maxBufferSize);
    
    Params initParams = {1, 1, 1};
    memcpy(paramsBuffer->contents(), &initParams, sizeof(Params));

    function->release();
    library->release();

    warmUp(); 
}

MatMulKernel::~MatMulKernel() {


    if (bufferA) { 
        bufferA->release(); 
        bufferA = nullptr;
    }
    if (bufferB) { 
        bufferB->release(); 
        bufferB = nullptr;
    }
    if (bufferC) { 
        bufferC->release(); 
        bufferC = nullptr;
    }
    if (paramsBuffer) { 
        paramsBuffer->release(); 
        paramsBuffer = nullptr;
    }

    if (pipelineState) { 
        pipelineState->release(); 
        pipelineState = nullptr;
    }

    if (commandQueue) { 
        commandQueue->release(); 
        commandQueue = nullptr;
    }

    if (device) { 
        device->release(); 
        device = nullptr;
    }

}


void MatMulKernel::warmUp() {
    if (!commandQueue || !pipelineState) {
        return;
    }

    if (!bufferA || !bufferB || !bufferC || !paramsBuffer) {
        return;
    }


    MTL::CommandBuffer* cmd = commandQueue->commandBuffer();
    if (!cmd) {
        return;
    }

    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    if (!enc) {
        cmd->release();
        return;
    }


    enc->setComputePipelineState(pipelineState);
    enc->setBuffer(bufferA, 0, 0);
    enc->setBuffer(bufferB, 0, 1);
    enc->setBuffer(bufferC, 0, 2);
    enc->setBuffer(paramsBuffer, 0, 3);


    MTL::Size threads = MTL::Size::Make(1, 1, 1);
    MTL::Size groups = MTL::Size::Make(1, 1, 1);
    
    enc->dispatchThreadgroups(groups, threads);
    enc->endEncoding();
    
    cmd->commit();
    cmd->waitUntilCompleted();


    // enc->release();
    // cmd->release();
}



bool MatMulKernel::run(const std::vector<float>& matrixA,
                       const std::vector<float>& matrixB,
                       std::vector<float>& result,
                       size_t rowsA, size_t colsA,
                       size_t rowsB, size_t colsB) {
    
    if (colsA != rowsB) {
        return false;
    }

    if (matrixA.size() != rowsA * colsA || matrixB.size() != rowsB * colsB) {
        return false;
    }

    size_t outSize = rowsA * colsB;
    result.resize(outSize);

    if (matrixA.size() * sizeof(float) > maxBufferSize ||
        matrixB.size() * sizeof(float) > maxBufferSize ||
        outSize * sizeof(float) > maxBufferSize) {
        return false;
    }

    memcpy(bufferA->contents(), matrixA.data(), matrixA.size() * sizeof(float));
    memcpy(bufferB->contents(), matrixB.data(), matrixB.size() * sizeof(float));

    Params p = { static_cast<uint32_t>(rowsA),
                 static_cast<uint32_t>(colsB),
                 static_cast<uint32_t>(colsA) };

    memcpy(paramsBuffer->contents(), &p, sizeof(Params));

    auto* cmd = commandQueue->commandBuffer();
    if (!cmd) {
        return false;
    }
    
    auto* enc = cmd->computeCommandEncoder();
    if (!enc) {
        cmd->release();
        return false;
    }

    enc->setComputePipelineState(pipelineState);
    enc->setBuffer(bufferA, 0, 0);
    enc->setBuffer(bufferB, 0, 1);
    enc->setBuffer(bufferC, 0, 2);
    enc->setBuffer(paramsBuffer, 0, 3);

    constexpr int TILE = 16;
    MTL::Size threads = MTL::Size::Make(TILE, TILE, 1);
    MTL::Size groups = MTL::Size::Make((colsB + TILE - 1) / TILE,
                                       (rowsA + TILE - 1) / TILE,
                                       1);

    enc->dispatchThreadgroups(groups, threads);
    enc->endEncoding();

    cmd->commit();
    cmd->waitUntilCompleted();

    memcpy(result.data(), bufferC->contents(), outSize * sizeof(float));

    // enc->release();
    // cmd->release();
    return true;
}

std::unique_ptr<MatMulKernel> new_matmul_kernel(
    const std::string& metallibPath,
    const std::string& functionName,
    size_t maxElements
) {
    return std::make_unique<MatMulKernel>(metallibPath, functionName, maxElements);
}
