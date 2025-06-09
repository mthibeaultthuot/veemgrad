
#include <metal_stdlib>
#include "shader_params.h"
using namespace metal;

kernel void matmul(
    device const float* A [[ buffer(0) ]],
    device const float* B [[ buffer(1) ]],
    device float* C       [[ buffer(2) ]],
    device const MatMulParams& params [[ buffer(3) ]],
    uint2 tid [[ thread_position_in_threadgroup ]],
    uint2 gid [[ thread_position_in_grid ]],
    uint2 group_id [[ threadgroup_position_in_grid ]])
{
    const uint TILE_SIZE = 16;

    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;

    float acc = 0.0;

    for (uint t = 0; t < (params.inner_dim + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        uint tiledRow = row;
        uint tiledCol = t * TILE_SIZE + tid.x;
        if (tiledRow < params.row_dim_y && tiledCol < params.inner_dim)
            tileA[tid.y][tid.x] = A[tiledRow * params.inner_dim + tiledCol];
        else
            tileA[tid.y][tid.x] = 0.0;

        tiledRow = t * TILE_SIZE + tid.y;
        tiledCol = col;
        if (tiledRow < params.inner_dim && tiledCol < params.col_dim_x)
            tileB[tid.y][tid.x] = B[tiledRow * params.col_dim_x + tiledCol];
        else
            tileB[tid.y][tid.x] = 0.0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; ++k)
            acc += tileA[tid.y][k] * tileB[k][tid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < params.row_dim_y && col < params.col_dim_x)
        C[row * params.col_dim_x + col] = acc;
}

