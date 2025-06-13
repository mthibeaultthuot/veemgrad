cmake_minimum_required(VERSION 3.16)
project(cake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_METAL "Enable Metal backend" ON)
option(USE_CUDA  "Enable CUDA backend" OFF)
option(USE_ROCM  "Enable ROCm backend" OFF)

include_directories(include)

if(USE_METAL)
    message(STATUS "Building with Metal backend")

  
    add_library(metal_backend STATIC
        src/backend/metal_backend.cpp
    )

    target_link_libraries(metal_backend
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework QuartzCore"
    )


    target_include_directories(metal_backend PRIVATE
        src/backend
        src/backend/metal-cpp
    )
endif()

if(USE_CUDA)
    enable_language(CUDA)
    message(STATUS "Building with CUDA backend")

    add_library(cuda_backend STATIC
        src/backend/cuda_backend.cpp
    )
    target_include_directories(cuda_backend PRIVATE src/backend)
endif()

if(USE_ROCM)
    message(STATUS "Building with ROCm backend")

    add_library(rocm_backend STATIC
        src/backend/rocm_backend.cpp
    )
    target_include_directories(rocm_backend PRIVATE src/backend)
endif()

add_executable(backend
    src/main.cpp
)

target_include_directories(backend PRIVATE
    include
    src/backend
    src/backend/metal-cpp
)

if(USE_METAL)
    target_link_libraries(backend metal_backend)
endif()

if(USE_CUDA)
    target_link_libraries(backend cuda_backend)
endif()

if(USE_ROCM)
    target_link_libraries(backend rocm_backend)
endif()


if(TARGET metal_backend)
    install(TARGETS metal_backend
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

if(TARGET cuda_backend)
    install(TARGETS cuda_backend
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

if(TARGET rocm_backend)
    install(TARGETS rocm_backend
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()
