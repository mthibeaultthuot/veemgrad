cmake_minimum_required(VERSION 3.16)
project(cake)

enable_testing()

include(FetchContent)
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(catch2)

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

    target_include_directories(metal_backend PRIVATE
        src/backend
        src/backend/metal-cpp
    )

    target_link_libraries(metal_backend PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework QuartzCore"
    )

    add_executable(metal_runtime_tests
        tests/metal_runtime_tests.cpp
    )

    target_include_directories(metal_runtime_tests PRIVATE
        include
        src/backend
        src/backend/metal-cpp
        ${catch2_SOURCE_DIR}/single_include
    )

    target_link_libraries(metal_runtime_tests PRIVATE
        metal_backend
        Catch2::Catch2WithMain
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework QuartzCore"
    )

    enable_testing()
    add_test(NAME MetalRuntimeTests COMMAND metal_runtime_tests)
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
