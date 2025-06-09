cmake_minimum_required(VERSION 3.16)
project(MatmulKernels)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/metal_matmul.metal)
set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/metal_matmul.air)
set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/metal_matmul.metallib)

add_custom_command(
    OUTPUT ${METAL_AIR}
    COMMAND xcrun -sdk macosx metal -c ${METAL_SRC} -o ${METAL_AIR}
    DEPENDS ${METAL_SRC}
    COMMENT "Compiling Metal shader to AIR"
    VERBATIM
)

add_custom_command(
    OUTPUT ${METAL_LIB}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
    DEPENDS ${METAL_AIR}
    COMMENT "Linking AIR to Metallib"
    VERBATIM
)

add_custom_target(compile_metal ALL DEPENDS ${METAL_LIB})

add_library(metal_matmul STATIC
    src/metal/metal_matmul.cpp
    src/metal/metal_wrapper.cpp
)





target_link_libraries(metal_matmul 
    "-framework Metal" 
    "-framework Foundation"
    "-framework CoreGraphics"
    "-framework QuartzCore"
)

add_dependencies(metal_matmul compile_metal)

add_executable(MatmulKernels
    src/main.cpp  
)


target_include_directories(metal_matmul PRIVATE 
    src/metal/metal-cpp
    src/metal
)

target_include_directories(MatmulKernels PRIVATE 
    src/metal/metal-cpp
    src/metal
)



target_link_libraries(MatmulKernels metal_matmul)

install(TARGETS metal_matmul
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

