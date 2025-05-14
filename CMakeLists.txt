cmake_minimum_required(VERSION 3.28)
project(llm)

set(CMAKE_CXX_STANDARD 17)


# Link external library
# Method 1: If it's a custom lib with path

# Adjust this path to where the ONNX Runtime include and lib directories are located
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")

# Add include directories
include_directories(${ONNXRUNTIME_DIR}/include)

# Add library directories
link_directories(${ONNXRUNTIME_DIR}/lib)

# Use pkg-config to check for libsndfile
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

# Set include and library paths for libsndfile
include_directories(${SNDFILE_INCLUDE_DIRS})
link_directories(${SNDFILE_LIBRARY_DIRS})


# Add source files
add_executable(llm Inference.cpp)

# Method 1: If it's a system-wide installed lib 
# Link the ONNX Runtime and libsndfile libraries
target_link_libraries(llm PRIVATE
        ${SNDFILE_LIBRARIES}
        onnxruntime
        lrkllmrt
        lsndfile
        lpthread
        ldl
        lm
)


#target_include_directories(MyCppProject PRIVATE onnxruntime/lib/include)
#target_link_directories(MyCppProject PRIVATE path/to/lib/lib)
#target_link_libraries(MyCppProject PRIVATE <libname>)

