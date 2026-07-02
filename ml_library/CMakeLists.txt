cmake_minimum_required(VERSION 3.16)

project(MLLibrary LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ml_algorithms STATIC
        src/perceptron.cpp
        src/mlp.cpp
        src/rbf.cpp
        src/svm.cpp
)

target_include_directories(ml_algorithms PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
)

add_executable(test_models
        src/main.cpp
)

target_link_libraries(test_models PRIVATE ml_algorithms)