cmake_minimum_required(VERSION 3.10)

include_directories(
        include
        external/eigen
)

add_executable(test_models

        src/main.cpp

        src/perceptron.cpp
        src/mlp.cpp
        src/rbf.cpp
        src/svm.cpp
)