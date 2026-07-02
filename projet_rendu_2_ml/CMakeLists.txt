cmake_minimum_required(VERSION 3.16)

project(ProjetRendu2 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD 11_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Importe la bibliothèque ML située dans le dossier frère ml_library
add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../ml_library
        ${CMAKE_CURRENT_BINARY_DIR}/ml_library_build
)

# Couche adaptatrice entre l'application C et les algorithmes C+
add_library(ml_core STATIC
        src/ml_api.cpp
)

target_include_directories(ml_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ml_core pourra utiliser Perceptron, MLP, RBF et SVM de ml_library.
target_link_libraries(ml_core PRIVATE
        ml_algorithms
)

# Gestion du chargement CSV et du split train/test.
add_library(c_support STATIC
        src/dataset.c
)

target_include_directories(c_support PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Application console principale.
add_executable(demo
        src/demo.c
        src/tests.c
)

target_link_libraries(demo PRIVATE
        ml_core
        c_support
)

# Serveur TCP de prédiction
add_executable(server
        src/server.c
)

target_link_libraries(server PRIVATE
        ml_core
)

# Client TCP
add_executable(client
        src/client.c
)

target_include_directories(client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(WIN32)
 target_link_libraries(server PRIVATE ws2_32)
 target_link_libraries(client PRIVATE ws2_32)
endif()