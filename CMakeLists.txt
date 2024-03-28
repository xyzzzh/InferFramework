cmake_minimum_required(VERSION 3.27)
project(InferFramework)
set(CMAKE_CXX_STANDARD 17)

find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

set(link_lib glog::glog GTest::gtest)

set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
aux_source_directory(./test DIR_TEST_ARMA)
aux_source_directory(./source DIR_SOURCE_ARMA)
aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./source/layer/details DIR_DETAIL_LAYER)
aux_source_directory(./source/parser DIR_PARSER)

add_executable(InferFramework main.cpp ${DIR_TEST_ARMA} ${DIR_PARSER} ${DIR_SOURCE_ARMA} ${DIR_DETAIL_LAYER} ${DIR_ABSTRACT_LAYER})

target_link_libraries(InferFramework ${link_lib} ${link_math_lib})

target_include_directories(InferFramework PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(InferFramework PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(InferFramework PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(InferFramework PUBLIC ./include)

enable_testing()