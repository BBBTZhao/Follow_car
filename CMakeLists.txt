cmake_minimum_required(VERSION 3.0.2)			# cmake版本太低不行
project(follow)						# 新建一个工程
find_package(Eigen3 REQUIRED)                              #检索需要的库
find_package(OsqpEigen REQUIRED)
find_package(Python3 COMPONENTS Development NumPy) #matplotlibcpp需要

include_directories(${EIGEN3_INCLUDE_DIRS})   #连接头文件   
include_directories(include) #连接功能包头文件
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})#matplotlibcpp需要

add_library(mpc_lib MPCSolver.cpp) #将除了主函数以外的cpp文件添加成静态库

add_executable(follow main.cpp)	# 将main.cpp源文件编译成follow可执行文件
target_link_libraries(follow OsqpEigen::OsqpEigen) #将osqpeigen库链接到可执行文件
target_link_libraries(follow Python3::Python Python3::NumPy) #matplotlibcpp需要
target_link_libraries(follow mpc_lib) #将可执行文件连接到创建的静态库