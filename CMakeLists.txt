cmake_minimum_required(VERSION 2.6)

project(APPEND_MODEL CXX)
find_package(pinocchio REQUIRED)


add_executable(mc contactDynamicsAndPartial.cpp)
target_link_libraries(mc PUBLIC pinocchio::pinocchio)