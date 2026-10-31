cmake_minimum_required(VERSION 3.25)
project(kamping-v2-install-test LANGUAGES CXX)

find_package(kamping-v2 REQUIRED COMPONENTS v2)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE kamping::v2)
