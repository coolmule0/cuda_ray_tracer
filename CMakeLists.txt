cmake_minimum_required(VERSION 3.11)
project(ExodusCMakeExample)

set(EXODUS_INCLUDE_DIR /home/john/Large/john/Programs/seacas/include)
set(EXODUS_LIB_DIR /home/john/Large/john/Programs/seacas/lib)

add_executable(main main.cc)

target_include_directories(main PRIVATE ${EXODUS_INCLUDE_DIR})

target_link_libraries(main PRIVATE ${EXODUS_LIB_DIR}/libexodus.so)
