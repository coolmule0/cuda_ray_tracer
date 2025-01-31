cmake_minimum_required(VERSION 3.11)
project(ExodusCMakeExample)

set(EXODUS_INCLUDE_DIR /home/john/Large/john/Programs/seacas/include)
set(EXODUS_LIB_DIR /home/john/Large/john/Programs/seacas/lib)


#### C ####
add_executable(exodustest exodus_test.cpp)

add_executable(exodusread exodus_reader.cpp)

target_include_directories(exodustest PRIVATE ${EXODUS_INCLUDE_DIR})
target_include_directories(exodusread PRIVATE ${EXODUS_INCLUDE_DIR})

target_link_libraries(exodustest PRIVATE ${EXODUS_LIB_DIR}/libexodus.so)
target_link_libraries(exodusread PRIVATE ${EXODUS_LIB_DIR}/libexodus.so)
