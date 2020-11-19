
cmake_minimum_required(VERSION 2.8)
 
project(3DMachadoMIRunner)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(3DMachadoMIRunner MACOSX_BUNDLE 3DMachadoMIRunner.cxx)
target_link_libraries(3DMachadoMIRunner ${ITK_LIBRARIES})
