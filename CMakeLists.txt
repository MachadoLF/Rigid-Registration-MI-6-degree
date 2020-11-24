
cmake_minimum_required(VERSION 2.8)
 
project(MchadoMIRigidRegistration)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(MchadoMIRigidRegistration MACOSX_BUNDLE MchadoMIRigidRegistration.cxx)
target_link_libraries(MchadoMIRigidRegistration ${ITK_LIBRARIES})
