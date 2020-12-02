
cmake_minimum_required(VERSION 2.8)
 
project(MIRigidRegistration)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(MIRigidRegistration MACOSX_BUNDLE MIRigidRegistration.cxx)
target_link_libraries(MIRigidRegistration ${ITK_LIBRARIES})
