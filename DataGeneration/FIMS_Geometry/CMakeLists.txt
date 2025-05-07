# Use the code in this section for standalone projects
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(gem)
if(NOT TARGET Garfield::Garfield)
  find_package(Garfield REQUIRED)
endif()

# ---Build executable------------------------------------------------------------
add_executable(fieldlines fieldlines.C)
target_link_libraries(fieldlines Garfield::Garfield)

# ---Copy all files locally to the build directory-------------------------------
foreach(_file input_file/dielectrics.dat input_file/fims.ep input_file/FIMS.msh input_file/FIMS.result input_file/mesh.boundary input_file/mesh.elements input_file/mesh.header input_file/mesh.nodes)
  configure_file(${_file} ${CMAKE_CURRENT_BINARY_DIR}/${_file} COPYONLY)
endforeach()
