# Use the code in this section for standalone projects
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(FIMS)
if(NOT TARGET Garfield::Garfield)
  find_package(Garfield REQUIRED)
endif()

find_package(ROOT REQUIRED COMPONENTS Hist Tree)



# ---Build executable------------------------------------------------------------
add_executable(fieldlines fieldlines.C)
target_link_libraries(fieldlines Garfield::Garfield)

add_executable(avalanche avalanche.cc)
target_link_libraries(avalanche Garfield::Garfield 
				GSL::gsl 
		    		GSL::gslcblas
				ROOT::Hist
				ROOT::Tree)

# ---Copy all files locally to the build directory-------------------------------
foreach(_file 	input_file/dielectrics.dat input_file/fims.ep input_file/FIMS.msh input_file/FIMS.result input_file/mesh.boundary input_file/mesh.elements input_file/mesh.header input_file/mesh.nodes input_file/runNo.txt input_file/runControl.txt)
		
  configure_file(${_file} ${CMAKE_CURRENT_BINARY_DIR}/${_file} COPYONLY)
endforeach()
