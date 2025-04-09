cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(simSingleE)

if(NOT TARGET Garfield::Garfield)
  find_package(Garfield REQUIRED)
endif()

find_package(ROOT REQUIRED COMPONENTS Hist Tree)

add_executable(runElectronSim simSingleE.cc)
target_link_libraries(runElectronSim
	Garfield::Garfield
	ROOT::Hist
	ROOT::Tree)



