cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(runElectronAvalanche)

if(NOT TARGET Garfield::Garfield)
  find_package(Garfield REQUIRED)
endif()

find_package(ROOT REQUIRED COMPONENTS Hist Tree)

add_executable(runElectronAvalanche electronAvalanche.cc)
target_link_libraries(runElectronAvalanche
	Garfield::Garfield
	ROOT::Hist
	ROOT::Tree)



