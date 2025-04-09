cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(runElectronAvalanche)

# Specify the C++ compiler (if you're using a Homebrew compiler)
find_program(CMAKE_CXX_COMPILER NAMES g++ clang++ PATHS /opt/homebrew/bin NO_DEFAULT_PATH)
if(NOT CMAKE_CXX_COMPILER)
  find_program(CMAKE_CXX_COMPILER NAMES g++ clang++) # Fallback to system paths
endif()
# You might also need to set the C compiler if you're using mixed languages
find_program(CMAKE_C_COMPILER NAMES gcc clang PATHS /opt/homebrew/bin NO_DEFAULT_PATH)
if(NOT CMAKE_C_COMPILER)
  find_program(CMAKE_C_COMPILER NAMES gcc clang)
endif()

# Tell CMake to use the specified compilers
set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER} CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "C++ compiler" FORCE)

# Tell CMake where to look for Homebrew packages
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

find_package(GSL REQUIRED) # Now it should look in /opt/homebrew

if(NOT TARGET Garfield::Garfield)
  find_package(Garfield REQUIRED)
endif()

find_package(ROOT REQUIRED COMPONENTS Hist Tree)

add_executable(runElectronAvalanche electronAvalanche.cc)

if(GSL_FOUND)
  message(STATUS "Found GSL library: ${GSL_LIBRARY}")
  message(STATUS "Found GSL include directory: ${GSL_INCLUDE_DIR}")
  target_include_directories(runElectronAvalanche PRIVATE ${GSL_INCLUDE_DIR})
  target_link_libraries(runElectronAvalanche
      GSL::gsl # Use imported targets if available
      GSL::gslcblas # Use imported targets if available
      Garfield::Garfield
      ROOT::Hist
      ROOT::Tree)
else()
  message(FATAL_ERROR "GSL library not found.")
endif()

