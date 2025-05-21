cmake_minimum_required(VERSION 3.12)
project(runAvalanche C CXX)

# Ensure Homebrew paths are considered on macOS for common libraries
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

# Find required packages
find_package(GSL REQUIRED)
find_package(Garfield REQUIRED)
find_package(ROOT REQUIRED COMPONENTS Hist Tree MathCore)

# Add the executable
add_executable(runAvalanche avalanche.cc)

# Link libraries and include directories
target_include_directories(runAvalanche PRIVATE
    ${GSL_INCLUDE_DIR}
    ${Garfield_INCLUDE_DIRS} # Assuming Garfield exports its include directories
    ${ROOT_INCLUDE_DIRS} # Assuming ROOT exports its include directories
)

target_link_libraries(runAvalanche
    GSL::gsl
    GSL::gslcblas
    Garfield::Garfield
    ROOT::Hist
    ROOT::Tree
    ROOT::MathCore
)

# Optional: Display status messages for found libraries
message(STATUS "Found GSL library: ${GSL_LIBRARY}")
message(STATUS "Found GSL include directory: ${GSL_INCLUDE_DIR}")
message(STATUS "Found Garfield library: ${Garfield_LIBRARY}") # Assuming Garfield exports its library variable
message(STATUS "Found ROOT libraries: ${ROOT_LIBRARIES}")
