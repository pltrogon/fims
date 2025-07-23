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
target_link_libraries(runAvalanche
    PRIVATE
        GSL::gsl
        GSL::gslcblas
        Garfield::Garfield
        ROOT::Hist
        ROOT::Tree
        ROOT::MathCore
)
