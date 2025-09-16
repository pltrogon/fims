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
find_package(OpenMP REQUIRED)

# Add the executable
add_executable(runAvalanche avalanche.cc)
add_executable(runFieldLines checkTransparency.cc)


# Link libraries and include directories
target_include_directories(runAvalanche PRIVATE
    $<TARGET_PROPERTY:GSL::gsl,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:Garfield::Garfield,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:ROOT::Hist,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(runAvalanche
    PRIVATE
        GSL::gsl
        GSL::gslcblas
        Garfield::Garfield
        ROOT::Hist
        ROOT::Tree
        ROOT::MathCore
	OpenMP::OpenMP_CXX
)

target_include_directories(runFieldLines PRIVATE
    $<TARGET_PROPERTY:GSL::gsl,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:Garfield::Garfield,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:ROOT::Hist,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(runFieldLines
    PRIVATE
        GSL::gsl
        GSL::gslcblas
        Garfield::Garfield
	OpenMP::OpenMP_CXX
)
