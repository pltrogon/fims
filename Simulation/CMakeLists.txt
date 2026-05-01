cmake_minimum_required(VERSION 3.15)
project(runAvalanche C CXX)

# Standard C++ version for Garfield++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure Homebrew paths are considered on macOS
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

# Find required packages
find_package(GSL REQUIRED)
find_package(Garfield REQUIRED)
find_package(ROOT REQUIRED COMPONENTS Hist Tree MathCore)
find_package(OpenMP REQUIRED)

# --- Define common dependencies ---
set(COMMON_LIBS
    GSL::gsl
    GSL::gslcblas
    Garfield::Garfield
    ROOT::Hist
    ROOT::Tree
    ROOT::MathCore
    OpenMP::OpenMP_CXX
)

# --- Define executables ---
# Format: executable_name source_file.cc
set(EXECUTABLES
    runAvalanche            	avalanche.cc
    runAvalancheGridPix         avalancheGridPix.cc
    runTransparency           	checkTransparency.cc
    runTransparencyGridPix      checkTransparencyGridPix.cc
    runEfficiency           	checkEfficiency.cc
    runMagboltz             	magboltz.cc
    runParallelPlate        	parallelPlate.cc
    runParallelPlateEfficiency 	parallelPlateEfficiency.cc
    runAvalancheSurrounding 	avalancheSurrounding.cc
    runFullField		        plotFullField.cc
)

# --- Loop to create and link everything ---
while(EXECUTABLES)
    list(POP_FRONT EXECUTABLES EXE_NAME EXE_SRC)
    add_executable(${EXE_NAME} "garfieldExecutables/${EXE_SRC}")
    target_link_libraries(${EXE_NAME} PRIVATE ${COMMON_LIBS})
endwhile()
