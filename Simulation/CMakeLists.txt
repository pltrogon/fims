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
find_package(nlohmann_json REQUIRED)

# --- Define common dependencies ---
set(COMMON_LIBS
    GSL::gsl
    GSL::gslcblas
    Garfield::Garfield
    ROOT::Hist
    ROOT::Tree
    ROOT::MathCore
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
)

# --- Define executables ---
# Format: executable_name source_file.cc
set(EXECUTABLES
    runAvalanche            	avalanche.cc
    runMagboltz             	magboltz.cc
    runEfficiency               checkEfficiency.cc
    runFullField		        plotFullField.cc
)

# --- Loop to create and link everything ---
while(EXECUTABLES)
    list(POP_FRONT EXECUTABLES EXE_NAME EXE_SRC)
    
    add_executable(${EXE_NAME} 
        "${CMAKE_CURRENT_SOURCE_DIR}/garfieldExecutables/${EXE_SRC}"
        "${CMAKE_CURRENT_SOURCE_DIR}/myFunctions.cc"
    )
    
    target_include_directories(${EXE_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    
    target_link_libraries(${EXE_NAME} PRIVATE ${COMMON_LIBS})
endwhile()
