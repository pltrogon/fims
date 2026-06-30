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

# --- Verify Garfield and ROOT ABI Compatibility ---
if(Garfield_FOUND AND ROOT_FOUND)
    include(CheckCXXSourceCompiles)
    
    # We use a core Garfield symbol to force the linker to check 
    # if libGarfield can successfully resolve its own internal ROOT symbols.
    set(CMAKE_REQUIRED_LIBRARIES Garfield::Garfield ROOT::Core)
    check_cxx_source_compiles("
        #include <Garfield/MediumMagboltz.hh>
        int main() {
            Garfield::MediumMagboltz med;
            return 0;
        }
    " GARFIELD_ABI_MATCHES)
    
    if(NOT GARFIELD_ABI_MATCHES)
        message(FATAL_ERROR "\n\n[FATAL ABI MISMATCH]\nThe central Garfield++ library found via GARFIELDPATH is broken or out of sync with the system ROOT library.\nTo fix this, please navigate to your central Garfield++ directory, clear its build cache, and run:\ncmake -DCMAKE_INSTALL_PREFIX=../install .. && make install\n\n")
    endif()
endif()

if(ROOT_FOUND)
    get_filename_component(ROOT_LIB_DIR "${ROOT_Hist_LIBRARY}" DIRECTORY)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ROOT_LIB_PATH.txt" "${ROOT_LIB_DIR}")
endif()
