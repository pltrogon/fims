
# FIMS - Data Generation: FIMS_SIM
## Description:
Simulation of the avalanche process for a single electron within the FIMS geometry.
Geometry is defined using Gmesh, and the field is determined using the finite-element methods in ELmer.
Electron avalanches are calculated using the Garfield++ method AvalancheMicroscopic.
The parameters of the geometry and simulation are contained in run_control.
Data is ouput to the directory Data/ in root format.

## Contents:
1. avalanche.cc - Simulation code.
2. CMakeLists.txt - CMake lists to build executable simulation.
3. run_control - Various user-defined simulation parameters.
4. runno - Contains the next simulation run number and is automatically updated.
5. Data/ - Data output by the simulation is saved here as root files: sim.runno.root

## Use:
1. To create executable:
   * In terminal execute: mkdir build; cd build; cmake ..; make;
2. To run simulation.
   * Set desired simulation parameters in run_control.
   * In terminal execute: ./runAvalanche

