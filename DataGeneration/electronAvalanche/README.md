# FIMS - Data Generation: electronAvalanche
## Description:
Garfield++ simulation of the avalanche process for a single electron within a uniform electric field to determine the gain.
Geometry is defined using Garfield++ analytical components.
Electron avalanches are calculated using AvalancheMicroscopic.
Data is ouput to the directory Data/ in root format.

## Contents:
1. electronAvalanche.cc - Simulation code.
2. CMakeLists.txt - CMake lists to build executable simulation.
3. run_control - Various user-defined simulation parameters.
4. runno - Contains the next simulation run number and is automatically updated.
5. Data/ - Data output by the simulation is saved here as root files: sim.runno.root

## Use:
1. To create executable:
   * In terminal execute: mkdir build; cd build; cmake ..; make;
2. To run simulation.
   * Set desired simulation parameters in run_control.
   * In terminal execute: ./runElectronAvalanche
