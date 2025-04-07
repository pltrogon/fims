# FIMS - Data Generation: electronAvalanche
## Description:
Garfield++ simulation of the avalanche process for a single electron within a uniform electric field to determine the gain.
Data is ouput to the directory Data/ in root format.

## Contents:
1. simSingleE.cc - Simulation code.
2. CMakeLists.txt - CMake lists to build executable simulation.
3. build/
    1. run_control - Various user-defined simulation parameters.
    2. runno - Contains the next simulation run number and is automatically updated.
    3. Data/ - Data output by the simulation is saved here as root files: sim.runno.root

## Use:
1. Create executable. In build/:
   *  cmake ..; make;
2. Set desired simulation parameters by editing run_control.
3. Run simulation. In build/:
   *  ./runElectronSim;
   
