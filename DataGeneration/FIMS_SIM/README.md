
# FIMS - Data Generation: FIMS_SIM
## Description:
Simulation of the avalanche process for a single electron within the FIMS geometry.
Geometry is defined using Gmesh, which creates a finite-element mesh.
The field is determined using the finite-element methods of ELmer.
Electron avalanches are calculated using the Garfield++ method AvalancheMicroscopic.
The relevant parameters of the geometry and simulation are contained in runControl.
Avalanche data is ouput to the directory Data/ in root format.

## Contents:
1. runFIMS.ipynb - Python script with functions to run simulation and perform analysis.
2. avalanche.cc - Garfield++ simulation code.
3. CMakeLists.txt - CMake lists to build executable simulation.
4. runControl - Various user-defined geometry and simulation parameters.
5. runNo - Contains the next simulation run number and is automatically updated.
6. Geometry/ - Gmsh and Elmer definition and output files are saved here.
   * FIMS.txt - Coordinate definitions for the FIMS geometry. (Required for Gmsh)
   * FIMS.sif - Execution control file for Elmer.
   * dielectrics.dat - Dielectric contents for the FIMS materials. (Required for Elmer)
7. Data/ - Data output by the simulation is saved here as root files: sim.runno.root
8. log/ - Text files of the terminal outputs for the various executables are saved here.

## Use:
1. To utilize Gmsh to create finite element mesh:
   * JAMESTODO
2. To use Elmer to calculate electric field strengths:
   * JAMESTODO
3. To run a Garfield++ avalanche simulation.
   * (This is only useful when not changing the geometry or fields.)
   * Set desired simulation parameters in runControl.
   * In terminal execute: build/runAvalanche
4. To perform a complete simulation with a specified geometry:
   * For a single simulation:
     * Set parameters in runControl directly.
     * Execute Python function runSimulation().
   * For iterating through a single variable:
     * Execute Python function such as varyRadius().

## Additional Requirements:

1. Edit the pathways for executable programs. This includes:
   * GARFIELDPATH, PYTHONPATH, and GARFIELDSOURCE in runFIMS.ipynb
   * The 'Include' statement at the beginning of Geometry/FIMS.txt
