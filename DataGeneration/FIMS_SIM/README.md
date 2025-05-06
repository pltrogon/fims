
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

# Run.py
python script that runs gmsh to create a finite element map, runs Elmer to create an electric field map using that geometry, and then imports the geometry and the fiedl map into garfield++ to generate field lines which are then stored on a .csv file. The .csv file is then used to calculate the field line transparency and the field bundle diameter. All data is then output to a Sim_Data.csv file
## Notes:
The program assumes the following file structure: 
  1. All relevent code and programs are stored in a single folder labeles "simulation"
  2. The "simulation" folder has four primary sub folders:
       -"input_file"
       -"output_file"
       -"Programs"
       -"build"
  3. The "input_file" folder contains all the gmsh files and the Elmer files. The "output_file" folder contains all of the output files from the simulation such as "Sim_Data.csv". The "Programs" folder contains the Elmer and gmsh programs. The "build" folder is where the files and executables created by cmake are stored.
  4. Run.py assumes that "simulation" is the current working directory. Errors will occur if it is not because the .sif file used by Elmer is hihgly sensitive to the current working directory.
  5. Run.py also calls gmsh using its file path. If your file path for gmsh is different, then you will need to adjust the relevent lines of code (it called at the begining of each loop, just after the initial if statement)
  6. You may wish to also have a fifth subfolder called "Plots", but this is optional and all plots can easily be stored in the "output_file" folder instead.
