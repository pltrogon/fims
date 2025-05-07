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
