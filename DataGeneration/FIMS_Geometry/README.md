# Run.py
python script that runs gmsh to create a finite element map, runs Elmer to create an electric field map using that geometry, and then imports the geometry and the field map into garfield++ to generate field lines which are then stored on a .csv file. The .csv file is then used to calculate the field line transparency and the field bundle diameter. All data is then output to a simData.csv file.

To run Run.py, 
1. Create a geometry using the gmsh software (This may be done via the gmsh GUI or by creating your own text script following gmsh file formatting. See gmsh documantation for details on how to create a script: https://gmsh.info/doc/texinfo/gmsh.html)
2. Name the file FIMS.txt and run gmsh using the following terminal command: "Programs/gmsh input_file/FIMS.txt -3" where "Programs" is the file pathway to your gmsh program and "input_file" is the file pathway to the FIMS.txt file (we have assumed that both file paths are in the working directory, as described below)
3. Open the Elmer GUI and set the "input_file" file directory as a new project directory.
4. Use the Elmer GUI to apply your relevant boundary conditions (see Elmer documantation for details on how to use Elmer: https://www.elmerfem.org/blog/documentation/)
5. While still in the Elmer GUI, go to the "MODEL" drop down menu and select "setup"
6. In the setup menu, change the second box of MeshDB to "input_file", change the results directory to "input_file", change the coordinate scaling to "1e-6", set the solver input file to "FIMS.sif", set the results file to "FIMS.ep", and in the free text box, type "Output file = "FIMS.result" "
7. Once the FIMS.sif file has been created by Elmer, open the command terminal and go to the simulation directory by using "cd simulation"
8. create the fieldlines.c executable by first building the "build" folder and then creating the executable via cmake: "mkdir build" followed by "cmake -B build && cmake --build build"
9. Edit the input values at the bottom of Run.py to fit your desired parameters
10. Run the python scrypt by typing "python3 Run.py" (check your python version if errors occur at this step)

## Notes:
The program assumes the following file structure: 
  1. All relevent code and programs are stored in a single folder labeled "simulation"
  2. The "simulation" folder has four primary sub folders:
       -"input_file"
       -"output_file"
       -"Programs"
       -"build"
  3. The "input_file" folder contains all the gmsh files and the Elmer files. The "output_file" folder contains all of the output files from the simulation such as "simData.csv". The "Programs" folder contains the Elmer and gmsh programs (although the exact location of the Elmer program is not relevant). The "build" folder is where the files and executables created by cmake are stored.
  4. Run.py assumes that "simulation" is the current working directory. Errors will occur if it is not because the .sif file used by Elmer is highly sensitive to the current working directory.
  5. Run.py also calls gmsh using its file path. If your file path for gmsh is different, then you will need to adjust the relevent lines of code (it is called in the "Terminal_Commands()" definition at the begining of Run.py and in the last elif statement of the "iterate_variable()" definition)
  6. You may also wish to have a fifth subfolder called "Plots", but this is optional and all plots can easily be stored in the "output_file" folder instead.
