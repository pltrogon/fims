###################################
# CLASS DEFINITION FOR SIMULATION #
###################################
from __future__ import annotations

import numpy as np
import pandas as pd
import uproot
import awkward_pandas
import matplotlib.pyplot as plt
import os
import sys
import math
import subprocess
import time
import itertools
import re

class FIMS_Simulation:
    """
    Class representing the FIMS simulation.

    Initializes to a set of default parameters via a dictionary.

    The parameters within this dictionary can be adjusted and then used to execute
    a simulation. This process is as follows:

        1. Check that all required parameters are present and defined.
        2. Read and the simulation run number.
        3. Write the simulation parameters to the control files.
        4. Execute Gmsh to generate a finite-element mesh of the geometry.
        5. Execute Elmer to solve the E field for the mesh.
        6. Execute Elmer to solve the weighting field for the electrode.
        7. Execute Garfield++ to simulate electron multiplication effects.
        8. Reset parameters to defaults.

    *****
    IMPORTANT: The parameters are reset to defaults after every simulation.
    *****

    Attributes:
        param (dict): Parameter dictionary with the following entries:
            - padLength: Length of the side of the hexagonal pad (micron).
            - pitch: Distance between neighboring pads (micron).
            - gridStandoff: Distance from the top to the SiO2 layer to the bottom of the grid (micron).
            - gridThickness: Thickness of the grid (micron).
            - holeRadius: Radius of the hole in the grid (micron).
            - cathodeHeight: Distance from the top to the grid to the cathode plane (micron).
            - thicknessSiO2: Thickness of the SiO2 layer (micron).
            - pillarRadius: The radius of the insulating support pillars (micron).
            - driftField: The strength of the electric field in the drift region (V/cm).
            - fieldRatio: Ratio of the amplification field to the drift field.
            - numFieldLine: Number of field lines to calculate for visualization.
            - numAvalanche: Number of electrons (avalanches) to initiate
            - avalancheLimit: Limit of the number of electrons within a single avalanche.
            - gasCompAr: Percentage of Argon within gas volume.
            - gasCompCO2: Percentage of CO2 within gas volume.
    
    Methods defined in FIMS_Simulation:
        defaultParam
        _checkParam
        _getParam
        _getGarfieldPath
        _setupSimulation
        _readParam
        _writeFile
        _writeRunControl
        _readSIF
        _calcPotentials
        _writeSIF
        _makeWeighting
        _writeParam
        resetParam
        _getRunNumber
        _runGmsh
        _runElmer
        _runElmerWeighting
        _runGarfield
        runSimulation
        runForOptimizer         <----- NEW
        _runGetEfficiency       <----- NEW
        _readEfficiencyFile     <----- NEW
        findFieldForEfficiency <----- NEW
        runMagboltz             <----- NEW, changed fromm runDiffusion
        findFieldForTransparency <------ Changed from findMinField
    """

#***********************************************************************************#
    def __init__(self):
        """
        Initializes a FIMS_Simulation object.
        """
        #Include the analysis object        
        sys.path.insert(1, '../Analysis')
        from runDataClass import runData

        self.param = self.defaultParam()
        if not self._checkParam():
            raise ValueError('Error initializing parameters.')

        self._GARFIELDPATH = self._getGarfieldPath()
        if self._GARFIELDPATH is None:
            raise RuntimeError('Error getting Garfield++ path.')

        if not self._setupSimulation():
            raise RuntimeError('Error setting up simulation.')

#***********************************************************************************#
    #String definition
    def __str__(self):
        """
        Returns a formatted string containing all of the simulation parameters.
        """
        paramList = [f'{key}: {value}' for key, value in self.param.items()]            
        return "FIMS Simulation Parameters:\n\t" + "\n\t".join(paramList)

#***********************************************************************************#    
    def defaultParam(self):
        """
        Default FIMS parameters.
            Dimensions in microns.
            Electric field in V/cm.

            When both gas compositions are 0, the simulated gas is T2K.
    
        Returns:
            dict: Dictionary of default parameters and values.
        """
        defaultParam = {
            'padLength': 44.,
            'pitch': 225.,
            'gridStandoff': 100.,
            'gridThickness': 1.,
            'holeRadius': 55.,
            'cathodeHeight': 200.,
            'thicknessSiO2': 5.,
            'pillarRadius': 20.,
            'driftField': 280.,
            'fieldRatio': 80.,
            'transparencyLimit': 0.99,
            'numFieldLine': 25,
            'numAvalanche': 1500,
            'avalancheLimit': 600,
            'gasCompAr': 0.,
            'gasCompCO2': 0.,
        }
        return defaultParam

#***********************************************************************************#    
    def _checkParam(self):
        """
        Ensures that values exist for all necessary parameters.
        """
        #Check that any parameters exist
        if self.param is None or not self.param:
            print(f'No parameters.')
            return False

        #Check that all parameters are present
        allParam = self.defaultParam()
        for inParam in allParam:
            if inParam not in self.param:
                print(f"Missing parameter: '{inParam}'")
                return False
                
        return True

#***********************************************************************************#
    def _getParam(self, parameter):
        """
        Gets and returns desired parameter.
        """
        if parameter not in self.param:
            print(f'Invalid parameter: {parameter}.')
            return None
            
        return self.param[parameter]

#***********************************************************************************#       
    def _getGarfieldPath(self):
        """
        Reads and returns the filepath to the Garfield++ source script.
        
        Attempts to read the path to the Garfield++ source script given in 
        'GARFIELDPATH'. If this path-file does not exist, it creates one with a 
        placeholder message. If the path read from the file does not
        point to a valid 'setupGarfield.sh' file, an error message is printed.
        
        Returns:
            str or None: The validated Garfield++ source path if successful,
                         otherwise None.
        """
        filename = 'GARFIELDPATH'
        try:
            with open(filename, 'r') as file:
                garfieldPath = file.read().strip()
                if not os.path.exists(garfieldPath):
                    print(f"Error: File 'setupGarfield.sh' not found at {garfieldPath}.")
                    return None
    
        except FileNotFoundError:
            with open(filename, "w") as file:
                file.write('<<< Enter Garfield source path here. >>>')
                print(f"File '{filename}' created. Please update.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None
            
        return garfieldPath

#***********************************************************************************#
    def _setupSimulation(self):
        """
        Initializes Garfield++ and creates an avalanche executable.
        
        Reads the Garfiled++ source path, and ensures a log and build directory.
        Compiles the executable using cmake and make.
        Initializes a simulation run counter if it does not already exist.
    
        Note: If a degmentation fault occurs, it is most likely that the
              Garfield++ library is not sources correctly.
    
        Returns:
            bool: True if the setup is successful, False otherwise.
        """

        #Check for file pathways
        if not os.path.exists("log"):
            os.makedirs("log")
            
        if not os.path.exists("build"):
            os.makedirs("build")

        if not os.path.exists("build/parallelData"):
            os.makedirs("build/parallelData")

        if not os.path.exists("../Data/Magboltz"):
            os.makedirs("../Data/Magboltz")

        # Get garfield path into environment
        envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'
        try:
            envOutput = subprocess.check_output(envCommand, shell=True, universal_newlines=True)
            newEnv = dict(line.split('=', 1) for line in envOutput.strip().split('\n') if '=' in line)
            os.environ.update(newEnv)
        except subprocess.CalledProcessError as e:
            print(f"Failed to source Garfield environment: {e}")
            return False

        #Make executable
        makeBuild = (
            f'cmake .. && '
            f'make'
        )
        # Change to the build directory and run cmake and make
        originalCWD = os.getcwd()
        os.chdir('build')
        try:
            result = subprocess.run(
                makeBuild,
                shell=True,
                check=True,
                env=os.environ,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f'Failed to build project: {e.stderr}')
            os.chdir(originalCWD)
            return False
        finally:
            os.chdir(originalCWD)

    
        #Check for run number file
        if not os.path.exists('runNo'):
            with open('runNo', 'w') as file:
                file.write('1')
                
        return True

#***********************************************************************************#        
    def _readParam(self):
        """
        Reads the simulation parameters contained in the simulation control file.
    
        Returns:
            bool: True if parameters are read from file successfully, False otherwise.
        """
        
        filename = 'runControl'
        readInParam = {}
        
        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()  # Remove leading/trailing whitespace
                    if line.startswith('/') or not line:  # Skip comments and empty lines
                        continue
                    # Split the line at the '='
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        value = value.rstrip(';') # Remove trailing semicolon
                        readInParam[key] = value
                    else:
                        print(f"Skipping malformed line: {line}")
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return False
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return False

        if not self._checkParam():
            print("Error: Not all parameters found in 'runControl'.")
            return False

        self.param = readInParam
        return True

#***********************************************************************************#    
    def _writeFile(self, filename, lines):
        """
        Writes a list of strings to a specified file.
        Each string in the list is treated as a new line.
    
        Args:
            filename (str): The path to the file to write.
            lines (list): A list of strings to be written.
            
        Returns:
            bool: True if file written successfully, False otherwise.
        """
        try:
            with open(filename, 'w') as file:
                file.writelines(lines)
                
        except Exception as e:
            print(f"An error occurred while writing to {filename}: {e}")
            return False
            
        return True

#***********************************************************************************#
    def _writeRunControl(self):
        """
        Rewrites the simulation control file with the parameters.

        Returns:
            bool: True if new file written successfully, False otherwise.
        """
        filename = 'runControl'
    
        if not self._checkParam():
            return False
    
        #Read the old runControl file
        try:
            with open(filename, 'r') as file:
                oldLines = file.readlines()  # Read all lines of the file
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return False
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return False
    
        #Replace the old parameters with those in param
        newLines = []
        for line in oldLines:
            line = line.strip()
            if line.startswith('/') or not line:
                newLines.append(line + '\n')  # Keep comments and empty lines
                continue
    
            parts = line.split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                if key in self.param:
                    newLines.append(f"{key} = {self.param[key]};\n")  # Update value
                else:
                    newLines.append(line + '\n') #keep original line
            else:
                newLines.append(line + '\n')  # Keep non-parameter lines
    
        #Write new runControl file
        return self._writeFile(filename, newLines)

#***********************************************************************************#        
    def _readSIF(self):
        """
        Reads the FIMS.sif file and returns its content as a list of lines.
        This file is assumed to be in the 'Geometry/' folder.
    
        Returns:
            list: A list of strings, each a line of FIMS.sif.
                  Returns None if an error occurs.
        """
        filename = os.path.join('./Geometry', 'FIMS.sif')
    
        try:
            with open(filename, 'r') as file:
                sifLines = file.readlines()  # Read all lines of the file
                
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return None 
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None
    
        return sifLines
        
#***********************************************************************************#
    def _calcPotentials(self):
        """
        Calculates the required potentials to achieve a desired field ratio.
    
        Assumes that the drift field is defined as V/cm and distances are in microns.
    
        Returns:
            dict: Dictionary containing the potentials for the cathode and grid (in V).
                  Empty if necessary parameters are unavailable. 
        """
        if not self._checkParam():
            return {}
            

        driftField = float(self._getParam('driftField'))/1e4 #V/micron
        amplificationField = driftField*float(self._getParam('fieldRatio'))
        
        # Calculate the voltage required to achieve amplification field
        gridDistance = float(self._getParam('gridStandoff')) - float(self._getParam('gridThickness'))/2. #micron
        gridVoltage = amplificationField*gridDistance
    
        # Calculate for drift field
        cathodeDistance = float(self._getParam('cathodeHeight')) - float(self._getParam('gridThickness'))/2. #micron
        cathodeVoltage = driftField*cathodeDistance + gridVoltage
    
        potentials = {
            'cathodeVoltage': -cathodeVoltage,
            'gridVoltage': -gridVoltage
        }
        
        return potentials

#***********************************************************************************#
    def _writeSIF(self):
        """
        Rewrites the FIMS.sif file boundary conditons based on the given parameters.
    
        Assumes that 'Potential' is defined on the line following 'Name'.
    
        Returns:
            bool: True if file written successfully, False otherwise.
        """
        if not self._checkParam():
            return False
    
        #Read old .sif file
        sifLines = self._readSIF()
        if not sifLines:
            print('An error occurred while reading sif file.')
            return False
    
        potentials = self._calcPotentials()
    
        writeCathode = -1
        writeGrid = -1
        
        # Find the cathode and grid naming lines
        for i, line in enumerate(sifLines):
            if 'Name = "Cathode"' in line:
                writeCathode = i+1
            if 'Name = "Grid"' in line:
                writeGrid = i+1
    
        if writeCathode == -1 or 'Potential =' not in sifLines[writeCathode]:
            print('Error with cathode.')
            return False
        if writeGrid == -1 or 'Potential =' not in sifLines[writeGrid]:
            print('Error with grid.')
            return False
    
        #rewrite appropriate lines
        sifLines[writeCathode] = f"\tPotential = {potentials['cathodeVoltage']}\n"
        sifLines[writeGrid] = f"\tPotential = {potentials['gridVoltage']}\n"
    
        #Write new .sif file
        filename = os.path.join('./Geometry', 'FIMS.sif')
        return self._writeFile(filename, sifLines)

#***********************************************************************************#
    def _makeWeighting(self):
        """
        Writes a new .sif file for determining the weighting field.
    
        Sets all electrode boundary conditions to 0, then sets the pad potential to 1.
    
        Returns:
            bool: True if file written successfully, False otherwise.
        """
        #Read original sif file
        sifLines = self._readSIF()
        if not sifLines:
            print('An error occurred while reading sif file.')
            return False
    
        # Process lines one by one
        sifLinesNew = []
        centralPadBC = False

        for line in sifLines:
            #Replace all 'FIMS' with 'FIMSWeighting'
            line = line.replace('FIMS', 'FIMSWeighting')

            #Check if BC for Central Pad
            if 'CentralPad' in line:
                centralPadBC = True

            #Set potentials -> 1 if Central Pad, 0 otherwise
            if 'Potential = ' in line:
                if centralPadBC:
                    sifLinesNew.append('\tPotential = 1.0\n')
                    centralPadBC = False
                    continue
                else:
                    sifLinesNew.append('\tPotential = 0.0\n')
                    continue

            #Keep all other non-potential lines unchanged
            sifLinesNew.append(line)

        #Write new sif file
        filename = 'Geometry/FIMSWeighting.sif'
        return self._writeFile(filename, sifLinesNew)
        

#***********************************************************************************#
    def _writeParam(self):
        """
        Updates the simulation control files with the specified parameters.
    
        Validates input params, then writes simulation files.
    
        Returns:
            bool: True if all write operations were successful, False otherwise.
        """
        if not self._checkParam():
            return False
            
        if not self._writeRunControl():
            return False
        if not self._writeSIF():
            return False
        
        return True

#***********************************************************************************#
    def resetParam(self, verbose=True):
        """
        Rewrites the run control files with the default simulation parameters.
    
        Args:
            verbose (bool): Option available to supress reset notification.
    
        Returns:
            bool: True if reset successful, False otherwise.
        """
        self.param = self.defaultParam()
        if not self._writeParam():
            print('Error resetting parameters.')
            return False
    
        if verbose:
            print('Parameters have been reset.')
        
        return True

#***********************************************************************************#
    def _getRunNumber(self):
        """
        Gets the simulation number for the NEXT simulation.
    
        This number is stored in 'runNo'.
    
        Return:
            int: The simulation run number. Returns -1 if an error occurs.
        """
        filename = 'runNo'
    
        try:
            with open(filename, 'r') as file:
                content = file.read().strip()
                runNo = int(content)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return -1
        except ValueError:
            print(f"Error: Invalid number format in '{filename}")
            return -1
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return -1
        
        return runNo

#***********************************************************************************#
    def _runGmsh(self):
        """
        Runs the Gmsh program to generate a 3D finite-element mesh of the simulation geometry.
        Writes the output of Gmsh to 'log/logGmsh'.
    
        Utilizes several Gmsh options, including:
            -order 2: Second-order meshing
            -optimize_ho: Optimize the higher-order mesh
            -clextend: Extends the characteristic lengths to the whole geometry
            OptimizeNetgen: Enables Netgen algorithm optimization
            MeshSizeFromPoints: Uses mesh sizes defined at specific points
    
        Returns:
            bool: True if Gmsh runs successfully, False otherwise.
        """
        try:
            geoFile = 'FIMS.txt'
            with open(os.path.join(os.getcwd(), 'log/logGmsh.txt'), 'w+') as gmshOutput:
                startTime = time.monotonic()
                runReturn = subprocess.run(
                    ['gmsh', os.path.join('./Geometry/', geoFile),
                     '-order', '2', '-optimize_ho',
                     '-clextend', '1',
                     '-setnumber', 'Mesh.OptimizeNetgen', '1',
                     '-setnumber', 'Mesh.MeshSizeFromPoints', '1',
                     '-3',
                     '-format', 'msh2'],
                    stdout=gmshOutput, 
                    check=True
                )
                endTime = time.monotonic()
                gmshOutput.write(f'\n\nGmsh run time: {endTime - startTime} s')
    
                if runReturn.returncode != 0:
                    print('Gmsh failed. Check log for details.')
                    return False
    
        except FileNotFoundError:
            print("Unable to write to 'log/logGmsh.txt'.")
            return False
            
        return True

#***********************************************************************************#
    def _runElmer(self):
        """
        Runs Elmer to determine a finite-element Electric field solution.
    
        Converts a gmsh mesh to elmer format using ElmerGrid.
        Calculates potentials and E fields for the mesh using ElmerSolver.
        Output files are saved to a subdirectory called 'elmerResults/'.
        Writes the output of the programs to 'log/logElmerGrid' and 'log/logElmerSolver'.
    
        Returns:
            bool: True if ElmerGrid and ElmerSolver both run successfully.
                  False otherwise.
        """
        originalCWD = os.getcwd()
        os.chdir('./Geometry')
    
        os.makedirs("elmerResults", exist_ok=True)
            
        try:
            with open(os.path.join(originalCWD, 'log/logElmerGrid.txt'), 'w+') as elmerOutput:
                startTime = time.monotonic()
                runReturn = subprocess.run(
                    ['ElmerGrid', '14', '2', 'FIMS.msh', 
                     '-names',
                     '-out', 'elmerResults', 
                     '-autoclean'], 
                    stdout=elmerOutput,
                    check=True
                )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\nElmerGrid run time: {endTime - startTime} s')
                
                if runReturn.returncode != 0:
                    print('ElmerGrid failed. Check log for details.')
                    return False
                
            with open(os.path.join(originalCWD, 'log/logElmerSolver.txt'), 'w+') as elmerOutput:
                startTime = time.monotonic()
                runReturn = subprocess.run(
                    ['ElmerSolver', 'FIMS.sif'],
                    stdout=elmerOutput,
                    check=True
                )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\nElmerSolver run time: {endTime - startTime} s')
    
            if runReturn.returncode != 0:
                    print('ElmerSolver failed. Check log for details.')
                    return False
        finally:
            os.chdir(originalCWD)
        return True

#***********************************************************************************#
    def _runElmerWeighting(self):
        """
        Runs ElmerSolver to determine the weighing field for a simulation.
    
        Assumes that the Gmsh mesh has already been converted to 
        Elmer format by ElmerGrid. Creates the appropriate .sif file.
        Writes the ElmerSolver output to 'log/logElmerWeighting'.
    
        Returns:
            bool: True if ElmerSolver runs successfully, False othwerwise.
        """
        if not self._makeWeighting():
            print("Error occured creating weighting '.sif' file.")
            return False
        
        originalCWD = os.getcwd()
        os.chdir('./Geometry')
        try:
            with open(os.path.join(originalCWD, 'log/logElmerWeighting.txt'), 'w+') as elmerOutput:
                startTime = time.monotonic()
                runReturn = subprocess.run(
                    ['ElmerSolver', 'FIMSWeighting.sif'],
                    stdout=elmerOutput, 
                    check=True
                )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\nElmerSolver run time: {endTime - startTime} s')
    
            if runReturn.returncode != 0:
                    print('ElmerSolver failed for weighting. Check log for details.')
                    return False
        finally:
            os.chdir(originalCWD)
        return True

#***********************************************************************************#
    def _runGarfield(self):
        """
        Runs a Garfield++ executable to determine field lines and simulate 
        electron avalanches based on the parameters in 'runControl'.
    
        First links garfield libraries, creates the executable, and then runs the simulation.
        The simulation is numbered based on the number found in 'runNo';
        This also incremenmts this number.
        The information from this simulation is saved in .root format within 'Data/'.
        
        Returns:
            bool: True if Garfield executable runs successfully, False otherwise.
        """
        originalCWD = os.getcwd()
        try:
            os.chdir('./build/')

            # Get garfield path into environment
            envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'
            envOutput = subprocess.check_output(envCommand, shell=True, universal_newlines=True)
            newEnv = dict(line.split('=', 1) for line in envOutput.strip().split('\n') if '=' in line)
            os.environ.update(newEnv)

            with open(os.path.join(originalCWD, 'log/logGarfieldAvalanche.txt'), 'w+') as garfieldOutput:
                startTime = time.monotonic()
                setupAvalanche = (
                    f'./runAvalanche'
                )
                runReturn = subprocess.run(
                    setupAvalanche, 
                    stdout = garfieldOutput, 
                    shell = True, 
                    check = True,
                    env = os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')
    
            if runReturn.returncode != 0:
                    print('Garfield++ execution failed. Check log for details.')
                    return False
        finally:
            os.chdir(originalCWD)
        return True

#***********************************************************************************#
    def _runFieldLines(self):
        """
        Runs a Garfield++ executable to determine field lines based on the parameters
        in 'runControl'.
    
        First links garfield libraries, creates the executable, and then runs the simulation.
        The information from this simulation is saved in .txt format within 'Data/'.
        
        Returns:
            bool: True if Garfield executable runs successfully, False otherwise.
        """
        originalCWD = os.getcwd()
        try:
            os.chdir('./build/')

            # Get garfield path into environment
            envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'
            envOutput = subprocess.check_output(envCommand, shell=True, universal_newlines=True)
            newEnv = dict(line.split('=', 1) for line in envOutput.strip().split('\n') if '=' in line)
            os.environ.update(newEnv)

            with open(os.path.join(originalCWD, 'log/logGarfieldFieldLines.txt'), 'w+') as garfieldOutput:
                startTime = time.monotonic()
                setupFieldLines = (
                    f'./runFieldLines'
                )
                runReturn = subprocess.run(
                    setupFieldLines, 
                    stdout = garfieldOutput, 
                    shell = True, 
                    check = True,
                    env = os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')
    
            if runReturn.returncode != 0:
                    print('Garfield++ execution failed. Check log for details.')
                    return False
        finally:
            os.chdir(originalCWD)
        return True
    

#***********************************************************************************#
    def _runGetEfficiency(self, targetEfficiency=.95, threshold=10):
        """
        Runs a Garfield++ executable the generates electron avalanches until a target
        efficiency is either excluded or surpassed with a 2-sigma confidence.

        Args:
            targetEfficiency (float): The target efficiency to compare to.
            threshold: The minimum number of electrons required to be considered a 'success'.

        Returns:
            bool: True if program runs successfully, False otherwise.
        """
        originalCWD = os.getcwd()
        try:
            os.chdir('./build/')

            # Get garfield path into environment
            envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'
            envOutput = subprocess.check_output(envCommand, shell=True, universal_newlines=True)
            newEnv = dict(line.split('=', 1) for line in envOutput.strip().split('\n') if '=' in line)
            os.environ.update(newEnv)

            with open(os.path.join(originalCWD, 'log/logGarfieldGetEfficiency.txt'), 'w+') as garfieldOutput:
                startTime = time.monotonic()
                setupGetEfficiency = (
                    f'./runEfficiency {targetEfficiency} {threshold}'
                )
                runReturn = subprocess.run(
                    setupGetEfficiency, 
                    stdout = garfieldOutput, 
                    shell = True, 
                    check = True,
                    env = os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')
    
            if runReturn.returncode != 0:
                    print('Garfield++ execution failed. Check log for details.')
                    return False
            
        finally:
            os.chdir(originalCWD)
        return True
    
#***********************************************************************************#
    def runSimulation(self, changeGeometry=True):
        """
        Executes the full simulation process for the given parameters.
        Resets parameters upon completion.
    
        Simulation process:
            1. Check that all required parameters are present in 'param'.
            2. Read the run number in 'runNo'.
            3. Write the simulation parameters to the control files.
            4. Execute Gmsh to generate a finite-element mesh of the geometry.
            5. Execute Elmer to solve the E field for the mesh.
            6. Execute Elmer to solve the weighting field for the electrode.
            7. Execute the Garfield++ simulation for charge transport.
    
        Args:
            changeGeometry (bool): Allows for bypassing some executions such as Gmsh
                                   and ElmerWeighting. Decreases runtime.
                                   (For when geometry does not change.)    
        Returns:
            int: The run number of the simulation that was executed. 
                 Returns -1 if any errors occur.
        """
    
        if not self._checkParam():
            return -1
    
        #If geometry does not change, gmsh and weighting do not need to be done.
        #However, check that the mesh and weighting field files exist.
        #If not, override input and generate.
        if not changeGeometry:
            meshFile = os.path.exists('Geometry/FIMS.msh')
            weightFile = os.path.exists('Geometry/elmerResults/FIMSWeighting.result')
            if not (meshFile and weightFile):
                print('Warning. Attempt to skip Gmsh and ElmerWeighting. Overriding input.')
                changeGeometry = True


        #get the run number for this simulation
        runNo = self._getRunNumber()
        if runNo == -1:
            print("Error reading 'runNo'")
            return -1
        print(f'Running simulation - Run number: {runNo}')
        
        #write parameters for sim
        if not self._writeParam():
            print('Error writing parameters.')
            return -1
    
        #Allow for skipping gmsh if geometry has not changed.
        if changeGeometry:
            if not self._runGmsh():
                print('Error executing Gmsh.')
                return -1
    
        #Determine the Electric and weighting fields
        if not self._runElmer():
                print('Error executing Elmer (base).')
                return -1    

        #If geometry does not change, neither will weighting field.
        if changeGeometry: 
            if not self._runElmerWeighting():
                print('Error executing Elmer (weighting).')
                return -1
    
        #Run the electron transport simulation
        if not self._runGarfield():
            print('Error executing Garfield.')
            return -1
    
        #reset parameters to finish
        self.resetParam()
        
        return runNo
        
#***********************************************************************************#
#***********************************************************************************#
# METHODS FOR RUNNING MINIMUM FIELD
#***********************************************************************************#
#***********************************************************************************#

#***********************************************************************************#
    def _readEfficiencyFile(self):
        """
        Reads the file containing the simulated efficiency for a given field strength. 

        Returns:
            dict: Dictionary containing the parsed efficiency data. 
                  None if unsuccessful. Includes:
                  - 'stopCondition' (str): The avalanche stop condition (Converged or not).
                  - 'efficiency' (float): The simulated efficiency.
                  - 'efficiencyErr'(float): The uncertainty of the simulated efficiency.

        """

        try:
            with open('../Data/efficiencyFile.dat', 'r') as inFile:
                allLines = [inLine.strip() for inLine in inFile.readlines()]

        except FileNotFoundError:
            print(f"Error: File 'efficiencyFile.dat' not found.")
            return None


        if len(allLines) < 7:
            raise IndexError("Malformed file.")
        
        findEffValues = {}
        try:
            findEffValues['stopCondition'] = allLines[3].strip()

            findEffValues['efficiency'] = float(allLines[5].strip())
            findEffValues['efficiencyErr'] = float(allLines[6].strip())

        except (IndexError, ValueError) as e:
            print(f'Error extracting values - {e}')
            return None
                

        return findEffValues
    
#***********************************************************************************#
    def _getEfficiencyField(self, fields, efficiencies, efficienciesErr=None, damping=0.8):
        """
        Calculates the next field strength using the secant method to approach a target efficiency.

        If efficiency errors are provided, utilizes a step base on the maximum possible slope.
        Assumes that the efficiency is monotonically increasing.
        
        *TODO - Tested for when  current and prevoious effs are both LESS than target. Behaviour when bracketing target unknown.*

        Args:
            fields (np.array): Numpy array of the field strengths. Has form: [current, previous]
            efficiencies (np.array): Numpy array of the efficiencies. Has form: [current, previous, target]
            efficienciesErr (np.array): Optional. Array of errors in the efficiencies. Has form: [currentError, previousError]
            damping (float): Damping factor for the secant method to avoid too large of steps.

        Returns:
            float: Numerical solution to the field in order to achieve target efficiency
        """

        curField, prevField = fields
        curEff, prevEff, targetEff = efficiencies

        fieldDiff = curField - prevField

        #If errors exist, use the maximum possible slope
        if efficienciesErr is not None:
            curEffErr, prevEffErr = efficienciesErr
            prevEffMin = prevEff - prevEffErr
            curEffMax = curEff + curEffErr

            effDiff = curEffMax - prevEffMin
            targetDiff = targetEff - curEffMax

        else:
            effDiff = curEff - prevEff
            targetDiff = targetEff - curEff


        if abs(effDiff) < 0.001:
                print(f'Warning: Slope near zero. Using heuristic step of 5%.')
                return curField*1.05 

        fieldStep = damping*targetDiff*fieldDiff/effDiff
        print(f'Field Step: {fieldStep:.2f}')

        if fieldStep > 5:
            print(f'Warning: Step size limited to 5 for stability.')
            newField = curField+5

        elif fieldStep < 1:
            print(f'Warning: Field step small. Using heuristic step of 1.')
            newField = curField+1

        else:
            newField = curField+fieldStep

        return newField
    

#***********************************************************************************#
    def _getNextField(self, iterNo, efficiencyAtField, targetEfficiency):
        """
        TODO
        """
        # Determine new field strength
        newField = None

        if iterNo == 1:
            newField = self._getParam('fieldRatio')

        # Take constant step of 2 for 2nd iteration
        elif iterNo == 2:
            newField = efficiencyAtField['field'][0] + 2

        # Use secant method to determine new field
        else:
            newField = self._getEfficiencyField( 
                fields=np.array([efficiencyAtField['field'][-1], efficiencyAtField['field'][-2]]),
                efficiencies=np.array([efficiencyAtField['efficiency'][-1], efficiencyAtField['efficiency'][-2], targetEfficiency]),
                efficienciesErr=np.array([efficiencyAtField['efficiencyErr'][-1], efficiencyAtField['efficiencyErr'][-2]])
            )
        
        if newField is None:
            raise ValueError('Error: Invalid new field')
        
        return newField

#***********************************************************************************#
    def findFieldForEfficiency(self, targetEfficiency=.95, threshold=10):
        """
        Performs an iterative search to find the minimum Electric Field Ratio 
        required to achieve a specified detection efficiency for electron avalanches.

        Process:
            1 - Generates the geometry by executing Gmsh.
            2 - Solves an electric field using Elmer. 
            3 - Executes Garfield++ avalanches to determine a detection efficiency.
            4 - Repeats steps 2 and 3, increasing the field ration using a modified 
                secant method until a solution is reached.

        Note that this assumes that the field strength is monotonically increasing.

        Args:
            targetEfficiency (float): The target electron detection efficiency.
            threshold (int): The minimum avalanche size required for an event to 
                             be counted as 'detected'.

        Returns:
            bool: True if a solution is found or the search process completes.
                  False if there is a fatal execution error.
        """

        #Ensure all parameters exist and save them
        if not self._checkParam():
            return False
        saveParam = self.param.copy()

        #Write parameters and generate geometry
        if not self._writeParam():
            print('Error writing parameters.')
            return False
            
        print('Executing gmsh...')
        if not self._runGmsh():
                print('Error executing Gmsh.')
                return False
        
        print(f'Beginning search for minimum field to achieve {targetEfficiency*100:.0f}% efficiency...')
        
        iterNo = 0
        iterNoLimit = 10

        efficiencyAtField = {
            'field': [],
            'efficiency': [],
            'efficiencyErr': []
        }        

        validEfficiency = False
        self.param['numAvalanche'] = 3000
        self.param['avalancheLimit'] = 20 #Limit can be low - Check is boolean - above threshold or not
        
        while not validEfficiency:

            iterNo += 1
            if iterNo > iterNoLimit:
                print(f'Warning - Iteration limit reached ({iterNoLimit})')
                break

            print(f'Begining iteration: {iterNo}')
            print(efficiencyAtField)

            newField = self._getNextField(iterNo, efficiencyAtField, targetEfficiency)
            newFieldRounded = float(math.ceil(newField))

            efficiencyAtField['field'].append(newFieldRounded)
                
            print(f'Current field ratio: {newFieldRounded}')
            self.param['fieldRatio'] = newFieldRounded
            
            if not self._writeParam():
                print('Error writing parameters.')
                return False
            
            #Determine the electric field
            print('\tExecuting Elmer...')
            if not self._runElmer():
                print('Error executing Elmer.')
                return False

            #Determine the efficiency
            print('\tGetting efficiency...')
            if not self._runGetEfficiency(targetEfficiency=targetEfficiency, threshold=threshold):
                print('Error getting efficiency.')
                return False

            #Get efficiency values from file
            effResults = self._readEfficiencyFile()

            efficiencyAtField['efficiency'].append(effResults['efficiency'])
            efficiencyAtField['efficiencyErr'].append(effResults['efficiencyErr'])


            match effResults['stopCondition']:
                
                case 'DID NOT CONVERGE':
                    validEfficiency = False
                    print(f'Did not converge at {efficiencyAtField['field'][-1]} ({self.param['numAvalanche']} avalanches): {efficiencyAtField['efficiency'][-1]:.3f} +/- {efficiencyAtField['efficiencyErr'][-1]:.3f}')

                case 'CONVERGED':
                    validEfficiency = True
                    print(f'Converged at {efficiencyAtField['field'][-1]}: {efficiencyAtField['efficiency'][-1]:.3f} +/- {efficiencyAtField['efficiencyErr'][-1]:.3f}')
            
                case 'EXCLUDED':
                    validEfficiency = False
                    print(f'Excluded at {efficiencyAtField['field'][-1]}: {efficiencyAtField['efficiency'][-1]:.3f} +/- {efficiencyAtField['efficiencyErr'][-1]:.3f}')

                case _:
                    raise ValueError('Error - Malformed efficiency file.')      
        #End of find field for efficiency loop

        #Print solution
        finalField = self._getParam('fieldRatio')
        print(f'\nSolution: Field ratio = {finalField}')
        
        #Reset parameters
        self.resetParam()
        #load saved parameters back into class. Update field ratio with solution.
        self.param = saveParam
        self.param['fieldRatio'] = finalField

        return True

#***********************************************************************************#
    def _calcMinField(self): 
        """
        Calculates an initial guess for the minimum field ratio to achieve 100%
        field transparency.

        Calculation is based off of exponential fits to simulated data.

        Returns:
            float: Numerical solution to the minimum field for 100% transparency.
        """
        #Get geometry variables
        radius = self._getParam('holeRadius')
        standoff = self._getParam('gridStandoff')
        pad = self._getParam('padLength')
        pitch = self._getParam('pitch')
    
        #Calculate what the minimum field ratio should be
        gridArea = pitch**2*math.sqrt(3)/2
        holeArea = math.pi*radius**2
        optTrans = holeArea/gridArea
        
        standoffRatio = standoff/pitch
        padRatio = pad/pitch
        
        #Do calculation using values from fits
        '''Values come from exponential fits to data - 
            grid standoff ratio, pad length ratio, and 
            optical transparency vs minField.
          Results are then added together.'''
          #TODO - should this not be max(minFields)?
        radialMinField = 570.580*np.exp(-12.670*optTrans)
        standoffMinField = 27.121*np.exp(-15.9*standoffRatio)
        padMinField = 143.84*np.exp(-15.17*padRatio)
        
        minField = radialMinField + standoffMinField + padMinField + 3
        
        return minField

#***********************************************************************************#
    def findFieldForTransparency(self, runGMSH=True, minStepSize=1.2):
        """
        Runs simulations to determine what the minimum electric field ratio
        needs to be in order to have 100% E-field transparency.

        First, it optionally generates a gmsh FEM of the geometry before solving the
        E-field based on a preset field strength via Elmer FEM. It then generates
        field lines via Garfield++ and determines the transparency. If the 
        transparency is below the limit, a new field ratio is determined. The 
        resulting field is solved, and new field lines are generated. This continues
        until the criteria is reached.
        
        Upon completion, simulation files are reset.
        
        Args:
            margin (float): number multiplied to the predicted value to create a
                            buffer in case the raw value is too large or too small.
            
            minStepSize (float): number used as the step size once the calculated step
            size becomes too small.

        Returns:            
            finalField (float): The minimum field ratio that results in 100% transparency.
                                Retuns -1 on a failure or if an error occurs.
        """
        #Ensure all parameters exist and save them
        if not self._checkParam():
            return -1
        saveParam = self.param.copy()
       
        #Adjust the number of field lines to fill only last 20% of the unit cell
        numLines = int(self._getParam('numFieldLine')*.2)
        self.param['numFieldLine'] = numLines
        
        #Write parameters and generate geometry
        if not self._writeParam():
            print('Error writing parameters.')
            return -1
        
        if runGMSH:
            print('Executing gmsh...')
            if not self._runGmsh():
                    print('Error executing Gmsh.')
                    return -1

        print('Beginning search for minimum field...')
        isTransparent = False
        stepSize = 1.
        transLimit = self._getParam('transparencyLimit')
        curField = self._getParam('fieldRatio')
        
        while not isTransparent:
            #Calculate and write the new field ratio
            curField *= stepSize
            self.param['fieldRatio'] = curField
            if not self._writeParam():
                print('Error writing parameters.')
                return -1
            print(f'Testing field ratio of {curField}')            
            
            #Determine the electric field
            print('\tExecuting Elmer...')
            if not self._runElmer():
                print('Error executing Elmer.')
                return -1

            #Generate field lines
            print('\tGenerating field lines...')
            if not self._runFieldLines():
                print('Error generating field lines.')
                return  -1
            
            #Get the resulting field transparency
            with open('../Data/fieldTransparency.txt', 'r') as readFile:
                try:
                    transparency = float(readFile.read())

                except (ValueError, FileNotFoundError):
                    print("Error: Could not read or parse transparency file.")
                    return -1
            
            #Determine new step size
            if transLimit/transparency < minStepSize:
                stepSize = minStepSize
            else:
                stepSize = transLimit/transparency
            
            #Check transparency to terminate loop
            if transparency >= transLimit:
                isTransparent = True

        #Print solution
        finalField = self._getParam('fieldRatio')
        print(f'\nSolution: Field ratio = {finalField}')
        
        #Reset parameters
        self.resetParam()
        #load saved parameters back into class. Update field ratio with solution.
        self.param = saveParam
        self.param['fieldRatio'] = finalField

        return finalField





#***********************************************************************************#
#***********************************************************************************#
# METHODS FOR RUNNING CHARGE BUILDUP - UNTESTED (TODO)
#***********************************************************************************#
#***********************************************************************************#


#***********************************************************************************#
    def resetCharge(self):
        """
        Resets the charge buildup file to be empty.

        Returns:
            bool: True is reset is successful, False otherwise.
        """
        filename = 'Geometry/chargeBuildup.dat'

        try:
            with open(filename, 'w') as file:
                file.write('')
                
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return False
        except Exception as e:
            print(f'An error occurred with the file: {e}')
            return False
        
        return True
    
#***********************************************************************************#
    def _saveCharge(self, runNumber):
        """
        Saves the surface charge buildup to a file designated by runNumber.

        Copies the current charge buildup file into 'savedCharge/' as 'runXXXX.charge.dat'

        Args:
            runNumber (int): Run identifier for saved file.

        Returns:
            bool: True is file copy is successful, False otherwise.
        """
        chargeDirectory = 'savedCharge'
        if not os.path.exists(chargeDirectory):
            try:
                os.makedirs(chargeDirectory)
            except OSError as e:
                print(f"Error creating directory '{chargeDirectory}': {e}")
                return False
                
        saveFile = f'run{runNumber:04d}.charge.dat'
        saveFilePath = os.path.join('savedCharge', saveFile)

        filename = 'Geometry/chargeBuildup.dat'
        try:
            shutil.copyfile(filename, saveFilePath)
        except FileNotFoundError:
            print(f"Error: Source file '{filename}' not found.")
            return False
        except Exception as e:
            print(f'An error occurred while saving charge history: {e}')
            return False
        
        return True


#***********************************************************************************#
    def _readCharge(self):
        """
        Reads the file containing the built-up surface charge distribution.

        Assumes a space-separated dataset.

        Returns:
            dataframe: Pandas dataframe containing: x, y, z, and charge density.
                       None if no data is available or an error occurs.
        """
        filename = 'Geometry/chargeBuildup.dat'

        try:
            chargeData = pd.read_csv(
                filename, 
                sep=r'\s+', 
                header=None, 
                names=[
                    'x', 
                    'y', 
                    'z', 
                    'chargeDensity'
                ], 
                comment='#'
            )

            if chargeData.empty:
                print('No charge density.')
                return None
                
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return None
        except Exception as e:
            print(f"An error occurred with the file: {e}")
            return None

        return chargeData

#***********************************************************************************#
    def _writeCharge(self, builtUpCharge):
        """
        Writes the built-up surface charge to a file.

        Args:
            dataframe: Pandas dataframe containing: x, y, z, and charge density.

        Returns:
            bool: True is write is successful, otherwise False.
        """
        filename = 'Geometry/chargeBuildup.dat'

        try:
            builtUpCharge.to_csv(
                filename,
                sep=' ',
                index=False,
                header=False,
                float_format='%.10e'
            )
                
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return False
        except Exception as e:
            print(f"An error occurred with the file: {e}")
            return False
        
        return True
    

#***********************************************************************************#
    def _calculateSurfaceCharge(self, electronLocations):
        """
        Calculates the surface charge density based on the provided electron locations.

        Args:
            electronLocations (pd.DataFrame): A DataFrame containing the x, y, z 
                                              coordinates of the stuck electrons.

        Returns:
            dataframe: Pandas dataframe with the calculated surface charge density
                       at the coordinates (x, y, z).
        """
        electronCharge = -1.602176634e-19

        #Geometry parameters
        pitch = self._getParam('pitch')        
        xMax = math.sqrt(3)/2.*pitch
        xMin = -xMax
        yMax = pitch
        yMin = -yMin

        #Bin resolution and number of bins
        binResolution = pitch/100.
        numBinsX = int(np.ceil((xMax - xMin) / binResolution))
        numBinsY = int(np.ceil((yMax - yMin) / binResolution))

        #Isolate electrons from a given area
        filteredElectrons = electronLocations[
            (electronLocations['x'] > xMin) & 
            (electronLocations['x'] < xMax) &
            (electronLocations['y'] > yMin) & 
            (electronLocations['y'] < yMax)
        ].copy()

        #Assume the same z-coordinate for all data
        z = 0.
        if not filteredElectrons.empty:
            z = filteredElectrons['z'].iloc[0]

        #Generate 2D histogram of stuck electrons
        electronCounts, xEdges, yEdges = np.histogram2d(
            filteredElectrons['x'],
            filteredElectrons['y'],
            bins=[numBinsX, numBinsY],
            range=[[xMin, xMax], [yMin, yMax]]
        )
        electronCounts = electronCounts.T #because hist is weird
        
        # Calculate bin centers from the edges
        xCenters = (xEdges[:-1] + xEdges[1:]) / 2
        yCenters = (yEdges[:-1] + yEdges[1:]) / 2

        nonZeroY, nonZeroX = np.where(electronCounts > 0)

        # Calculate total charge and charge density for all bins
        binArea = binResolution**2
        totalBinCharge = electronCounts*electronCharge
        binChargeDensity = totalBinCharge/binArea

        # Extract the relevant data using the indices
        chargeData = pd.DataFrame({
            'x': xCenters[nonZeroX],
            'y': yCenters[nonZeroY],
            'z': z,
            'chargeDensity': binChargeDensity[nonZeroY, nonZeroX]
        })

        surfaceCharge = pd.DataFrame(chargeData)
        
        return surfaceCharge

#***********************************************************************************#
    def _sumChargeDensity(self, charge1, charge2):
        """
        Sums two charge density dataframes.

        Args:
            charge1, charge2 (df): Pandas dataframes containing x,y,z coordiantes and 
                                   a charge density to be summed together.
        Returns:
            DataFrame: The total summed charge density from the input dataframes. 
        """
        combinedCharge = pd.merge(
            charge1,
            charge2,
            on=['x', 'y', 'z'],
            how='outer',
            suffixes=('1', '2')
        )

        combinedCharge['chargeDensity1'] = combinedCharge['chargeDensity1'].fillna(0)
        combinedCharge['chargeDensity2'] = combinedCharge['chargeDensity2'].fillna(0)
        
        combinedCharge['chargeDensity'] = combinedCharge['chargeDensity1'] + combinedCharge['chargeDensity2']
        
        totalCharge = combinedCharge[['x', 'y', 'z', 'chargeDensity']]

        return totalCharge


#***********************************************************************************#
    def runChargeBuildup(self, buildupThreshold=1):
        """
        Runs an iterative simulation of building up charge on the SiO2 layer until a 
        steady-state solution is reached.

        This steady state corresponds to charge accumulation across the entire SiO2
        being less than a defined threshold. 
        
        Defines the geometry and detemrines an initial electric field solution 
        with no charge on the SiO2. Executes a series of electron avalanches, finding 
        the locations of any electrons whose track intersects with the top of the SiO2.
        A surface charge density is calculated from these locations. A new electric 
        field solution is determined with this surface charge present, and a new series
        of avalanches are executed. This process repeats until the amount of new charge
        is below a given amount. 
        
        The final surface charge density is saved to a file specified by the final
        simulation run number, then all parameter, including the surface charge, is reset.
    
        Args:
            buildupThreshold (int): The threshold for new surface charges signifying
                                    a steady-state solution.
    
        Returns:
            dict: Dictionary containing a summary of the iterative simulation.
                  None if an error occurs.
        """
        if not self._checkParam():
            return None
        
        #Check that the number of avalanches is a reasonable number
        # too many = too much charge unaffected by new stucks
        # too few = not enough stuck charges to make a difference
        numAvalanche = self.param('numAvalanche')
        if ((numAvalanche < 10) | (numAvalanche > 100)):
            print(f'Adjust number of avalanches ({numAvalanche}).')
            return None
        
        #Reset to ensure no initial charge
        self._resetCharge()
        
        #Prepare for iterations
        totalElectrons = 0
        numRuns = 0
        initialRun = True

        #Repeat simulations until electron buildup is less than threshold 
        newElectrons = buildupThreshold+1
        while newElectrons > buildupThreshold:
            #Reset built up charge and run simulation
            numRuns += 1
            newElectrons = 0

            #Only have to generate geometry on first run
            if numRuns > 1:
                initialRun = False
                
            #Run simulation - save parameters s reset at end of sim
            saveParam = self.param.copy()
            doneRun = self.runSimulation(changeGeometry=initialRun)
            self.param = saveParam
            
            #rGet the simulation data
            simData = runData(doneRun)

            #find locations of stuck electrons
            electronLocations = simData.findStuckElectrons()

            #Determine the amount of built-up charge
            newElectrons = len(electronLocations)
            totalElectrons += newElectrons
            newCharge = self._calculateSurfaceCharge(electronLocations)
            oldCharge = self._readCharge()
            totalCharge = self._sumChargeDensity(oldCharge, newCharge)

            #Update boundary condition with new surface charges
            self._writeCharge(totalCharge)


        #Save charge based on last run number, then reset file and all parameters.
        self._saveCharge(doneRun)
        self.resetCharge()    
        self.resetParam()

        chargeBuildupSummary = {
            'numRuns': numRuns,
            'finalRun': doneRun,
            'totalCharge': totalElectrons,
        }
        return chargeBuildupSummary


#***********************************************************************************#
    def runForOptimizer(self):
        """
        Runs the simulation, preserving the parameters.

        NOTE: This does not run gmsh or elmer to detemine the E Field. 
              It is assumed that these results exist already.
            
        Returns:
            int: The run number of the simulation that was executed.
        """

        #Check and save parameters
        if not self._checkParam():
            return -1
        saveParam = self.param.copy()


        #get the run number for this simulation
        runNo = self._getRunNumber()
        if runNo == -1:
            print("Error reading 'runNo'")
            return -1
        print(f'Running simulation - Run number: {runNo}')


        #Solve for the weighting field
        if not self._runElmerWeighting():
            print('Error executing Elmer (weighting).')
            return -1
        
        #Run the electron transport simulation
        if not self._runGarfield():
            print('Error executing Garfield.')
            return -1
        
        #Reset parameters
        self.resetParam()
        #Ensure saved parameters are still maintained
        self.param = saveParam
        
        return runNo
        
        
