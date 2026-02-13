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
import copy

#Include the analysis object        
sys.path.insert(1, '../Analysis')
from runDataClass import runData

class FIMS_Simulation:
    """
    Class representing the FIMS simulation.

    Initializes to a set of default parameters via a dictionary.

    The parameters within this dictionary can be adjusted and 
    then used to execute a simulation. This process is as follows:

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
            - padLength: Length of the side of the hexagonal pad 
                    (micron).
            - pitch: Distance between neighboring pads (micron).
            - gridStandoff: Distance from the top to the SiO2 layer
                    to the bottom of the grid (micron).
            - gridThickness: Thickness of the grid (micron).
            - holeRadius: Radius of the hole in the grid (micron).
            - cathodeHeight: Distance from the top to the grid to the
                    cathode plane (micron).
            - thicknessSiO2: Thickness of the SiO2 layer (micron).
            - pillarRadius: The radius of the insulating support 
                    pillars (micron).
            - driftField: The strength of the electric field in the 
                    drift region (V/cm).
            - fieldRatio: Ratio of the amplification field to the 
                    drift field.
            - numFieldLine: Number of field lines to calculate for 
                    visualization.
            - numAvalanche: Number of electrons (avalanches) to initiate
            - avalancheLimit: Limit of the number of electrons within a 
                    single avalanche.
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
        runForOptimizer
        _runFieldLines
        _runGetEfficiency
        -----Find Minimum Field Methods-----
        _calcMinField
        _readEfficiencyFile
        _readTransparencyFile
        _getFieldRatio
        _getNextField
        _findFieldForEfficiency
        _findFieldForTransparency
        findMinFieldRatio
        findMinFieldRatioAlt         <------New
        _generateGeometry
        _solveEField
        _solveWeightingField
        _runElectronTransport
        runForOptimizer
        -----Untested Charge Buildiup------
    """

#**********************************************************************#
    def __init__(self):
        """
        Initializes a FIMS_Simulation object.
        """
        try:
            self.param = self.defaultParam()
            self._checkParam()
        except KeyError:
            raise RuntimeError('Error initializing parameters.')

        self._GARFIELDPATH = self._getGarfieldPath()
        if self._GARFIELDPATH is None:
            raise RuntimeError('Error getting Garfield++ path.')

        self._setupSimulation()

        return

#**********************************************************************#
    #String definition
    def __str__(self):
        """
        Returns a formatted string containing all of the simulation parameters.
        """
        paramList = [f'{key}: {value}' for key, value in self.param.items()]            
        return "FIMS Simulation Parameters:\n\t" + "\n\t".join(paramList)

#**********************************************************************#    
    def defaultParam(self):
        """
        Default FIMS parameters.
            Dimensions in microns.
            Electric field in V/cm.

        Returns:
            dict: Dictionary of default parameters and values.
        """
        defaultParam = {
            'padLength': 12.,
            'pitch': 80.,
            'gridStandoff': 45.,
            'gridThickness': 1.,
            'holeRadius': 20.,
            'cathodeHeight': 140.,
            'thicknessSiO2': 5.,
            'pillarRadius': 2.,
            'driftField': 280.,
            'fieldRatio': 80.,
            'numFieldLine': 25,
            'numAvalanche': 6000,
            'avalancheLimit': 800,
            'gasCompAr': 0.95,
            'gasCompCO2': 0.00,
            'gasCompCF4': 0.03,
            'gasCompIsobutane': 0.02,
            'gasPenning': 0.385
        }
        return defaultParam

#**********************************************************************#    
    def _checkParam(self):
        """
        Ensures that values exist for all necessary parameters.
        """
        #Check that any parameters exist
        if self.param is None or not self.param:
            raise(KeyError('Parameter dictionary is empty.'))

        #Check that all parameters are present
        allParam = self.defaultParam()
        for inParam in allParam:
            if inParam not in self.param:
                raise KeyError(f"Parameter '{inParam}' missing from "
                                "parameter dictionary.")
            
        #Check gas composition 
        self._checkGasComp()
                
        return
    
#**********************************************************************#    
    def _checkGasComp(self):
        """
        Ensures that the gas composition percentages sum to 1.0.
        """
        totalComp = (
            float(self._getParam('gasCompAr')) +
            float(self._getParam('gasCompCO2')) +
            float(self._getParam('gasCompCF4')) +
            float(self._getParam('gasCompIsobutane'))
        )
        
        if not math.isclose(totalComp, 1.0, rel_tol=1e-3):
            raise ValueError(f'Gas composition percentages must sum '
                                f'to 1.0. Current sum: {totalComp}')
        
        return

#**********************************************************************#
    def _getParam(self, parameter):
        """
        Gets and returns a copy of the desired parameter.
        """
        if parameter not in self.param:
            raise KeyError(f'Error - Invalid parameter: {parameter}.')
            
        return copy.copy(self.param[parameter])

#**********************************************************************#       
    def _getGarfieldPath(self):
        """
        Reads and returns the filepath to the Garfield++ source script.
        
        Attempts to read the path to the Garfield++ source script given
        in 'GARFIELDPATH'. If this path-file does not exist, it creates
        one with a placeholder message. If the path read from the file 
        does not point to a valid 'setupGarfield.sh' file, an error 
        message is printed.
        
        Returns:
            str or None: The validated Garfield++ source path if 
            successful. Otherwise, None.
        """
        filename = 'GARFIELDPATH'
        try:
            with open(filename, 'r') as file:
                garfieldPath = file.read().strip()
                if not os.path.exists(garfieldPath):
                    print("Error: File 'setupGarfield.sh' not found "
                            f"at '{garfieldPath}'.")
                    return None
    
        except FileNotFoundError:
            with open(filename, "w") as file:
                file.write('<<< Enter Garfield source path here. >>>')
                print(f"File '{filename}' created. Please update.")
            return None
        except Exception as e:
            print(f'An error occurred while reading the file: {e}')
            return None
            
        return garfieldPath

#**********************************************************************#
    def _setupSimulation(self):
        """
        Initializes Garfield++ and creates an avalanche executable.
        
        Reads the Garfiled++ source path, and ensures a log and build
        directory. Compiles the executable using cmake and make.
        Initializes a simulation run counter if it does not already 
        exist.
    
        Note: If a segmentation fault occurs, it is most likely that the
              Garfield++ library is not sourced correctly.
    
        """

        #Check for necessary pathways and create if not present
        paths = [
            'log', 
            'build',
            'build/parallelData', 
            '../Data/Magboltz'
        ]
        for inPath in paths:
            os.makedirs(inPath, exist_ok=True)

        #Make runControl file
        self._makeRunControl()

        # Get garfield path into environment
        envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'
        try:
            envOutput = subprocess.check_output(envCommand, shell=True, universal_newlines=True)
            newEnv = dict(line.split('=', 1) for line in envOutput.strip().split('\n') if '=' in line)
            os.environ.update(newEnv)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'ERROR - Failed to source Garfield++ environment: {e.stderr}')

        #Make executables
        makeBuild = (
            f'cmake .. && '
            f'make'
        )

        # Change to the build directory and run cmake and make
        originalCWD = os.getcwd()
        os.chdir('build')
        
        try:
            subprocess.run(
                makeBuild,
                shell=True,
                check=True,
                env=os.environ,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'ERROR - Failed to build project: {e.stderr}')
        finally:
            os.chdir(originalCWD)

        #Check for run number file
        if not os.path.exists('runNo'):
            with open('runNo', 'w') as file:
                file.write('1000')  # Starting run number
                
        return

#**********************************************************************#
    def _makeRunControl(self):
        """
        Creates a runControl file with the default parameters.
        """

        filename = 'runControl'
        defaultParams = self.defaultParam()

        runControlLines = [
            '// FIMS Simulation Control File //',
            '// Assumes the form "variable = value;" for each line.',
            '// Number of input parameters (numInputs included):',
            'numInputs = 19;',   ##### NOTE: Change this if parameters are added/removed #####
            '',
            '//----- Geometry parameters -----//',
            '// Dimensions in microns.',
            '',
            '// Pad and pitch',
            f'padLength = {defaultParams["padLength"]:.1f};',
            f'pitch = {defaultParams["pitch"]:.1f};',
            '',
            '// Grid',
            f'gridStandoff = {defaultParams["gridStandoff"]:.1f};',
            f'gridThickness = {defaultParams["gridThickness"]:.1f};',
            f'holeRadius = {defaultParams["holeRadius"]:.1f};',
            '',
            '// Other',
            f'cathodeHeight = {defaultParams["cathodeHeight"]:.1f};',
            f'thicknessSiO2 = {defaultParams["thicknessSiO2"]:.1f};',
            f'pillarRadius = {defaultParams["pillarRadius"]:.1f};',
            '',
            '//----- Electric field parameters -----//',
            '// Electric field in V/cm.',
            f'driftField = {defaultParams["driftField"]:.1f};',
            f'fieldRatio = {defaultParams["fieldRatio"]:.1f};',
            f'numFieldLine = {defaultParams["numFieldLine"]};',
            '',
            '//----- Simulation parameters -----//',
            '// Avalanche controls',
            f'numAvalanche = {defaultParams["numAvalanche"]};',
            f'avalancheLimit = {defaultParams["avalancheLimit"]};',
            '',
            '// Gas composition (in percentage)',
            f'gasCompAr = {defaultParams["gasCompAr"]:.2f};',
            f'gasCompCO2 = {defaultParams["gasCompCO2"]:.2f};',
            f'gasCompCF4 = {defaultParams["gasCompCF4"]:.2f};',
            f'gasCompIsobutane = {defaultParams["gasCompIsobutane"]:.2f};',
            '',
            f'gasPenning = {defaultParams["gasPenning"]:.3f};',
            '',
            '// End of runControl file\n'
        ]

        runControlInfo = '\n'.join(runControlLines)
        try:
            with open(filename, 'w') as file:
                file.write(runControlInfo)
        except Exception as e:
            raise RuntimeError(f"An error occurred while writing to the file '{filename}': {e}")
        
        return

#**********************************************************************#        
    def _readParam(self):
        """
        Reads the simulation parameters contained in the simulation control file.
        """
        
        filename = 'runControl'
        readInParam = {}
        try:
            with open(filename, 'r') as file:
                for lineNo, line in enumerate(file, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith(('/', '#')): 
                        continue
                    
                    # Split at the first '='
                    if '=' not in line:
                        raise ValueError(f"Malformed line {lineNo} in {filename}: Missing '='")
                    key, _, value = line.partition('=')
                    
                    key = key.strip()
                    value = value.strip().rstrip(';')
                    
                    if not key:
                        raise ValueError(f"Malformed line {lineNo} in {filename}: Missing Key")

                    readInParam[key] = value

        except FileNotFoundError:
            raise FileNotFoundError(f"Critical Error: Configuration file '{filename}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error reading parameters from {filename}: {e}")

        self.param = readInParam
        self._checkParam()
        
        return

#**********************************************************************#    
    def _writeFile(self, filename, lines):
        """
        Writes a list of strings to a specified file.
        Each string in the list is treated as a new line.
    
        Args:
            filename (str): The path to the file to write.
            lines (list): A list of strings to be written.
        """
        try:
            with open(filename, 'w') as file:
                file.writelines(lines)
                
        except Exception as e:
            raise RuntimeError(f"An error occurred while writing to the file '{filename}': {e}")
            
        return

#**********************************************************************#
    def _writeRunControl(self):
        """
        Rewrites the simulation control file with the current parameters.
        """
        filename = 'runControl'
    
        self._checkParam()
    
        #Read the old runControl file
        try:
            with open(filename, 'r') as file:
                oldLines = file.readlines()  # Read all lines of the file
        except FileNotFoundError:
            raise FileNotFoundError(f"Critical Error: '{filename}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error reading parameters from {filename}: {e}")
    
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
        self._writeFile(filename, newLines)

        return

#**********************************************************************#        
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
            raise FileNotFoundError(f"Error: File '{filename}' not found.") 
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the file '{filename}': {e}")
    
        return sifLines
        
#**********************************************************************#
    def _calcPotentials(self):
        """
        Calculates the required potentials to achieve a desired field ratio.
    
        Assumes that the drift field is defined as V/cm and distances are in microns.
    
        Returns:
            dict: Dictionary containing the potentials for the cathode and grid (in V).
                  Empty if necessary parameters are unavailable. 
        """
        self._checkParam()
            

        driftField = float(self._getParam('driftField'))/1e4 #V/micron
        fieldRatio = float(self._getParam('fieldRatio'))
        amplificationField = driftField*fieldRatio #V/micron
        
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

#**********************************************************************#
    def _writeSIF(self):
        """
        Rewrites the FIMS.sif file boundary conditons based on the given parameters.
    
        Assumes that 'Potential' is defined on the line following 'Name'.
        """
        self._checkParam()
    
        #Read old .sif file
        sifLines = self._readSIF()
    
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
            raise RuntimeError('Error with cathode.')
        if writeGrid == -1 or 'Potential =' not in sifLines[writeGrid]:
            raise RuntimeError('Error with grid.')

        #rewrite appropriate lines
        sifLines[writeCathode] = f"\tPotential = {potentials['cathodeVoltage']}\n"
        sifLines[writeGrid] = f"\tPotential = {potentials['gridVoltage']}\n"
    
        #Write new .sif file
        filename = os.path.join('./Geometry', 'FIMS.sif')
        self._writeFile(filename, sifLines)

        return

#**********************************************************************#
    def _makeWeighting(self):
        """
        Writes a new .sif file for determining the weighting field.
    
        Sets all electrode boundary conditions to 0, then sets the pad potential to 1.

        """
        #Read original sif file
        sifLines = self._readSIF()
    
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
        self._writeFile(filename, sifLinesNew)

        return
        

#**********************************************************************#
    def _writeParam(self):
        """
        Updates the simulation control files with the specified parameters.
    
        Validates input params, then writes simulation files.
        """
        self._checkParam()
        self._writeRunControl()
        self._writeSIF()
        
        return

#**********************************************************************#
    def resetParam(self, verbose=True):
        """
        Rewrites the run control files with the default simulation parameters.
    
        Args:
            verbose (bool): Option available to supress reset notification.
        """
        self.param = self.defaultParam()
        self._writeParam()
    
        if verbose:
            print('Parameters have been reset.')
        
        return

#**********************************************************************#
    def _getRunNumber(self):
        """
        Gets the simulation number for the ** NEXT ** simulation.
    
        This number is stored in 'runNo'.
    
        Return:
            int: The simulation run number.
        """
        filename = 'runNo'
    
        try:
            with open(filename, 'r') as file:
                content = file.read().strip()
                runNo = int(content)

        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File '{filename}' not found.")
        except ValueError:
            raise ValueError(f"Error: Invalid number format in '{filename}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the file '{filename}': {e}")
        
        return runNo

#**********************************************************************#
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
            print('\tExecuting Gmsh...')

            geoFile = 'FIMS.txt'
            with open(os.path.join(os.getcwd(), 'log/logGmsh.txt'), 'w+') as gmshOutput:
                startTime = time.monotonic()
                subprocess.run(
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
    
    
        except FileNotFoundError:
            raise FileNotFoundError('Error: Gmsh log file not found.')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Gmsh failed with exit code {e.returncode}.')
            
        return

#**********************************************************************#
    def _runElmer(self):
        """
        Runs Elmer to determine a finite-element Electric field solution.
    
        Converts a gmsh mesh to elmer format using ElmerGrid.
        Calculates potentials and E fields for the mesh using ElmerSolver.
        Output files are saved to a subdirectory called 'elmerResults/'.
        Writes the output of the programs to 'log/logElmerGrid' and 'log/logElmerSolver'.
        """
        originalCWD = os.getcwd()
        os.chdir('./Geometry')
    
        os.makedirs("elmerResults", exist_ok=True)
            
        try:
            print('\tExecuting Elmer...')

            with open(os.path.join(originalCWD, 'log/logElmerGrid.txt'), 'w+') as elmerOutput:
                startTime = time.monotonic()
                subprocess.run(
                    ['ElmerGrid', '14', '2', 'FIMS.msh', 
                     '-names',
                     '-out', 'elmerResults', 
                     '-autoclean'], 
                    stdout=elmerOutput,
                    check=True
                )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\nElmerGrid run time: {endTime - startTime} s')

                
            with open(os.path.join(originalCWD, 'log/logElmerSolver.txt'), 'w+') as elmerOutput:
                startTime = time.monotonic()
                subprocess.run(
                    ['ElmerSolver', 'FIMS.sif'],
                    stdout=elmerOutput,
                    check=True
                )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\nElmerSolver run time: {endTime - startTime} s')

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Elmer failed with exit code {e.returncode}.')
        
        finally:
            os.chdir(originalCWD)
        return

#**********************************************************************#
    def _runElmerWeighting(self):
        """
        Runs ElmerSolver to determine the weighing field for a simulation.
    
        Assumes that the Gmsh mesh has already been converted to 
        Elmer format by ElmerGrid. Creates the appropriate .sif file.
        Writes the ElmerSolver output to 'log/logElmerWeighting'.
        """
        self._makeWeighting()
        
        originalCWD = os.getcwd()
        os.chdir('./Geometry')
        try:
            print('\tExecuting Elmer weighting...')

            with open(os.path.join(originalCWD, 'log/logElmerWeighting.txt'), 'w+') as elmerOutput:
                startTime = time.monotonic()
                subprocess.run(
                    ['ElmerSolver', 'FIMSWeighting.sif'],
                    stdout=elmerOutput, 
                    check=True
                )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\nElmerSolver run time: {endTime - startTime} s')
    
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'ElmerSolver weighting failed with exit code {e.returncode}.')
        
        finally:
            os.chdir(originalCWD)

        return

#**********************************************************************#
    def _runGarfield(self):
        """
        Runs a Garfield++ executable to determine field lines and simulate 
        electron avalanches based on the parameters in 'runControl'.
    
        First links garfield libraries, creates the executable, and then runs the simulation.
        The simulation is numbered based on the number found in 'runNo';
        This also incremenmts this number.
        The information from this simulation is saved in .root format within 'Data/'.
        """
        originalCWD = os.getcwd()
        try:
            print('\tExecuting Garfield++...')
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
                subprocess.run(
                    setupAvalanche, 
                    stdout = garfieldOutput, 
                    shell = True, 
                    check = True,
                    env = os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')
    
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Garfield++ avalanche execution failed with exit code {e.returncode}.')
        
        finally:
            os.chdir(originalCWD)

        return
    

#**********************************************************************#
    def _runFieldLines(self):
        """
        Runs a Garfield++ executable to determine field lines based on the parameters
        in 'runControl'.
    
        First links garfield libraries, creates the executable, and then runs the simulation.
        The information from this simulation is saved in .txt format within 'Data/'.
        """
        originalCWD = os.getcwd()
        try:
            print('\tGenerating field lines...')
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
                subprocess.run(
                    setupFieldLines, 
                    stdout = garfieldOutput, 
                    shell = True, 
                    check = True,
                    env = os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Garfield++ fieldine execution failed with exit code {e.returncode}.')

        finally:
            os.chdir(originalCWD)

        return
    
#**********************************************************************#
    def _runGetEfficiency(self, targetEfficiency=.95, threshold=10):
        """
        Runs a Garfield++ executable the generates electron avalanches until a target
        efficiency is either excluded or surpassed with a 2-sigma confidence.

        Args:
            targetEfficiency (float): The target efficiency to compare to.
            threshold: The minimum number of electrons required to be considered a 'success'.
        """
        originalCWD = os.getcwd()
        try:
            print('\tGetting efficiency...')
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
                subprocess.run(
                    setupGetEfficiency, 
                    stdout = garfieldOutput, 
                    shell = True, 
                    check = True,
                    env = os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Garfield++ efficiency execution failed with exit code {e.returncode}.')
            
        finally:
            os.chdir(originalCWD)

        return
    
#**********************************************************************#
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
                                   (For use when geometry does not change.)    
        Returns:
            int: The run number of the simulation that was executed. 
        """
    
        self._checkParam()
    
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
        print(f'Running simulation - Run number: {runNo}')
        
        #write parameters for sim
        self._writeParam()
    
        #Allow for skipping gmsh if geometry has not changed.
        if changeGeometry:
            self._generateGeometry()
    
        #Determine the Electric and weighting fields
        self._solveEField()

        #If geometry does not change, neither will weighting field.
        if changeGeometry: 
            self._solveWeightingField()
    
        #Run the electron transport simulation
        self._runElectronTransport()
    
        #reset parameters to finish
        self.resetParam()
        
        return runNo
        
#**********************************************************************#
#**********************************************************************#
# METHODS FOR RUNNING MINIMUM FIELD
#**********************************************************************#
#**********************************************************************#
    def _calcMinField(self): 
        """
        Calculates an initial guess for the minimum field ratio to 
        achieve 100% field transparency and 95% efficiency.

        Calculation is based off of exponential fits to simulated data.
        
        Values come from 3 separate exponential fits to simulated data:
            grid standoff ratio, pad length ratio, and optical
            transparency vs minField. Results from each fit are then 
            added together.
        
        Note: ratios are assumed to be completely independent of each
        other, which is not strictly true. However, it is usually 
        sufficient for the purposes of an initial guess.    
        
        Returns:
            float: Numerical solution to the minimum field for 
            100% transparency and 95% efficiency.
        """
        #Get geometry variables
        radius = self._getParam('holeRadius')
        standoff = self._getParam('gridStandoff')
        pad = self._getParam('padLength')
        pitch = self._getParam('pitch')

        gridArea = (pitch**2)*math.sqrt(3)/2
        holeArea = math.pi*(radius**2)
        
        #Convert to dimensionless variables
        optTrans = holeArea/gridArea
        standoffRatio = standoff/pad
        padRatio = pad/pitch
        
        #Inserting values into fitted equations
        radialMinField = 570.580*np.exp(-12.670*optTrans)
        standoffMinField = 26.85*np.exp(-4.46*standoffRatio)
        padMinField = 143.84*np.exp(-15.17*padRatio)
        
        #Ar+CO2 (depreciated)
        #standEffField = 53.21*np.exp(-0.38*standoffRatio)
        #constEffField = 23.47
        
        #T2K gas
        padEffField = 371.4*np.exp(-0.16*pad)
        standEffField = 190.6*np.exp(-0.02*standoff)
        constEffField = 50
        
        #Minimum field for 100% transparency
        minFieldTrans = radialMinField + standoffMinField + padMinField + 3
        
        #Minimum field for 95% efficiency
        minFieldEff = padEffField + standEffField + constEffField
        
        #Choose the larger of the two fields so that both 
        #conditions are satisfied simultaneously.
        minField = max(minFieldTrans, minFieldEff)
        
        return minField

#**********************************************************************#
    def _readEfficiencyFile(self):
        """
        Reads the file containing the simulated efficiency for a given field strength. 

        Returns:
            dict: Dictionary containing the parsed efficiency data. 
                  Includes:
                  - 'stopCondition' (str): The avalanche stop condition (Converged or not).
                  - 'efficiency' (float): The simulated efficiency.
                  - 'efficiencyErr'(float): The uncertainty of the simulated efficiency.

        """

        with open('../Data/efficiencyFile.dat', 'r') as inFile:
            allLines = [inLine.strip() for inLine in inFile.readlines()]

        if len(allLines) < 7:
            raise IndexError('Malformed file.')
        
        findEffValues = {}
        try:
            findEffValues['stopCondition'] = allLines[3].strip()

            findEffValues['efficiency'] = float(allLines[5].strip())
            findEffValues['efficiencyErr'] = float(allLines[6].strip())

        except (IndexError, ValueError) as e:
            raise RuntimeError(f'Error parsing efficiency file: {e}')
                
        return findEffValues
    

#**********************************************************************#
    def _readTransparencyFile(self):
        """
        Reads the file containing the simulated transparency for a given field strength. 

        Returns:
            dict: Dictionary containing the parsed efficiency data. 
                  Includes:
                  - 'transparency' (float): The simulated transparency.
                  - 'transparencyErr'(float): The uncertainty of the simulated transparency.

        """

        with open('../Data/fieldTransparency.dat', 'r') as inFile:
            allLines = [inLine.strip() for inLine in inFile.readlines()]

        if len(allLines) < 2:
            raise IndexError('Malformed file.')
        
        findTransparencyValues = {}
        try:
            findTransparencyValues['transparency'] = float(allLines[3].strip())
            findTransparencyValues['transparencyErr'] = float(allLines[4].strip())

        except (IndexError, ValueError) as e:
            raise RuntimeError(f'Error parsing transparency file: {e}')
                
        return findTransparencyValues
    
    
#**********************************************************************#
    def _getFieldRatio(self, fields, values, valuesErr=None, damping=0.8):
        """
        Calculates the next field strength ratio using the secant method to approach a target value.

        If errors are provided, utilizes a step base on the maximum possible slope.
        Assumes that the value is monotonically increasing for field ratios.

        The resulting field ratio is rounded up to the nearest integer.
        
        *TODO - Tested for when  current and previous values are both LESS than target. Behavior when bracketing target unknown.*

        Args:
            fields (np.array): Numpy array of the field ratio. Has form: [current, previous]
            values (np.array): Numpy array of the values. Has form: [current, previous, target]
            valuesErr (np.array): Optional. Array of errors in the values. Has form: [currentError, previousError]
            damping (float): Damping factor for the secant method to avoid too large of steps.

        Returns:
            float: Numerical solution to the field ratio in order to achieve target value
        """

        curField, prevField = fields
        curVal, prevVal, targetVal = values

        fieldDiff = curField - prevField

        #If errors exist, use the maximum possible slope
        if valuesErr is not None:
            curValErr, prevValErr = valuesErr
            prevValMin = prevVal - prevValErr
            curValMax = curVal + curValErr

            valDiff = curValMax - prevValMin
            targetDiff = targetVal - curValMax

        else:
            valDiff = curVal - prevVal
            targetDiff = targetVal - curVal


        if abs(valDiff) < 0.001:
                print(f'Warning: Slope near zero. Using heuristic step of 2.')
                return curField + 2

        fieldStep = damping*targetDiff*fieldDiff/valDiff

        #Limit step size for stability
        stepSizeLimit = 12
        if fieldStep > stepSizeLimit:
            print(f'Warning: Step size limited to {stepSizeLimit}.')
            newField = curField+stepSizeLimit

        #Ensure new field is at least 2
        elif fieldStep < 2:
            print(f'Warning: Field step small. Using heuristic step of 2.')
            newField = curField + 2

        #Round step up to nearest integer
        else:
            roundedStep = math.ceil(fieldStep)
            newField = curField+roundedStep

        return newField

#**********************************************************************#
    def _getNextField(self, iterNo, valueAtField, targetValue):
        """
        Determines the next field ratio for achieving a target value. 
        Utilizes the iteration number to choreograph a secant-based root-finding method.
        
        Args:
            iterNo (int): Iteration number.
            valueAtField (dict): Dictionary containing field and value information:
                - 'field': Array of previous field ratios.
                - 'value': Array of previous values.
                - 'valueErr': Array of previous value errors.
            verbose (bool): Optional parameter for displaying the intermediate results

        Returns:
            float: Calculated field ratio for target value
        """

        #Identify data key
        dataKeys = [k for k in valueAtField.keys() if k != 'field' and not k.endswith('Err')]
    
        if not dataKeys:
            raise KeyError("Could not find a valid data key in the dictionary.")
        
        valueKey = dataKeys[0]
        errorKey = f'{valueKey}Err'

        # Determine new field strength
        newField = None

        if iterNo == 1:
            newField = self._getParam('fieldRatio')

        # Take constant step of 2 for 2nd iteration
        elif iterNo == 2:
            newField = valueAtField['field'][0] + 2

        # Use secant method to determine new field
        else:
            newField = self._getFieldRatio( 
                fields=np.array([valueAtField['field'][-1], valueAtField['field'][-2]]),
                values=np.array([valueAtField[valueKey][-1], valueAtField[valueKey][-2], targetValue]),
                valuesErr=np.array([valueAtField[errorKey][-1], valueAtField[errorKey][-2]])
            )
        
        if newField is None:
            raise ValueError('Error: Invalid new field')
        
        return newField

#**********************************************************************#
    def _findFieldForEfficiency(self, targetEfficiency=.95, threshold=10):
        """
        Performs an iterative search to find the minimum Electric Field Ratio 
        required to achieve a specified detection efficiency for electron avalanches.

        Process:
            1 - Solves an electric field using Elmer. 
            2 - Executes Garfield++ avalanches to determine a detection efficiency.
            3 - Repeats steps 1 and 2, increasing the field ratio using a modified 
                secant method until a solution is reached.

        Note that this assumes that the field strength is monotonically increasing.

        Args:
            targetEfficiency (float): The target electron detection efficiency.
            threshold (int): The minimum avalanche size required for an event to 
                             be counted as 'detected'.

        Returns:
            float: The minimum field ratio required to achieve the target efficiency.
                   This value is also loaded into the class parameters upon completion.
        """

        #Ensure all parameters exist and save them
        self._checkParam()
        saveParam = self.param.copy()
        
        print(f'Beginning search for minimum field to achieve {targetEfficiency*100:.0f}% efficiency...')
        
        iterNo = 0
        iterNoLimit = 20

        efficiencyAtField = {
            'field': [],
            'efficiency': [],
            'efficiencyErr': []
        }        

        validEfficiency = False

        self.param['numAvalanche'] = 3000
        self.param['avalancheLimit'] = 20 #Limit can be low. Check is boolean - above threshold or not
        
        while not validEfficiency:

            iterNo += 1
            if iterNo > iterNoLimit:
                print(f'Warning - Iteration limit reached ({iterNoLimit})')
                break

            #Solve for the new electric field
            newField = self._getNextField(iterNo, efficiencyAtField, targetEfficiency)
            efficiencyAtField['field'].append(newField)
            self.param['fieldRatio'] = newField
            self._solveEField()

            #Determine the efficiency and read results from file
            self._runGetEfficiency(targetEfficiency=targetEfficiency, threshold=threshold)
            effResults = self._readEfficiencyFile()

            efficiencyAtField['efficiency'].append(effResults['efficiency'])
            efficiencyAtField['efficiencyErr'].append(effResults['efficiencyErr'])


            match effResults['stopCondition']:
                
                case 'DID NOT CONVERGE':
                    validEfficiency = False
                    print(f'Did not converge at field ratio = {efficiencyAtField['field'][-1]}:')
                    print(f'\tEfficiency = {efficiencyAtField['efficiency'][-1]:.3f} +/- {efficiencyAtField['efficiencyErr'][-1]:.3f}')

                case 'CONVERGED':
                    validEfficiency = True
                    print(f'Converged at field ratio = {efficiencyAtField['field'][-1]}:')
                    print(f'\tEfficiency = {efficiencyAtField['efficiency'][-1]:.3f} +/- {efficiencyAtField['efficiencyErr'][-1]:.3f}')
            
                case 'EXCLUDED':
                    validEfficiency = False
                    print(f'Excluded at field ratio = {efficiencyAtField['field'][-1]}:')
                    print(f'\tEfficiency = {efficiencyAtField['efficiency'][-1]:.3f} +/- {efficiencyAtField['efficiencyErr'][-1]:.3f}')

                case _:
                    raise ValueError('Error - Malformed efficiency file.')      
        #End of find field for efficiency loop

        #Print solution
        finalField = self._getParam('fieldRatio')
        print(f'Solution found for {targetEfficiency*100:.0f}% efficiency: Field ratio = {finalField}')
        #print(efficiencyAtField)
        
        #Reset parameters
        self.resetParam()
        #load saved parameters back into class.
        self.param = saveParam
        self.param['fieldRatio'] = finalField
        self._writeParam()

        return finalField

#**********************************************************************#
    def _findFieldForTransparency(self, targetTransparency=0.99):
        """
        Runs simulations to determine what the minimum electric field ratio
        needs to be in order to have >99% E-field transparency.

        Process:
            1 - Solves an electric field using Elmer. 
            2 - Executes Garfield++ to draw field lines to determine a transparency.
            3 - Repeats steps 1 and 2, increasing the field ratio until a solution is reached.

        Note that this assumes that the field strength is monotonically increasing.

        Args:
            targetTransparency (float): The target electric field line transparency.

        Returns:            
            float: The minimum field ratio required to achieve the 100% transparency.
                   This value is also loaded into the class parameters upon completion.
        """

        #Ensure all parameters exist and save them
        self._checkParam()
        saveParam = self.param.copy()

        print(f'Beginning search for minimum field to achieve >{targetTransparency}% transparency...')
       
        iterNo = 0
        iterNoLimit = 10

        transparencyAtField = {
            'field': [],
            'transparency': [],
            'transparencyErr': []
        } 

        isTransparent = False
        self.param['numFieldLine'] = 200
        
        while not isTransparent:

            iterNo += 1
            if iterNo > iterNoLimit:
                print(f'Warning - Iteration limit reached ({iterNoLimit})')
                break

            #Solve the new electric field
            newField = self._getNextField(iterNo, transparencyAtField, targetTransparency)
            transparencyAtField['field'].append(newField)
            self.param['fieldRatio'] = newField
            self._solveEField()

            #Generate field lines and read results from file
            self._runFieldLines()  
            transparencyResults = self._readTransparencyFile()

            transparencyAtField['transparency'].append(transparencyResults['transparency'])
            transparencyAtField['transparencyErr'].append(transparencyResults['transparencyErr'])
                

            #Check transparency to terminate loop
            if transparencyAtField['transparency'][-1] >= targetTransparency:
                isTransparent = True
                print(f'Transparent at field ratio = {transparencyAtField['field'][-1]}:')
                print(f'\tTransparency = {transparencyAtField['transparency'][-1]:.3f} +/- {transparencyAtField['transparencyErr'][-1]:.3f}')
            else:
                isTransparent = False
                print(f'Not transparent at field ratio = {transparencyAtField['field'][-1]}:')
                print(f'\tTransparency = {transparencyAtField['transparency'][-1]:.3f} +/- {transparencyAtField['transparencyErr'][-1]:.3f}')
        #End of find field for transparency loop


        #Print solution
        finalField = self._getParam('fieldRatio')
        print(f'Solution: Field ratio = {finalField}')
        #print(transparencyAtField)
        
        #Reset parameters
        self.resetParam()
        #load saved parameters back into class.
        self.param = saveParam
        self.param['fieldRatio'] = finalField
        self._writeParam()

        return finalField

#**********************************************************************#
    def findMinFieldRatio(self):
        """
        Determines the minimum Electric Field Ratio required to achieve:   
        - 95% detection efficiency, and
        - 100% electric field line transparency.

        Returns:
            float: The minimum field ratio that satisfies both conditions.
                   This value is also loaded into the class parameters upon completion.
        """

        #Ensure all parameters exist and save them
        self._checkParam()
        saveParam = self.param.copy()

        #Get absolute drift field value
        driftField = self._getParam('driftField')
        print(f'Finding minimum field ratio for geometry with drift field: {driftField} V/cm')

        #Choose initial field ratio guess
        minFieldGuess = self._calcMinField()
        self.param['fieldRatio'] = minFieldGuess
        print(f'\tInitial field ratio guess: {minFieldGuess}')

        #Generate the FEM geometry
        self._generateGeometry()

        ## Determine minimum field ratio for conditions
        # 95% efficiency
        efficiencyField = self._findFieldForEfficiency()
        print('************************************************************')
        print(f'\tMinimum field ratio for 95% efficiency: {efficiencyField}')
        print('************************************************************')
        
        # 100% transparency
        transparencyField = self._findFieldForTransparency()
        print('************************************************************')
        print(f'\tMinimum field ratio for 100% transparency: {transparencyField}')
        print('************************************************************')
        
        # Set final field ratio to the maximum of the two found
        finalField = max(efficiencyField, transparencyField)
        self.param['fieldRatio'] = finalField
        print(f'Solution for minimum field ratio: {finalField}')
        print('************************************************************')
        
        return finalField

#**********************************************************************#
    def findMinFieldRatioAlt(self, targetEfficiency=.95, targetTransparency=.99, threshold=10):
        #TODO: verify that this works
        """
        Performs an iterative search to find the minimum Electric 
        Field Ratio required to achieve a specified detection 
        efficiency for electron avalanches and a given target for the
        Electric Field Transparency

        Process:
            1 - Solves an electric field using Elmer. 
            2 - Executes Garfield++ avalanches to determine a detection
                    efficiency.
            3 - Executes Garfield++ field line generator to determine 
                    the transparency.
            4 - Repeats steps 1, 2, and 3, increasing the field ratio 
                    using a modified secant method until a solution 
                    is reached.

        Note that this assumes that the field strength is monotonically 
        increasing.

        Args:
            targetEfficiency (float): The target electron detection 
                    efficiency.
            targetTransparency (float): The target Efield transparency.
            
            threshold (int): The minimum avalanche size required for an
                    event to be counted as 'detected'.

        Returns:
            float: The minimum field ratio required to achieve both targets.
                   This value is also loaded into the class parameters upon completion.
        """

        #Ensure all parameters exist and save them
        self._checkParam()
        saveParam = self.param.copy()
        
        print('Beginning search for minimum field to achieve:\n'
                f'\t{targetEfficiency*100:.0f}% efficiency\n'
                f'\t{targetTransparency*100}% transparency\n')
        
        iterNo = 0
        iterNoLimit = 30

        valuesAtField = {
            'field': [],
            'efficiency': [],
            'efficiencyErr': [],
            'transparency': [],
            'transparencyErr': []
        }        

        isTransparent = False
        validEfficiency = False

        self.param['numFieldLine'] = 200
        self.param['numAvalanche'] = 3000
        self.param['avalancheLimit'] = 20 
        
        while not isTransparent or not validEfficiency:

            iterNo += 1
            if iterNo > iterNoLimit:
                print(f'Warning - Iteration limit reached ({iterNoLimit})')
                break

            ## Solve the new electric field
            #Estimated transparency step
            newFieldTrans = self._getNextField(iterNo, transparencyAtField, targetTransparency)
            
            #Estimated efficiency step
            newFieldEff = self._getNextField(iterNo, efficiencyAtField, targetEfficiency)
            
            #Take the larger of the two steps as the new electric field
            newField = max(newFieldEff, newFieldTrans)
            valuesAtField['field'].append(newField)
            self.param['fieldRatio'] = newField
            self._solveEField()

            #Generate field lines and read results from file
            self._runFieldLines()  
            transparencyResults = self._readTransparencyFile()

            valuesAtField['transparency'].append(transparencyResults['transparency'])
            valuesAtField['transparencyErr'].append(transparencyResults['transparencyErr'])
                
            #Determine the efficiency and read results from file
            self._runGetEfficiency(targetEfficiency=targetEfficiency, threshold=threshold)
            effResults = self._readEfficiencyFile()

            valuesAtField['efficiency'].append(effResults['efficiency'])
            valuesAtField['efficiencyErr'].append(effResults['efficiencyErr'])


            #Check transparency condition
            if valuesAtField['transparency'][-1] >= targetTransparency:
                isTransparent = True
                print(f"Transparent at field ratio = {valuesAtField['field'][-1]}: ")
                print(f"\tTransparency = {valuesAtField['transparency'][-1]:.3f} "
                         f"+/- {valuesAtField['transparencyErr'][-1]:.3f}")
            else:
                isTransparent = False
                print(f"Not transparent at field ratio = {valuesAtField['field'][-1]}: ")
                print(f"\tTransparency = {valuesAtField['transparency'][-1]:.3f} "
                         f"+/- {valuesAtField['transparencyErr'][-1]:.3f}")
       
            match effResults['stopCondition']:
                
                case 'DID NOT CONVERGE':
                    validEfficiency = False
                    print(f"Did not converge at field ratio = {valuesAtField['field'][-1]}: ")
                    print(f"\tEfficiency = {valuesAtField['efficiency'][-1]:.3f} "
                            f"+/- {valuesAtField['efficiencyErr'][-1]:.3f}")

                case 'CONVERGED':
                    validEfficiency = True
                    print(f"Converged at field ratio = {valuesAtField['field'][-1]}: ")
                    print(f"\tEfficiency = {valuesAtField['efficiency'][-1]:.3f} "
                            f"+/- {efficiencyAtField['efficiencyErr'][-1]:.3f}")
            
                case 'EXCLUDED':
                    validEfficiency = False
                    print(f"Excluded at field ratio = {valuesAtField['field'][-1]}: ")
                    print(f"\tEfficiency = {valuesAtField['efficiency'][-1]:.3f} "
                            f"+/- {efficiencyAtField['efficiencyErr'][-1]:.3f}")

                case _:
                    raise ValueError('Error - Malformed efficiency file.')
       
        #Print solution
        finalField = self._getParam('fieldRatio')
        print('*************************************************')
        print(f'Solution: Field ratio = {finalField}')
        print('*************************************************')
        
        #Reset parameters
        self.resetParam()
        #load saved parameters back into class.
        self.param = saveParam
        self.param['fieldRatio'] = finalField
        self._writeParam()

        return finalField
    
#**********************************************************************#
    def _generateGeometry(self):
        """
        Generates the finite-element mesh of the simulation geometry using gmsh.
        """

        self._checkParam()
        self._writeParam()
        self._runGmsh()

        return
    
#**********************************************************************#
    def _solveEField(self):
        """
        Solves the electric field in the simulation geometry using elmer.
        """
        
        self._checkParam()
        self._writeParam()
        self._runElmer()

        return
    
#**********************************************************************#
    def _solveWeightingField(self):
        """
        Solves the weighting field in the simulation geometry using elmer.
        """
        
        self._checkParam()
        self._writeParam()
        self._runElmerWeighting()

        return
    
#**********************************************************************#
    def _runElectronTransport(self):
        """
        Runs the electron transport simulation using Garfield++.
        """
        
        self._checkParam()
        self._writeParam()
        self._runGarfield()

        return
    
#**********************************************************************#
    def runForOptimizer(self):
        """
        Executes a full avalanche simulation of the FIMS geometry 
        for efficient running within an optimizer.

        Determines the electric field ratio required for
        95% efficiency and 100% transparency.
            Note this generates the geometry and solves the electric field.
        Solves the weighting field and runs the electron transport simulation.
            
        Returns:
            int: The run number of the simulation that was executed.
        """

        #Check and save parameters
        self._checkParam()

        #get the run number for this simulation
        runNo = self._getRunNumber()
        print(f'Running simulation - Run number: {runNo}')

        #Find the minimum field ratio for this geometry that satisfies:
        #  95% efficiency and 100% transparency
        minField = self.findMinFieldRatio()

        #Solve for the weighting field
        self._solveWeightingField()
        
        #Run the electron transport simulation
        self._runElectronTransport()
        
        #Reset parameters
        self.resetParam()
        
        return runNo


#**********************************************************************#
#**********************************************************************#
# METHODS FOR RUNNING CHARGE BUILDUP - UNTESTED (TODO)
#**********************************************************************#
#**********************************************************************#


#**********************************************************************#
    def resetCharge(self):
        """
        Resets the charge buildup file to be empty.
        """
        filename = 'Geometry/chargeBuildup.dat'

        try:
            with open(filename, 'w') as file:
                file.write('')
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File '{filename}' not found.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while resetting the charge file '{filename}': {e}")
        
        return
    
#**********************************************************************#
    def _saveCharge(self, runNumber):
        """
        Saves the surface charge buildup to a file designated by runNumber.

        Copies the current charge buildup file into 'savedCharge/' as 'runXXXX.charge.dat'

        Args:
            runNumber (int): Run identifier for saved file.
        """
        chargeDirectory = 'savedCharge'
        if not os.path.exists(chargeDirectory):
            try:
                os.makedirs(chargeDirectory)
            except OSError as e:
                raise RuntimeError(f"An error occurred while creating directory '{chargeDirectory}': {e}")
                
        saveFile = f'run{runNumber:04d}.charge.dat'
        saveFilePath = os.path.join('savedCharge', saveFile)

        filename = 'Geometry/chargeBuildup.dat'
        try:
            shutil.copyfile(filename, saveFilePath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File '{filename}' not found.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while saving the charge file '{filename}': {e}")
        
        return


#**********************************************************************#
    def _readCharge(self):
        """
        Reads the file containing the built-up surface charge distribution.

        Assumes a space-separated dataset.

        Returns:
            dataframe: Pandas dataframe containing: x, y, z, and charge density.
                       None if no data is available.
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
                
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the charge file '{filename}': {e}")

        return chargeData

#**********************************************************************#
    def _writeCharge(self, builtUpCharge):
        """
        Writes the built-up surface charge to a file.

        Args:
            dataframe: Pandas dataframe containing: x, y, z, and charge density.
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
        except Exception as e:
            raise RuntimeError(f"An error occurred while writing the charge file '{filename}': {e}")
        
        return
    

#**********************************************************************#
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

#**********************************************************************#
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


#**********************************************************************#
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
        """
        self._checkParam()
        
        #Check that the number of avalanches is a reasonable number
        # too many = too much charge unaffected by new stucks
        # too few = not enough stuck charges to make a difference
        numAvalanche = self.param('numAvalanche')
        if ((numAvalanche < 10) | (numAvalanche > 100)):
            raise ValueError('Error: numAvalanche should be between 10 and 100 for charge buildup simulations.')
        
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
            self._writeParam()
            
            #Get the simulation data
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
        
