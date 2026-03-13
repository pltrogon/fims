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

from geometryClass import geometryClass

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

    Private Attributes:
        param (dict): Parameter dictionary with the following entries:
            All dimensions are in microns. Electric field is in V/cm.
            - padLength: Length of the side of the hexagonal pad.
            - pitch: Distance between neighboring pads.
            - gridStandoff: Distance from the top to the SiO2 layer to the bottom of the grid.
            - gridThickness: Thickness of the grid.
            - holeRadius: Radius of the hole in the grid.
            - cathodeHeight: Distance from the top to the grid to the cathode plane.
            - thicknessSiO2: Thickness of the SiO2 layer.
            - pillarRadius: The radius of the insulating support pillars.
            - driftField: The strength of the electric field in the drift region.
            - fieldRatio: Ratio of the amplification field to the drift field.
            - numFieldLine: Number of field lines to calculate for visualization.
            - numAvalanche: Number of electrons (avalanches) to initiate
            - avalancheLimit: Limit of the number of electrons within a single avalanche.
            - gasCompAr: Percentage of Argon within gas volume.
            - gasCompCO2: Percentage of CO2 within gas volume.
            - gasCompCF4: Percentage of CF4 within gas volume.
            - gasCompIsobutane: Percentage of Isobutane within gas volume.
            - gasPenning: Penning transfer rate for the gas mixture.

        GARFIELDPATH (str): Filepath to the Garfield++ source script. Read from 'GARFIELDPATH' file.

        geoemetry (geometryClass): A geometry class object representing the geometry and fields solvers.
    """

#***********************************************************************************#
    def __init__(self):
        """
        Initializes a FIMS_Simulation object.
        """
        try:
            self._param = self._defaultParam()
            self._checkParam()
        except KeyError:
            raise RuntimeError('Error initializing parameters.')

        self._GARFIELDPATH = self._getGarfieldPath()
        if self._GARFIELDPATH is None:
            raise RuntimeError('Error getting Garfield++ path.')

        self._setupSimulation()

        self._geometry = None

        return

#***********************************************************************************#
    #String definition
    def __str__(self):
        """
        Returns a formatted string containing all of the simulation parameters.
        """
        paramList = [f'{key}: {value}' for key, value in self._param.items()]   

        return "FIMS Simulation Parameters:\n\t" + "\n\t".join(paramList)

#***********************************************************************************#    
    def _defaultParam(self):
        """
        Default FIMS parameters.
            Dimensions in microns.
            Electric field in V/cm.

        Returns:
            dict: Dictionary of default parameters and values.
        """
        defaultParameters = {
            'padLength': 12.,
            'pitch': 55.,
            'gridStandoff': 50.,
            'gridThickness': 1.,
            'holeRadius': 20.,
            'cathodeHeight': 200.,
            'thicknessSiO2': 5.,
            'pillarRadius': 2.,
            'driftField': 280.,
            'fieldRatio': 80.,
            'numFieldLine': 25,
            'numAvalanche': 2000,
            'avalancheLimit': 600,
            'gasCompAr': 0.95,
            'gasCompCO2': 0.00,
            'gasCompCF4': 0.03,
            'gasCompIsobutane': 0.02,
            'gasPenning': 0.385
        }
        return defaultParameters

#***********************************************************************************#    
    def _checkParam(self):
        """
        Ensures that values exist for all necessary parameters.
        """
        #Check that any parameters exist
        if self._param is None or not self._param:
            raise(KeyError('Parameter dictionary is empty.'))

        #Check that all parameters are present
        allParam = self._defaultParam()
        for inParam in allParam:
            if inParam not in self._param:
                raise KeyError(f"Parameter '{inParam}' missing from parameter dictionary.")
            
        #Check gas composition 
        self._checkGasComp()
                
        return
    
    #***********************************************************************************#    
    def setParameters(self, paramDict):
        """
        Updates the parameter dictionary with the provided values.

        Args:
            paramDict (dict): A dictionary containing parameter names and their values.
        """

        for inParam, inValue in paramDict.items():
            if inParam not in self._param:
                raise KeyError(f"Error - Invalid parameter: {inParam}.")
            if inValue < 0:
                raise ValueError(f'Error - {inParam} cannot be negative.')
            
            self._param[inParam] = inValue

        self._checkParam()

        return
    
#***********************************************************************************#    
    def _checkGasComp(self):
        """
        Ensures that the gas composition percentages sum to 1.0.
        """
        totalComp = (
            float(self.getParam('gasCompAr')) +
            float(self.getParam('gasCompCO2')) +
            float(self.getParam('gasCompCF4')) +
            float(self.getParam('gasCompIsobutane'))
        )
        
        if not math.isclose(totalComp, 1.0, rel_tol=1e-3):
            raise ValueError(f'Gas composition percentages must sum to 1.0. Current sum: {totalComp}')
        
        return

#***********************************************************************************#
    def getParam(self, parameter):
        """
        Gets a copy of the desired parameter.
        """
        if parameter not in self._param:
            raise KeyError(f'Error - Invalid parameter: {parameter}.')
            
        return copy.copy(self._param[parameter])

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
                    print(f"Error: File 'setupGarfield.sh' not found at '{garfieldPath}'.")
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

#***********************************************************************************#
    def _setupSimulation(self):
        """
        Initializes Garfield++ and creates an avalanche executable.
        
        Reads the Garfiled++ source path, and ensures a log and build directory.
        Compiles the executable using cmake and make.
        Initializes a simulation run counter if it does not already exist.
    
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

#***********************************************************************************#

## TODO - This form may no longer be necessary with geometry class. 
## Check if can be better aligned with garfield++ file needs.
    def _makeRunControl(self):
        """
        Creates a runControl file with the parameters.
        """

        writeParam = self._param.copy()

        runControlLines = [
            '// FIMS Simulation Control File //',
            '// DO NOT EDIT THIS FILE MANUALLY //',
            '',
            '// Assumes the form "variable = value;" for each line.',
            '// Number of input parameters (numInputs included):',
            f'numInputs = {len(writeParam)+1};',
            '',
            '//----- Geometry parameters -----//',
            '// Dimensions in microns.',
            '',
            '// Pad and pitch',
            f'padLength = {writeParam['padLength']:.1f};',
            f'pitch = {writeParam['pitch']:.1f};',
            '',
            '// Grid',
            f'gridStandoff = {writeParam['gridStandoff']:.1f};',
            f'gridThickness = {writeParam['gridThickness']:.1f};',
            f'holeRadius = {writeParam['holeRadius']:.1f};',
            '',
            '// Other',
            f'cathodeHeight = {writeParam['cathodeHeight']:.1f};',
            f'thicknessSiO2 = {writeParam['thicknessSiO2']:.1f};',
            f'pillarRadius = {writeParam['pillarRadius']:.1f};',
            '',
            '//----- Electric field parameters -----//',
            '// Electric field in V/cm.',
            f'driftField = {writeParam['driftField']:.1f};',
            f'fieldRatio = {writeParam['fieldRatio']:.1f};',
            f'numFieldLine = {writeParam['numFieldLine']};',
            '',
            '//----- Simulation parameters -----//',
            '// Avalanche controls',
            f'numAvalanche = {writeParam['numAvalanche']};',
            f'avalancheLimit = {writeParam['avalancheLimit']};',
            '',
            '// Gas composition (in percentage)',
            f'gasCompAr = {writeParam['gasCompAr']:.2f};',
            f'gasCompCO2 = {writeParam['gasCompCO2']:.2f};',
            f'gasCompCF4 = {writeParam['gasCompCF4']:.2f};',
            f'gasCompIsobutane = {writeParam['gasCompIsobutane']:.2f};',
            '',
            f'gasPenning = {writeParam['gasPenning']:.3f};',
            '',
            '// End of runControl file\n'
        ]

        runControlInfo = '\n'.join(runControlLines)
        try:
            with open('runControl', 'w') as file:
                file.write(runControlInfo)
        except Exception as e:
            raise RuntimeError(f"An error occurred while writing to the file 'runControl': {e}")
        
        return

#***********************************************************************************#
    def _resetParam(self, verbose=True):
        #TODO - This may no longer be necessary????????
        """
        Rewrites the run control files with the default simulation parameters.
    
        Args:
            verbose (bool): Option available to supress reset notification.
        """
        self._param = self._defaultParam()
    
        if verbose:
            print('Parameters have been reset.')
        
        return

#***********************************************************************************#
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

        #TODO - Generic exception handling?
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File '{filename}' not found.")
        except ValueError:
            raise ValueError(f"Error: Invalid number format in '{filename}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the file '{filename}': {e}")
        
        return runNo

#***********************************************************************************#
    def _generateGeometry(self):
        """
        Generates the geometry for the FIMS simulation using the geometryClass.
        """
        self._geometry = geometryClass(self._param)
        self._geometry.createGeometry()

        return
    
#***********************************************************************************#
    def _solveEFields(self, solveWeighting=True):
        """
        Solves the electric field for the FIMS simulation using the geometryClass.

        Creates the geometry if it does not already exist, and then solves the E field for the mesh.
        
        Args:
            solveWeighting (bool): Option to additonally solve for weighting fields.

        """
        if not hasattr(self, '_geometry'):
            self._generateGeometry()

        self._geometry.calculateEFields(solveWeighting=solveWeighting)

        return

#***********************************************************************************#
    def _runGarfield(self):
        """
        Runs a Garfield++ executable to determine field lines and simulate 
        electron avalanches based on the parameters in 'runControl'.
    
        First links garfield libraries, creates the executable, and then runs the simulation.
        The simulation is numbered based on the number found in 'runNo';
        This also incremenmts this number.
        The information from this simulation is saved in .root format within 'Data/'.

        Args:
            surrounding (bool): Option to run Garfield for the surrounding geometry.
        """
        originalCWD = os.getcwd()

        executable = 'runAvalanche'
        
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
                    f'./{executable}'
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
    

#***********************************************************************************#
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
    

#***********************************************************************************#
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
    
#***********************************************************************************#
    def runSimulation(self, changeGeometry=True):
        """
        Executes a full simulation run with the current parameters.

        This process is as follows:
            1. Check that all required parameters are present and defined.
            2. Read and the simulation run number.
            3. Write the simulation parameters to the control files.
            4. Execute Gmsh to generate a finite-element mesh of the geometry.
            5. Execute Elmer to solve the E field for the mesh.
            6. Execute Elmer to solve the weighting field for the electrode.
            7. Execute Garfield++ to simulate electron multiplication effects.
            8. Reset parameters to defaults.

        Note that if 'changeGeometry' is False, step 4 is skipped. 
        This can be used to save time when only adjusting field parameters.

        Args:
            changeGeometry (bool): Option to skip geometry generation 
                                   if geometry parameters are unchanged.

        Returns:
            int: The run number for this simulation.
        """
    
        self._checkParam()
        self._makeRunControl()
    
        # If geometry does not change, can skip Gmsh.
        if changeGeometry or self._geometry is None:
            self._generateGeometry()
        else:
            print('Warning: Skipping Gmsh. (Ignore if geometry unchanged.)')
            
        #get the run number for this simulation
        runNo = self._getRunNumber()
        print(f'Running simulation - Run number: {runNo}')

        #Solve fields and run Garfield
        self._solveEFields()
        self._runGarfield()
    
        #TODO - is resetting parameters necessary anymore? 
        self._resetParam()
        
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
    

#***********************************************************************************#
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
    
    
#***********************************************************************************#
    def _getFieldRatioSecant(self, fields, values, valuesErr=None, damping=0.8):
        """
        Calculates the next field strength ratio using the secant method to approach a target value.

        If errors are provided, utilizes a step base on the maximum possible slope.
        Assumes that the value is monotonically increasing for field ratios.

        The resulting field ratio is rounded up to the nearest integer.
        
        *TODO - Tested for when  current and prevoious values are both LESS than target. Behaviour when bracketing target unknown.*

        Args:
            fields (np.array): Numpy array of the field ratio. Has form: [current, previous]
            valus (np.array): Numpy array of the values. Has form: [current, previous, target]
            valuesErr (np.array): Optional. Array of errors in the values. Has form: [currentError, previousError]
            damping (float): Damping factor for the secant method to avoid too large of steps.

        Returns:
            float: Numerical solution to the field ratio in order to achieve target value
        """

        curField, prevField = fields
        curEff, prevEff, targetEff = values

        fieldDiff = curField - prevField

        #If errors exist, use the maximum possible slope
        if valuesErr is not None:
            curEffErr, prevEffErr = valuesErr
            prevEffMin = prevEff - prevEffErr
            curEffMax = curEff + curEffErr

            effDiff = curEffMax - prevEffMin
            targetDiff = targetEff - curEffMax

        else:
            effDiff = curEff - prevEff
            targetDiff = targetEff - curEff


        if abs(effDiff) < 0.001:
                print(f'Warning: Slope near zero. Using heuristic step of 1.')
                return curField+1

        fieldStep = damping*targetDiff*fieldDiff/effDiff

        #Limit step size for stability
        stepSizeLimit = 10
        if fieldStep > stepSizeLimit:
            print(f'Warning: Step size limited to {stepSizeLimit}.')
            newField = curField+stepSizeLimit

        #Ensure step is at least 1
        elif fieldStep < 1:
            print(f'Warning: Field step small. Using heuristic step of 1.')
            newField = curField+1

        #Round step up to nearest integer
        else:
            roundedStep = math.ceil(fieldStep)
            newField = curField+roundedStep

        return newField
    

#***********************************************************************************#
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
            newField = self.getParam('fieldRatio')

        # Take constant step of 2 for 2nd iteration
        elif iterNo == 2:
            newField = valueAtField['field'][0] + 2

        # Use secant method to determine new field
        else:
            newField = self._getFieldRatioSecant( 
                fields=np.array([valueAtField['field'][-1], valueAtField['field'][-2]]),
                values=np.array([valueAtField[valueKey][-1], valueAtField[valueKey][-2], targetValue]),
                valuesErr=np.array([valueAtField[errorKey][-1], valueAtField[errorKey][-2]])
            )
        
        if newField is None:
            raise ValueError('Error: Invalid new field')
        
        return newField

#***********************************************************************************#
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
        
        print(f'Beginning search for minimum field to achieve {targetEfficiency*100:.0f}% efficiency...')
        
        # Ensure geometry exists
        if self._geometry is None:
            self._generateGeometry()
        
        iterNo = 0
        iterNoLimit = 20

        efficiencyAtField = {
            'field': [],
            'efficiency': [],
            'efficiencyErr': []
        }        

        validEfficiency = False
        self._param['numAvalanche'] = 3000
        self._param['avalancheLimit'] = 20 #Limit can be low. Check is boolean
        
        while not validEfficiency:

            iterNo += 1
            if iterNo > iterNoLimit:
                print(f'Warning - Iteration limit reached ({iterNoLimit})')
                break

            #Solve for the new electric field
            newField = self._getNextField(iterNo, efficiencyAtField, targetEfficiency)
            efficiencyAtField['field'].append(newField)
            self._param['fieldRatio'] = newField
            self._solveEFields(solveWeighting=False)

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
        finalField = self.getParam('fieldRatio')
        print(f'Solution found for {targetEfficiency*100:.0f}% efficiency: Field ratio = {finalField}')
        print(efficiencyAtField)

        self._param['fieldRatio'] = finalField

        return finalField

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
        standoff = self.getParam('gridStandoff')
        pad = self.getParam('padLength')
        pitch = self.getParam('pitch')
    
        optTrans = self._calcOpticalTransparency()
        
        standoffRatio = standoff/pitch
        padRatio = pad/pitch
        
        #Do calculation using values from fits
        '''Values come from exponential fits to data - 
            grid standoff ratio, pad length ratio, and 
            optical transparency vs minField.
          Results are then added together.'''
        radialMinField = 570.580*np.exp(-12.670*optTrans)
        standoffMinField = 27.121*np.exp(-15.9*standoffRatio)
        padMinField = 143.84*np.exp(-15.17*padRatio)
        
        #TODO for James - should this not be max(minFields)?
        minField = radialMinField + standoffMinField + padMinField + 3

        #Round down to nearest integer
        minFieldRounded = math.floor(minField)
        
        return minFieldRounded
    
#***********************************************************************************#
    def _calcOpticalTransparency(self):
        """
        Calculates the optical transparency of the grid based on the geometry parameters.

        Returns:
            float: The optical transparency of the grid.
        """ 

        #Get geometry variables
        radius = self.getParam('holeRadius')
        pitch = self.getParam('pitch')
    
        gridArea = pitch**2*math.sqrt(3)/2
        holeArea = math.pi*radius**2

        opticalTransparency = holeArea/gridArea

        return opticalTransparency


#***********************************************************************************#
    def _findFieldForTransparency(self, targetTransparency=0.99):
        """
        ## TODO - The garfield code may be able to do the 2-sigma check instead of here.
        ##Should be able to  do the same while-loop doing "bunches" of field lines
    
        Runs simulations to determine what the minimum electric field ratio
        needs to be in order to have >99% E-field transparency.

        Process:
            1 - Solves an electric field using Elmer. 
            2 - Executes Garfield++ to draw field lines to determine a transparency.
            3 - Repeats steps 1 and 2, increasing the field ratio using a modified 
                secant method until a solution is reached.

        Note that this assumes that the field strength is monotonically increasing.

        Args:
            targetTransparency (float): The target electric field line transparency.

        Returns:            
            float: The minimum field ratio required to achieve the 100% transparency.
                   This value is also loaded into the class parameters upon completion.
        """

        # Ensure geometry exists
        if self._geometry is None:
            self._generateGeometry()

        print(f'Beginning search for minimum field to achieve >{targetTransparency}% transparency...')
       
        iterNo = 0
        iterNoLimit = 10

        transparencyAtField = {
            'field': [],
            'transparency': [],
            'transparencyErr': []
        } 

        isTransparent = False
        self._param['numFieldLine'] = 200
        
        while not isTransparent:

            iterNo += 1
            if iterNo > iterNoLimit:
                print(f'Warning - Iteration limit reached ({iterNoLimit})')
                break

            #Solve the new electric field
            newField = self._getNextField(iterNo, transparencyAtField, targetTransparency)
            transparencyAtField['field'].append(newField)
            self._param['fieldRatio'] = newField
            self._solveEFields(solveWeighting=False)

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
        finalField = self.getParam('fieldRatio')
        print(f'Solution: Field ratio = {finalField}')
        #print(transparencyAtField)
        
        self._param['fieldRatio'] = finalField

        return finalField
    


#***********************************************************************************#
    def findMinFieldRatio(self):
        """
        Determines the minimum Electric Field Ratio required to achieve:   
        - 95% detection efficiency, and
        - 100% electric field line transparency.

        This assumes that the field strength is monotonically increasing.

        Note that this does not solve for the weighting fields.

        Returns:
            float: The minimum field ratio that satisfies both conditions.
                   This value is also loaded into the class parameters upon completion.
        """

        # Ensure geometry exists
        if self._geometry is None:
            self._generateGeometry()

        #Get absolute drift field value
        driftField = self.getParam('driftField')
        print(f'Finding minimum field ratio for geometry with drift field: {driftField} V/cm')

        #Choose initial field ratio guess
        opticalTransparency = self._calcOpticalTransparency()
        minFieldGuess = .9*(2/opticalTransparency - 1)

        self._param['fieldRatio'] = minFieldGuess
        print(f'\tInitial field ratio guess: {minFieldGuess}')

        ## Determine minimum field ratio for conditions
        # 95% efficiency
        efficiencyField = self._findFieldForEfficiency()
        print(f'\tMinimum field ratio for 95% efficiency: {efficiencyField}')

        # 100% transparency
        transparencyField = self._findFieldForTransparency()
        print(f'\tMinimum field ratio for 100% transparency: {transparencyField}')

        # Set final field ratio to the maximum of the two found
        finalField = max(efficiencyField, transparencyField)

        self._param['fieldRatio'] = finalField
        print(f'Solution for minimum field ratio: {finalField}')

        return finalField
    
#***********************************************************************************#
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


        #get the run number for this simulation
        runNo = self._getRunNumber()
        print(f'Running simulation - Run number: {runNo}')

        #Get the optimal drift field for this gas
        self._param['driftField'] = self._getOptimalDriftField()

        # Generate Geometry
        self._generateGeometry()

        #Find the minimum field ratio for this geometry that satisfies:
        #  95% efficiency and 100% transparency
        minField = self.findMinFieldRatio()
        self._param['fieldRatio'] = minField

        #Solve fields and run Garfield
        self._solveEFields(solveWeighting=True)
        self._runElectronTransport()

        #TODO - check how optimizer runs. is it changing the class elements 
        # or instantiating a new class for each run? 
        # if the latter, may not need to reset parameters here since they will be reloaded for each run.
        self._resetParam()

        
        return runNo
    
#***********************************************************************************#
    def runForOptimizerALT(self, efficiencyGoal=0.95, efficiencyThreshold=10, transparencyGoal=.99):
        """
        ## TODO - This method is outdated with new capabilities. Update or remove as necessary.
        Executes a full avalanche simulation of the FIMS geometry at a given geometry and field ratio.

        Gets the detection efficiency and E field transparency at the current field ratio.
        Executes the electron transport simulation.

        Args:
            efficiencyGoal (float): The target efficiency.
            efficiencyThreshold (int): The minimum number of electrons required to be considered a 'success'.
            transparencyGoal (float): The target transparency.

        Returns:
            tuple containing:
                - int: The run number of the simulation that was executed.
                - float: The detection efficiency at the current field ratio.
                - float: The field transparency at the current field ratio.
        """

        #Check and save parameters
        self._checkParam()
        

        #get the run number for this simulation
        runNo = self._getRunNumber()
        print(f'Running simulation - Run number: {runNo}')

        #Get the optimal drift field for this gas
        self._param['driftField'] = self._getOptimalDriftField()

        #Generate Geometry and solve the Electric and weighting fields
        self._generateGeometry()
        self._solveEField()
        self._solveWeightingField()

        #Get the detection efficiency
        efficiency = self._getEfficiency(efficiencyGoal=efficiencyGoal, efficiencyThreshold=efficiencyThreshold)

        #Get the field transparency
        transparency = self._getTransparency(transparencyGoal=transparencyGoal)

        #Run the electron transport simulation
        self._runElectronTransport()

        #Reset parameters
        self._resetParam()

        return runNo, efficiency, transparency


#***********************************************************************************#
    def _getEfficiency(self, efficiencyGoal=0.95, efficiencyThreshold=10):
        """
        Runs a Garfield++ executable the generates electron avalanches to determine
        the detection efficiency at the current field ratio.

        Args:
            efficiencyGoal (float): The target efficiency.
            efficiencyThreshold (int): The minimum number of electrons required to be considered a 'success'.

        Returns:
            float: The detection efficiency at the current field ratio.
        """

        saveParam = self._param.copy()

        self._param['numAvalanche'] = 3000
        self._param['avalancheLimit'] = 20 #Limit can be low. Check is boolean

        self._makeRunControl()
        self._runGetEfficiency(targetEfficiency=efficiencyGoal, threshold=efficiencyThreshold)
        effResults = self._readEfficiencyFile()

        print(f'\tDetection efficiency: {effResults['efficiency']:.3f} +/- {effResults['efficiencyErr']:.3f}')

        self._param = saveParam

        return effResults['efficiency']
    
#***********************************************************************************#
    def _getTransparency(self, transparencyGoal=0.99):
        """
        Runs a Garfield++ executable to determine field lines and calculate
        the field transparency.

        Args:
            transparencyGoal (float): The target transparency.

        Returns:
            float: The field transparency at the current field ratio.
        """

        saveParam = self._param.copy()

        self._param['numFieldLine'] = 200

        self._makeRunControl()
        self._runFieldLines()  
        transparencyResults = self._readTransparencyFile()

        print(f'\tField transparency: {transparencyResults['transparency']:.3f} +/- {transparencyResults['transparencyErr']:.3f}') 

        self._param = saveParam

        return transparencyResults['transparency']


#***********************************************************************************#
#***********************************************************************************#
# METHODS FOR RUNNING CHARGE BUILDUP - UNTESTED (TODO)
#***********************************************************************************#
#***********************************************************************************#
##TODO - not adjusted with new class implementation. Update as necessary.
#***********************************************************************************#
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
    
#***********************************************************************************#
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


#***********************************************************************************#
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

#***********************************************************************************#
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
        pitch = self.getParam('pitch')        
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
        """
        self._checkParam()
        
        #Check that the number of avalanches is a reasonable number
        # too many = too much charge unaffected by new stucks
        # too few = not enough stuck charges to make a difference
        numAvalanche = self._param('numAvalanche')
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
            saveParam = self._param.copy()
            doneRun = self.runSimulation(changeGeometry=initialRun)
            self._param = saveParam
            
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
        self._resetParam()

        chargeBuildupSummary = {
            'numRuns': numRuns,
            'finalRun': doneRun,
            'totalCharge': totalElectrons,
        }
        return chargeBuildupSummary



        

#***********************************************************************************#
    def _getOptimalDriftField(self):
        """
        Determines the optimal drift field for the current gas mixture in V/cm.

        NOTE: This assumes that the gas composition is a T2K-like mixture of Ar, CF4, and Isobutane.

        It rounds the drift field to the nearest integer value.

        Returns:
            float: The optimal drift field in V/cm.
        """

        self._checkGasComp()

        try:
            allOptimalFieldData = self._loadOptimalDriftFile()

            intAr = round(self.getParam('gasCompAr')*100)
            intCF4 = round(self.getParam('gasCompCF4')*100)
            intIsobutane = round(self.getParam('gasCompIsobutane')*100)

            gasCompFilter = (
                (allOptimalFieldData['Ar'] == intAr) &
                (allOptimalFieldData['CF4'] == intCF4) &
                (allOptimalFieldData['Isobutane'] == intIsobutane)
            )

            #Get optimal field for this gas composition
            optimalFieldData = allOptimalFieldData.loc[gasCompFilter, 'optimalField']
            if optimalFieldData.empty:
                raise ValueError('Error: No optimal drift field data found for the current gas composition.')
            
            #Get field strength in V/cm
            optimalFieldStrength = optimalFieldData.iloc[0]*1000

        except Exception as e:
            print(f'Error loading optimal drift field data: {e}')
            print('Using default optimal drift field of 280 V/cm.')
            optimalFieldStrength = 280

        return round(optimalFieldStrength)


#********************************************************************************#
    def _loadOptimalDriftFile(self):
        """
        Loads pre-computed optimal drift field data from a .pkl file.

        This data can be generated using the gasDataClass.py module.
        """

        filePath = '../Data/Magboltz'
        fileName = f'OptimalDriftFields.myT2K.pkl'
        filename = os.path.join(filePath, fileName)

        optimalFieldData = pd.read_pickle(filename)  

        return optimalFieldData



    

