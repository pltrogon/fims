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
import json
import subprocess
import time
import itertools
import re
import copy

from scipy.optimize import curve_fit


#Include the analysis object        
sys.path.insert(1, '../Analysis')
from runDataClass import runData

# Class to handle numpy data types in JSON serialization
class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumPyEncoder, self).default(obj)

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

        geometry (geometryClass): A geometry class object representing the geometry and field solvers.
    """

#**********************************************************************#
    def __init__(self):
        """
        Initializes a FIMS_Simulation object.
        """
        
        self._GARFIELDPATH = self._getGarfieldPath()
        if self._GARFIELDPATH is None:
            raise RuntimeError('Error getting Garfield++ path.')
        self._setupSimulation()

        try:
            self._param = self._defaultParam()
            self._param['runNumber'] = self._getRunNumber()
            self._checkParam()
        except KeyError:
            raise RuntimeError('Error initializing parameters.')
        
        self._geometry = None
        self._unitCell = 'FIMS'
        self._surroundingCells = False
        self._runMode = 'FIMS'

        self._filenames = None
        
        self._iterationNumberLimit = 100
        self._fieldLimit = 250

        return

#**********************************************************************#
    #String definition
    def __str__(self):
        """
        Returns a formatted string containing all of the simulation parameters.
        """
        paramList = [f'{key}: {value}' for key, value in self._param.items()]   

        return "FIMS Simulation Parameters:\n\t" + "\n\t".join(paramList)

#**********************************************************************#    
    def _defaultParam(self):
        """
        Default FIMS parameters.
            Dimensions in microns.
            Electric field in V/cm.

        Returns:
            dict: Dictionary of default parameters and values.
        """
        defaultParameters = {
            'runNumber': -1,
            'padLength': 15.,
            'pitch': 65.,
            'gridStandoff': 50.,
            'gridThickness': 1.,
            'holeRadius': 16.,
            'cathodeHeight': 200.,
            'thicknessSiO2': 5.,
            'pillarRadius': 5.,
            'driftField': 280.,
            'fieldRatio': 135.,
            'numFieldLine': 25,
            'numAvalanche': 5000,
            'avalancheLimit': 500,
            'gasCompAr': 0.95,
            'gasCompCO2': 0.00,
            'gasCompCF4': 0.03,
            'gasCompIsobutane': 0.02,
            'gasPenning': 0.385
        }
        return defaultParameters

#**********************************************************************#    
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

        #check run Number
        self._checkRunNumber()
                
        return
    
    #******************************************************************#    
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
    
#**********************************************************************#    
    def _checkGasComp(self):
        """
        Ensures that the gas composition percentages sum to 1.0.
        """
        totalComp = (
            float(self._param['gasCompAr'])
            + float(self._param['gasCompCO2'])
            + float(self._param['gasCompCF4'])
            + float(self._param['gasCompIsobutane'])
        )
        
        if not math.isclose(totalComp, 1.0, rel_tol=1e-3):
            raise ValueError(f'Gas composition incomplete. ({totalComp})')
        
        return
    
#**********************************************************************#    
    def _checkRunNumber(self):
        """
        Ensures that the run number is valid.
        """
        runNumber = self.getParam('runNumber')
        if runNumber < 1:
            raise ValueError(f'Error - Invalid run number: {runNumber}.')
        
        return
    
#**********************************************************************#
    def getParam(self, parameter):
        """
        Gets a copy of the desired parameter.

        Args:
            parameter (str): The name of the parameter to retrieve. 

        Returns:
            A copy of the requested parameter.
        """
        if parameter not in self._param:
            raise KeyError(f'Error - Invalid parameter: {parameter}.')
            
        return copy.copy(self._param[parameter])
    
#**********************************************************************#
    def getAllParam(self):
        """
        Gets a copy of the entire parameter dictionary.
        """
        return copy.deepcopy(self._param)

#**********************************************************************#       
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
                    print(f"Error: File '{garfieldPath}' not found.")
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

        # Get garfield path into environment
        newEnv = self._getGarfieldEnvironment()
        os.environ.update(newEnv)
        
        #Make executables
        makeBuild = ('cmake .. && make')

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

        #Check for run number file and create if not present
        if not os.path.exists('runNo'):
            self.setRunNumber()
                
        return
    
#**********************************************************************#
    def _getGarfieldEnvironment(self):
        """
        Sourced the Garfield++ environment via bash.

        Returns:
            dict: Dictionary of environment variables.
        """
        garfieldEnv = {}
        envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'
        
        try:
            envOutput = subprocess.check_output(
                envCommand, 
                shell=True, 
                universal_newlines=True,
                stderr=subprocess.STDOUT
            )
            
            for line in envOutput.strip().splitlines():
                if '=' in line:
                    key, value = line.split('=', 1)
                    garfieldEnv[key.strip()] = value.strip()
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f'ERROR - Failed to source Garfield++: {e.output}'
            )

        return garfieldEnv

#**********************************************************************#
    def setGeometry(self, unitCell='FIMS', surrounding=False):
        """
        Sets the geometry for the simulation.

        Args:
            unitCell (str): The unit cell to use for the geometry.
            surrounding (bool): Option to include surrounding cells in the geometry.
        """

        if unitCell not in ['FIMS', 'GridPix']:
            raise ValueError('Error - Invalid unit cell. Options are "FIMS" and "GridPix".')
        
        self._unitCell = unitCell
        self._surroundingCells = surrounding
        if self._surroundingCells:
            self._runMode = unitCell + 'Surrounding'

        return

#**********************************************************************#
    def _resetParam(self, verbose=True):
        """
        Resets the simulation to the default parameters.
    
        Args:
            verbose (bool): Option available to supress reset notification.
        """
        self._param = self._defaultParam()
    
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

        except Exception as e:
            raise RuntimeError(f'Error with runNo file: {e}')
        
        return runNo

#**********************************************************************#
    def _incrementRunNumber(self):
        """
        Increments the run number after a successful avalanche run.
        """
        try:
            currentRun = self._param['runNumber']
            with open('runNo', 'w') as file:
                file.write(str(currentRun + 1))
        except Exception as e:
            print(f'Warning: Could not increment run number: {e}')
        return

#**********************************************************************#
    def setRunNumber(runNumber=None):
        """
        Sets the simulation run number to a given input value.
        Defaults to 1000 if input not given or if the file is created.

        Args:
            runNumber (int): The run number to set. Must be a non-negative integer.
        """

        if runNumber is None or not os.path.exists('runNo'):
            runNumber = 1000

        if not isinstance(runNumber, int) or runNumber < 0:
            raise ValueError('Error: Run number must be a non-negative integer.')

        try:
            with open('runNo', 'w') as file:
                file.write(str(runNumber))

        except Exception as e:
            print(f'Error occurred while setting run number: {e}')

        return
    
#**********************************************************************#
    def _generateGeometry(self):
        """
        Generates the geometry for the simulation using the geometryClass.
        """
        self._geometry = geometryClass(self._param)
        self._geometry.setUnitCell(self._unitCell)
        self._geometry.setSurroundingCells(self._surroundingCells)
        self._geometry.buildGeometry()

        return        
    
#**********************************************************************#
    def visualizeGeometry(self):
        """
        Generates the geometry for the FIMS simulation 
        and visualizes it using the Gmsh GUI.

        Args:
            unitCell (str): The type of unit cell to use.
            surroundingCells (bool): Whether to include surrounding cells.
        """

        print('Visualizing geometry...')
        self._geometry = geometryClass(self._param)

        self._geometry.setGUI(runGUI=True)
        
        self._geometry.setUnitCell(self._unitCell)
        self._geometry.setSurroundingCells(self._surroundingCells)

        self._geometry.buildGeometry()

        return
    
#**********************************************************************#
    def _solveEFields(self, solveWeighting=True):
        """
        Solves the electric field for the FIMS simulation using the geometryClass.

        Creates the geometry if it does not already exist, and then solves the E field for the mesh.
        
        Args:
            solveWeighting (bool): Additonally solve for weighting fields.

        """
        self._geometry.calculateEFields(solveWeighting=solveWeighting)

        return

#**********************************************************************#
    def _runGarfield(self, executable='runAvalanche', **kwargs):
        #TODO - Can we consolidate any of these executables?
        """
        Runs a Garfield++ simulation with the specified executable.

        Args:
            executable (str): The name of the Garfield++ executable to run. Options are:
                - 'runAvalanche': Simulates electron avalanches for the central pad.

                - 'runFullField': Generates field lines that populate the full unit cell.
                - 'runEfficiency': Simulates the efficiency for a given field strength. Requires additional arguments:
                    - targetEfficiency (str): Name of efficiency to consider (net, detection, collection).
                    - targetValue (float): The target efficiency to achieve (default: 0.95).
                    - threshold (int): The number of electrons to consider an avalanche successful (default: 10).
        """
        #TODOHERE

        executables = [
            'runAvalanche',
            'runEfficiency',
            'runFullField'
        ]

        if executable not in executables:
            raise ValueError(f'Invalid executable: {executable}')
        
        # Handle extra inputs for specific executables
        match executable:

            case 'runEfficiency':
                targetEfficiency = kwargs.get('targetEfficiency', 'net')
                targetValue = kwargs.get('targetValue', 0.95)
                threshold = kwargs.get('threshold', 10)
                args = f'{targetEfficiency} {targetValue} {threshold}'
                
            case _:
                args=''

        originalCWD = os.getcwd()

        try:
            print(f'\tExecuting Garfield++ ({executable})...')
            os.chdir('./build/')

            # Get garfield path into environment
            newEnv = self._getGarfieldEnvironment()
            os.environ.update(newEnv)

            logFile = f'logGarfield{executable}.txt'
            garfieldLog = os.path.join(originalCWD, 'log', logFile)

            # Pass parameters as JSON via stdin
            jsonParam = json.dumps(self._param, cls=NumPyEncoder)

            with open(garfieldLog, 'w+') as garfieldOutput:
                startTime = time.monotonic()
                setupAvalanche = (
                    f'./{executable} {self._runMode} {args}'.strip()
                )
                subprocess.run(
                    setupAvalanche, 
                    input=jsonParam,
                    stdout=garfieldOutput, 
                    shell=True, 
                    check=True,
                    text=True,
                    env=os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')
    
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Garfield++ execution failed with exit code {e.returncode}.')
        
        finally:
            os.chdir(originalCWD)
            if 'runAvalanche' in executable:
                self._incrementRunNumber()

        return  
    
#**********************************************************************#
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
        # Get the run number for this simulation
        runNo = self._param['runNumber']
        print(f'Running simulation - Run number: {runNo}')
    
        self._checkParam()
    
        # If geometry does not change, can skip Gmsh.
        if changeGeometry or self._geometry is None:
            self._generateGeometry()
        else:
            print('Warning: Skipping Gmsh. (Ignore if geometry unchanged.)')
            
        #Solve fields and run Garfield
        self._solveEFields()
        self._runGarfield()
        
        return runNo
    
#**********************************************************************#
    def runSurrounding(self):
        """
        Executes a simulation with the surrounding geometry.

        Determines the induced signal in each adjacent pad.

        Returns:
            int: The run number for this simulation.
        """

        # Get the run number for this simulation
        runNo = self._param['runNumber']
        print('Running Surrounding simulation...')
        print(f'Running simulation - Run number: {runNo}')
    
        self._checkParam()
    
        #Generate geometry for surrounding cells
        self.setGeometry(surrounding=True)
        self._generateGeometry()

        #Solve fields and run Garfield
        self._solveEFields(solveWeighting=True)
        self._runGarfield('runAvalancheSurrounding')
        
        return runNo
    
#**********************************************************************#
    def runGridPix(self):
        """
        Executes a simulation with the GridPix geometry.

        Returns:
            int: The run number for this simulation.
        """

        # Get the run number for this simulation
        runNo = self._param['runNumber']
        print('Running GridPix simulation...')
        print(f'Running simulation - Run number: {runNo}')
    
        self._checkParam()
    
        self.setGeometry(unitCell='GridPix')
        self._generateGeometry()
            
        #Solve fields and run Garfield
        self._solveEFields(solveWeighting=True)
        self._runGarfield()
        
        return runNo

#**********************************************************************#
    def runCapacitance(self):
        """
        Solves the capacitance matrix for the geometry using Elmer.
        Solves for a hexagonal unit celll and all neightboring cells.

        Elements are ordered as:
            1. Cathode
            2. Grid
            3. CenterPad
            4. TopPad
            5. BottomPad
            6. TopRightPad
            7. BottomRightPad
            8. TopLeftPad
            9. BottomLeftPad

        Returns:
            capacitanceMatrix (np.array): The capacitance matrix in fF.
        """

        self._checkParam()

        # Create surrounding-cell geometry
        self._geometryCapacitance = geometryClass(self._param)
        self._geometryCapacitance.setUnitCell('FIMSHexagonal')
        self._geometryCapacitance.setSurroundingCells(True)
        self._geometryCapacitance.buildGeometry()

        # Solve the capacitance matrix
        self._geometryCapacitance.calculateEFields(capacitance=True)

        # Read the capacitance matrix from the Elmer output file
        capacitanceMatrix = self._readCapacitanceMatrix()

        return capacitanceMatrix

#**********************************************************************#
    def _readCapacitanceMatrix(self):
        """
        Reads the capacitance matrix from the Elmer output file.

        Returns:
            capacitanceMatrix (np.array): The capacitance matrix in fF.

        """
        try:

            capacitanceFile = '../Simulation/Geometry/elmerResults/capacitancematrix.dat'
            capacitanceMatrix = np.loadtxt(capacitanceFile)

            capacitanceMatrix *= 1e15 #Convert from F to fF

        except FileNotFoundError:
            raise FileNotFoundError('Error: Capacitance file not found.')
        except Exception as e:
            raise RuntimeError(f'An error occurred while reading the capacitance file: {e}')
        
        return capacitanceMatrix

#**********************************************************************#
    def _readEfficiencyResults(self):
        """
        Parses the efficiency results file and merges it with current simulation parameters.

        Returns:
            dict: A dictionary containing the parsed results with the following keys:
                
                Metadata:
                - 'stopCondition' (str): The reason the avalanche simulation terminated.
                - 'fieldRatio' (int): The field ratio used for this calculation.

                Efficiencies (floats):
                    Are the Calculated mean values with lower and upper bounds for the 
                    net, detection, and collection efficiencies
                - 'netEff'
                - 'netErrLow'
                - 'netErrHigh'
                - 'detectionEff'
                - 'detectionErrLow'
                - 'detectionErrHigh'
                - 'collectionEff'
                - 'collectionErrLow'
                - 'collectionErrHigh'
        """

        dataPath = '../Data/'
        dataFilename = 'efficiencyResults.dat'
        
        resultsFile = os.path.join(dataPath, dataFilename)
        try:
            with open(resultsFile, 'r') as inFile:
                allLines = [inLine.strip() for inLine in inFile.readlines()]
        except Exception as e:
            raise RuntimeError(f'Error while reading the results file: {e}')
        
        results = {}
        efficiencies = ['net', 'detection', 'collection']
        
        try:
            for i, line in enumerate(allLines):
                line = line.lower()
                if line.startswith('stop condition:'):
                    results['stopCondition'] = allLines[i + 1].strip()
                    continue

                for inEfficiency in efficiencies:
                    if line.startswith(inEfficiency):
                        results[f'{inEfficiency}Eff'] = float(allLines[i + 1])
                        results[f'{inEfficiency}ErrLow'] = float(allLines[i + 2])
                        results[f'{inEfficiency}ErrHigh'] = float(allLines[i + 3])
                        break

            results['fieldRatio'] = self.getParam('fieldRatio')

        except Exception as e:
            raise RuntimeError(f'Error while parsing the results file: {e}')
        
        return results
    
#**********************************************************************#
    def _fitForNextField(self, targetEfficiency, efficiencyValues, targetValue):
        """
        Calculates the next field ratio using a sigmoid fit of historical data.
        Ensures forward progress by enforcing a minimum positive step.

        Args:
            targetEfficiency (str): The efficiency metric to target ('net', 'detection', 'collection').
            efficiencyValues (list): List of results dictionaries from previous runs.
            targetValue (float): The target efficiency.

        Returns:
            int: The next field ratio to be simulated.
        """

        def mySigmoid(x, k, x0):
            return 1.0 / (1.0 + np.exp(-k * (x - x0)))
        
        targetValue = min(targetValue, 0.999)
            
        minFieldStep = 1
        maxFieldStep = 10

        allPrevData = pd.DataFrame(efficiencyValues)

        xData = allPrevData['fieldRatio'].values
        yData = allPrevData[f'{targetEfficiency}Eff'].values
        yErrHigh = allPrevData[f'{targetEfficiency}ErrHigh'].values
        yErrLow = allPrevData[f'{targetEfficiency}ErrLow'].values
        yErrs = np.maximum(yErrLow, yErrHigh)#Use the maximum error for the fit

        lastField = xData[-1]
        lastResult = yData[-1]
        
        try:
            # Fit the sigmoid parameters
            popt, pcov = curve_fit(
                mySigmoid, xData, yData, 
                p0=[1.0, np.median(xData)], 
                sigma=yErrs, absolute_sigma=True
            )
            k, x0 = popt
            
            # Invert the sigmoid to find x for a given y
            numSolveField = x0 - (1/k) * np.log((1.0 / targetValue) - 1)

            # Find errors of the fit
            perr = np.sqrt(np.diag(pcov))

            # The uncertainty in x (field) due to uncertainties in x0 and k
            kErr = (1.0 / (k**2)) * np.log((1.0 / targetValue) - 1)
            fieldErr = np.sqrt(perr[1]**2 + (kErr * perr[0])**2)

            #shift solution by 2-sigma
            numSolveField = numSolveField + (2*fieldErr)

            #round step down to int
            fieldStep = numSolveField - lastField
        
        except Exception as e:
            print(f'Fit failed, falling back to incremental step: {e}')
            fieldStep = minFieldStep if lastResult <= targetValue else -minFieldStep

        #Limit to maximum and minimum step sizes
        if math.fabs(fieldStep) > maxFieldStep:
            fieldStep = np.sign(fieldStep) * maxFieldStep
        if math.fabs(fieldStep) < minFieldStep:
            fieldStep = np.sign(fieldStep) * minFieldStep

        #Proposed new field
        newField = int(lastField + fieldStep)

        #Check to make sure this field has not been tried
        
        if newField in xData:
            print(f'Warning - Field of {newField} already tried. Step to nearest neighbor...')
            index = np.where(xData == newField)[0][-1]
            oldEfficiency = yData[index]
            oldEffMin = oldEfficiency - yErrLow[index]

            stepDirection = 1 if oldEffMin < targetValue else -1

            numSteps = 0
            while numSteps < 3 and newField in xData:
                numSteps = numSteps+1
                newField = newField + stepDirection*minFieldStep

            print(f'Took {numSteps} steps to {newField}.')

        return newField

#**********************************************************************#
    def _getNextField(self, targetEfficiency, efficiencyValues, targetValue):
        """
        Determines the next field ratio for achieving a target value by
        utilizing a sigmoid-fit-based finder.
        
        Args:
            targetEfficiency (str): The efficiency (net, detection, collection) to consider
            efficiencyValues (list of dict): List of dictionaries containing the previous efficiency information.
            targetValue (float): The target value to reach.

        Returns:
            float: Calculated field ratio to achieve target efficiency value
        """

        numIterations = len(efficiencyValues)

        # Determine new field strength
        initialStep = 5 #TODO - This can be adjusted

        if numIterations == 0:
            newField = self._param['fieldRatio']

        elif numIterations == 1:
            newField = self._param['fieldRatio'] + initialStep

        # Determine new field from fit
        else:
            newField = self._fitForNextField(targetEfficiency, efficiencyValues, targetValue)
        
        return newField

#**********************************************************************#
    def _findMinimumField(self, targetEfficiency='net', targetValue=.95, threshold=10):
        """
        Finds the minimum field necessary to achive a target value in a given efficiency.

        Can be based on the collection, detection, or net efficiency.

        Args:
            targetEfficiency (str): The efficiency (net, detection, collection) to consider
            targetValue (float): The target value to reach.
            threshold (int): Electron alanche size that is considered detected.
        """

        print('\n'.join([
            'Searching for minimum field...',
            f'Target {targetEfficiency} efficiency is: {targetValue}'
        ]))

        saveParam = self.getAllParam()
        self.setParameters({'numAvalanche': 5000})#More is better. Adjust as needed.
        
        # Setting up variables
        resultsDictionary = {
            'Field': [],
            'Value': [],
            'Low Error': [],
            'High Error': []
        }
        efficiencyAtField = []
        fieldValues = []
        iterNo = 0

        while iterNo <= self._iterationNumberLimit:
            iterNo += 1

            newField = self._getNextField(targetEfficiency, efficiencyAtField, targetValue)

            if newField > self._fieldLimit:
                print(f'Warning - Field ratio exceeds limit. Escaping...')
                break

            if newField in fieldValues:
                print(f'Warning - Repeat field. Escaping.')
                break
            fieldValues.append(newField)

            self.setParameters({'fieldRatio': newField})
            print(f'Iteration {iterNo}: Field ratio = {newField}')

            self._solveEFields(solveWeighting=False)
            self._runGarfield(
                'runEfficiency',
                targetEfficiency=targetEfficiency, targetValue=targetValue, threshold=threshold
            )

            runResults = self._readEfficiencyResults()
            efficiencyAtField.append(runResults)

            currentEff = runResults[f'{targetEfficiency}Eff']
            currentErrLow = runResults[f'{targetEfficiency}ErrLow']
            currentErrHigh = runResults[f'{targetEfficiency}ErrHigh']
            twoSigmaEff = currentEff - 2*currentErrLow
            
            print(
                f'\tResult: {currentEff*100:.2f} +/- '
                f'({currentErrHigh*100:.2f}/{currentErrLow*100:.2f})% '
                f'(Stop Condition: {runResults['stopCondition']})'
            )
            
            # append results to results dictionary
            resultsDictionary['Field'].append(newField)
            resultsDictionary['Value'].append(currentEff)
            resultsDictionary['Low Error'].append(currentErrLow)
            resultsDictionary['High Error'].append(currentErrHigh)
            
            if twoSigmaEff > targetValue:
                print('Minimum field found. Terminating search...')
                break
        #End of find field loop

        allResults = pd.DataFrame(efficiencyAtField)
        converged = allResults[allResults['stopCondition'] == 'CONVERGED']
        isEfficient = converged[converged[f'{targetEfficiency}Eff'] >= targetValue]

        if not isEfficient.empty:
            finalField = int(isEfficient['fieldRatio'].min())
            print(f'Solution found: Minimum field ratio = {finalField}')
        else:
            finalField = int(allResults['fieldRatio'].iloc[-1])
            print(f'Warning: Target not reached. Closest field: {finalField}')

        saveParam['fieldRatio'] = finalField
        self.setParameters(saveParam)

        print(f'Solution found: Field ratio = {finalField}')
        #self._printFieldSolution(resultsDictionary) # TODO: consider implementing
        
        allResults.to_csv('../Data/allEfficiencyResults.csv', index=False)

        return finalField

#**********************************************************************#
    def _printFieldSolution(self, resultsAtField):
        """
        TODO - Consider if this is better than just printing the raw values (easier
        to copy + paste)

        Prints the results of the field search in a box format.

        Args:
            resultsAtField (dict): Dictionary containing lists of:
                - field ratios, values, and errors for each iteration. 
        """

        dataLabel = list(resultsAtField.keys())[1]

        header = '| #     Efield     Efficiency     Low Error     High Error |'
        boxWidth = len(header)

        print('\n\t' + '-'*boxWidth)
        print('\t' + header)
        print('\t|' + '='*(boxWidth-2) + '|')

        for testNum, (field, eff, errLow, errHigh) in enumerate(zip(*resultsAtField.values()), 1):
            print(
                f'\t| {testNum:<6}' 
                f'{field:<13.1f}'
                f'{eff:<15.3f}'
                f'{errLow:<14.3f}'
                f'{errHigh:<9.3f}|'
            )

        print('\t' + '-'*boxWidth + '\n')

        return
    
#***********************************************************************************#
    def runForEfficiency(self, targetEfficiency='net', targetValue=.95, threshold=10, initialField=None):
        """
        Executes an avalanche simulation of the FIMS geometry at the minimum field 
        ratio required to achieve a given efficiency.

        This implementation will internally find the field necessary to achieve the 
        desired result.
        The efficiency values from all attempted fields are not recorded.
        
        Args:
            targetEfficiency (str): Efficiency to target. (net, detection, or collection)
            targetValue (float): The target efficiency to reach.
            threshold (int): The avalanche size detection threshold.
            initialField (int): Initial field ratio guess.

        Returns:
            int: The run number of the simulation that was executed.   
        """

        efficiencyOptions = [
            'net', 
            'detection',
            'collection'
        ]
        if targetEfficiency not in efficiencyOptions:
            raise ValueError('Error - Invalid target efficiency.')
        
        #Ensure all parameters exist
        self._checkParam()

        #get the run number for this simulation
        runNo = self._param['runNumber']
        print(f'Running simulation - Run number: {runNo}')

        #Get absolute drift field value
        driftField = self._param['driftField']
        print(f'Finding minimum field ratio for geometry with drift field: {driftField} V/cm')

        #Choose initial field ratio guess
        if initialField is not None:
            minFieldGuess = initialField
        else:
            minFieldGuess = 10 #TODO - better guess?

        print(f'\tInitial field ratio guess: {minFieldGuess}')
        self.setParameters({'fieldRatio': minFieldGuess})

        #Generate the FEM geometry
        self._generateGeometry()

        # Determine minimum field ratio
        minFieldSolution = self._findMinimumField(
            targetEfficiency=targetEfficiency, 
            targetValue=targetValue, 
            threshold=threshold
            )

        self.setParameters({'fieldRatio': minFieldSolution})

        #Solve for the weighting field
        self._solveEFields(solveWeighting=True)
        
        #Run the electron transport simulation
        self._runGarfield()

        return runNo
    
#***********************************************************************************#
    def runForOptimizer(self):
        """
        Executes a full avalanche simulation of the FIMS geometry 
        for efficient running within an optimizer.

        Determines the electric field ratio required for
        95% net efficiency
            Note this generates the geometry and solves the electric field.
        Solves the weighting field and runs the electron transport simulation.
            
        Returns:
            int: The run number of the simulation that was executed.
        """

        #get the run number for this simulation
        runNo = self._param['runNumber']
        print(f'Running simulation - Run number: {runNo}')

        #Get the optimal drift field for this gas
        self.setParameters({'driftField': self._getOptimalDriftField()})

        # Generate Geometry
        self._generateGeometry()

        #Find the minimum field ratio for this geometry
        minField = self._findMinimumField()
        self.setParameters({'fieldRatio': minField})

        #Solve fields and run Garfield
        self._solveEFields(solveWeighting=True)
        self._runGarfield()
        
        return runNo   

#***********************************************************************************#
    def _getEfficiency(self, target, **kwargs):
        """
        Gets a target efficiency value for the current geometry and field ratio.

        Args:
            target (str): The target efficiency to get. Options are:
                - 'detection': The electron detection efficiency.
                - 'transparency': The electric field line transparency.
                - 'collection': The charge collection efficiency.
            kwargs: Additional keyword arguments for specific targets:
                - efficiencyGoal (float): The target detection efficiency.
                - efficiencyThreshold (int): The detection threshold for the detection target.
                - transparencyGoal (float): The target field transparency.
                - collectionGoal (float): The target charge collection efficiency.
        
        Returns:
            tuple containing:
                - float: The simulated target efficiency value.
                - float: The uncertainty in the simulated target efficiency value.

        """

        runSettings = {
            'detection': {'numAvalanche': 5000, 'avalancheLimit': kwargs.get('efficiencyThreshold', 10) + 5},
            'collection': {'numAvalanche': 5000, 'avalancheLimit': 5},
            'transparency': {'numFieldLine': 500}
        }
        saveParam = self.getAllParam()

        if target not in runSettings:
            raise ValueError(f"Invalid target '{target}'. Valid options are: {', '.join(runSettings.keys())}.")
        self.setParameters(runSettings[target])

        match target:
            case 'detection':
                self._runGarfield(
                    'runDetection', 
                    targetEfficiency=kwargs.get('efficiencyGoal', 0.95), 
                    threshold=kwargs.get('efficiencyThreshold', 10)
                )
            
            case 'collection':
                initialZ = kwargs.get('initialZ', 0.5*self._param['cathodeHeight'])
                self._runGarfield(
                    'runCollection', 
                    initialZ=initialZ
                )

            case 'transparency':
                self._runGarfield(
                    'runTransparency',
                    targetTransparency=kwargs.get('transparencyGoal', 0.99)
                )

            case _:
                raise RuntimeError('Unexpected error in target selection.')
            
        
        results = self._readResultsFile(target)
        print(f'\t{target}: {results['result']:.3f} +/- {results['resultErr']:.3f}') 

        self.setParameters(saveParam)

        return results['result'], results['resultErr']

#***********************************************************************************#
#***********************************************************************************#
# METHODS FOR RUNNING CHARGE BUILDUP  (TODO - these need to be largely redone)
#***********************************************************************************#
#***********************************************************************************#
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
        pitch = self._param['pitch']        
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
        numAvalanche = self.getParam('numAvalanche')
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
            saveParam = self.getAllParam()
            doneRun = self.runSimulation(changeGeometry=initialRun)
            self.setParameters(saveParam)
            
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
        self._resetParam()

        chargeBuildupSummary = {
            'numRuns': numRuns,
            'finalRun': doneRun,
            'totalCharge': totalElectrons,
        }
        return chargeBuildupSummary

# END METHODS FOR RUNNING CHARGE BUILDUP (TODO)
#***********************************************************************************#
#***********************************************************************************#


#**********************************************************************#
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

            intAr = round(self._param['gasCompAr']*100)
            intCF4 = round(self._param['gasCompCF4']*100)
            intIsobutane = round(self._param['gasCompIsobutane']*100)

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

#**********************************************************************#
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

