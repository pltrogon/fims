#######################################
# CLASS DEFINITION FOR GAS SIMULATION #
#######################################
from __future__ import annotations
import glob

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

from scipy.optimize import Bounds, minimize_scalar


class gasSimulation:
    """
    Class representing an Electron Avalanche in Gas Simulation.

    This is with a user-defined gas mixture in a parallel plate geometry.

    Attributes:
        gadID (str): Idetntifier of the user-defined gas mixture.
        gasFractions (list): List of the fractional gas components.
        gasArgs (str): String of the gas fractions, for use in calling the C++ executable.
    """


#***********************************************************************************#
    def __init__(self):
        """
        Initializes a gasSimulation object.
        """

        self._GARFIELDPATH = self._getGarfieldPath()
        if self._GARFIELDPATH is None:
            raise RuntimeError('Error getting Garfield++ path.')

        self._setupSimulation()

        self.gasID = None
        self.gasFractions = []
        self.gasArgs = ""

  
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
        Initializes Garfield++ and creates executables.
        
        Reads the Garfiled++ source path, and ensures a log and build directory.
        Compiles the executables using cmake and make.
    
        Note: If a segmentation fault occurs, it is most likely that the
              Garfield++ library is not sourced correctly.
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
        
        if not os.path.exists("../Data/ParallelPlate"):
            os.makedirs("../Data/ParallelPlate")

        # Get garfield path into environment
        envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'

        try:
            envOutput = subprocess.check_output(envCommand, shell=True, universal_newlines=True)
            newEnv = dict(line.split('=', 1) for line in envOutput.strip().split('\n') if '=' in line)
            os.environ.update(newEnv)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Failed to source Garfield environment: {e}')

        #Make executable
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
        finally:
            os.chdir(originalCWD)
                
        return


#***********************************************************************************#
    def setGasComposition(self, gasID=None, gasFractions=None):
        """
        Sets the gas ID and component composition fractions.

        Args:
            gasID (str): ID of the gas composition.
            gasFractions (float): Fractional components of the gas.


        """

        if gasID is None:
            raise ValueError('Error - No gas.')
        elif gasID == 'T2K':
            gasFractions = [95.0, 3.0, 2.0]

        if gasFractions is None:
            raise ValueError('Error - Invalid gas composition')

        self._checkGas(gasID, gasFractions)

        self.gasID = gasID
        self.gasFractions = gasFractions
        self.gasArgs = " ".join(map(str, self.gasFractions))

        return

#***********************************************************************************#
    def _checkGas(self, gasID, gasFractions):
        """
        Checks that the input gas is valid.
        
        This includes checking:  
            - Valid ID string
            - Appropriate number of components
            - Fractional amounts sum to 100%

        Args:
            gasID (str): ID of the gas composition.
            gasFractions (float): Fractional components of the gas.
        """
        gasOptions = {
            'T2K': 3, # Note components are Ar/CF4/Isobutane
            'ArCO2': 2, # Note components are Ar/CO2
            'myT2K': 3 # Note components are Ar/CF4/Isobutane
        }

        if gasID not in gasOptions:
            raise ValueError(f'Error: Invalid gas. Options are: {self.gasOptions.keys()}')
        
        if gasFractions is None:
            raise ValueError(f'Error: No input gas amounts for {gasID}.')
        
        if len(gasFractions) != gasOptions[gasID]:
            raise ValueError(f'Error: Invalid number of gases for {gasID}')
        
        if any(f < 0 for f in gasFractions):
             raise ValueError('Error: Gas amounts cannot be negative.')
        
        gasSum = sum(gasFractions)
        if abs(gasSum - 100) > 1e-3:
            raise ValueError('Error: Gas amounts do not total 100%')

        return

#***********************************************************************************#
    def runMagboltz(self, eField):
        """
        Executes a C++ program that utilizes Garfield++ and Magboltz to calculate and save
        the electron drift and diffusion properties at a given field strength.

        Args:
            eField (float): The magnitude of the electric field in kV/cm.
        """
        originalCWD = os.getcwd()

        if eField <= 0:
            raise ValueError('Error: Invalid electric field.')
        

        try:
            os.chdir('./build/')

            # Get garfield path into environment
            envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'
            envOutput = subprocess.check_output(envCommand, shell=True, universal_newlines=True)
            newEnv = dict(line.split('=', 1) for line in envOutput.strip().split('\n') if '=' in line)
            os.environ.update(newEnv)

            with open(os.path.join(originalCWD, 'log/logMagboltz.txt'), 'w+') as garfieldOutput:
                startTime = time.monotonic()
                setupMagboltz = (
                    f'./runMagboltz {eField} {self.gasID} {self.gasArgs}'
                )
                print(f'Running Magboltz for field of {eField:.3f} kV/cm...')
                subprocess.run(
                    setupMagboltz, 
                    stdout = garfieldOutput, 
                    shell = True, 
                    check = True,
                    env = os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')

        except subprocess.CalledProcessError:
            print('Garfield++ execution failed. Check log for details.')
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        finally:
            os.chdir(originalCWD)
        return 
    
#***********************************************************************************#
    def scanMagboltz(self):
        """
        Executes the Magboltz program to get the electron drift velocity and 
        diffusion coefficiencts for a range of fields between 0.1 kV/cm and 100 kV/cm.
        """
        eFields = np.logspace(-1, 2, 31)#for 100 kV/cm max

        print(f'Scanning {len(eFields)} fields...')
        for inField in eFields:
            self.runMagboltz(inField)
        print(f'Done {len(eFields)} fields...')

        return    

#***********************************************************************************#
    def _runParallelPlateAvalanches(self, fieldStrength=0, plateSeparation=0):
        """
        Executes a Garfield++ program to simulate electron avalanches in a parallel 
        plate geometry at a given plate separation and field strength.

        5000 single-electron avalanches are simulated, with a maximum size of 1000.
        Results are saved in the file: "Data/parallelPlateGain.dat"

        Args:
            fieldStrength (float): The field strength in kV/cm
            plateSeparation (int): Then separation between the plates in microns
        """
        if fieldStrength == 0:
            raise ValueError(f'Error: Field strength is 0.')
        if plateSeparation == 0:
            raise ValueError(f'Error: Plate separation is 0.')
        
        originalCWD = os.getcwd()
        try:
            os.chdir('./build/')

            # Get garfield path into environment
            envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'
            envOutput = subprocess.check_output(envCommand, shell=True, universal_newlines=True)
            newEnv = dict(line.split('=', 1) for line in envOutput.strip().split('\n') if '=' in line)
            os.environ.update(newEnv)

            with open(os.path.join(originalCWD, 'log/logParallelPlate.txt'), 'w+') as garfieldOutput:
                startTime = time.monotonic()
                setupParallelPlate = (
                    f'./runParallelPlate {plateSeparation} {fieldStrength} {self.gasID} {self.gasArgs}'
                )
                print(f'Running parallel plate with field strength {fieldStrength}kV/cm and gap {plateSeparation}um...')
                subprocess.run(
                    setupParallelPlate, 
                    stdout = garfieldOutput, 
                    shell = True, 
                    check = True,
                    env = os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')
            
        finally:
            os.chdir(originalCWD)
        return
    
#***********************************************************************************#
    def _runParallelPlateEfficiency(self, fieldStrength=0, plateSeparation=0, verbose=True):
        """
        Executes a Garfield++ program to simulate electron avalanches in a parallel 
        plate geometry at a given plate separation and field strength.

        Using Bayesyian statistics, the efficiency is determined with a 10-electron threshold.
        At least 100 avalanches are ran, with a size limitted to 20.
        The stopping conditions are as follows:
            - 95% detetction efficiency is excluded with 2-sigma confidence.
            - Efficiency is greater than 95% with 2-sigma confidence
            - A total of 2000 avalanches are simulated.

        Args:
            fieldStrength (float): The field strength in kV/cm
            plateSeparation (int): Then separation between the plates in microns
        """
        if fieldStrength == 0:
            raise ValueError(f'Error: Field strength is 0.')
        if plateSeparation == 0:
            raise ValueError(f'Error: Plate separation is 0.')
        
        originalCWD = os.getcwd()
        try:
            os.chdir('./build/')

            # Get garfield path into environment
            envCommand = f'bash -c "source {self._GARFIELDPATH} && env"'
            envOutput = subprocess.check_output(envCommand, shell=True, universal_newlines=True)
            newEnv = dict(line.split('=', 1) for line in envOutput.strip().split('\n') if '=' in line)
            os.environ.update(newEnv)

            with open(os.path.join(originalCWD, 'log/logParallelPlateEff.txt'), 'w+') as garfieldOutput:
                startTime = time.monotonic()
                setupParallelPlateEfficiency = (
                    f'./runParallelPlateEfficiency {plateSeparation} {fieldStrength} {self.gasID} {self.gasArgs}'
                )
                
                if verbose:
                    print(f'Running parallel plate efficiency with field strength {fieldStrength}kV/cm...')

                subprocess.run(
                    setupParallelPlateEfficiency, 
                    stdout = garfieldOutput, 
                    shell = True, 
                    check = True,
                    env = os.environ
                )
                endTime = time.monotonic()
                garfieldOutput.write(f'\n\nGarfield run time: {endTime - startTime} s')
            
        finally:
            os.chdir(originalCWD)
        return

#***********************************************************************************#
    def _readGainFile(self):
        """
        Reads the simulated avalanches results in "Data/parallelPlateGain.dat".

        Returns;
            pd.DataFrame: Dataframe containing the simulated avalanche statistics.
        """

        gainFilePath = '../Data/parallelPlateGain.dat'
        avalancheData = np.loadtxt(gainFilePath, comments='#')

        numAvalanche = len(avalancheData)
        numNonZero = np.count_nonzero(avalancheData)

        meanGain = avalancheData.mean()
        gainStdDev = np.std(avalancheData, ddof=1)
        gainErr = gainStdDev/np.sqrt(len(avalancheData))

        quantileData = self._bootstrapQuantile(avalancheData)
        n_95 = quantileData['quantile']
        n_95Err = quantileData['quantileErr']
        gainVariability = 1 - n_95/meanGain

        avalancheStats = {
            'meanGain': meanGain,
            'gainErr': gainErr,
            'gainStdDev': gainStdDev,
            'numAvalanche': numAvalanche,
            'n_95': n_95,
            'n_95Err': n_95Err,
            'gainVariability': gainVariability,
            'numNonZero': numNonZero,
        }

        return avalancheStats
    
#***********************************************************************************#
    def _bootstrapQuantile(self, data, numIterations=1000, quantile=0.05):
        """
        Estimate the confidence interval and standard error of a quantile using bootstrapping.

        This method performs a non-parametric bootstrap by resampling the input data 
        with replacement to calculate the variability of the specified quantile.

        Args:
            data (array): Array of simulated avalanche sizes.
            numIterations (int): Numer of bootstrapping iterations to generate.
            quantile (float): The quantile to estimate.

        Returns:
            Dict: Dictionary containing the quantile information.
        """

        trueQuantile = np.quantile(data, quantile)
        numData = len(data)

        quantileEstimates = np.array([
            np.quantile(np.random.choice(data, size=numData, replace=True), quantile)
            for _ in range(numIterations)
        ])
        quantileEstimateStdDev = np.std(quantileEstimates, ddof=1)

        lowBound = np.quantile(quantileEstimates, 0.025)
        highBound = np.quantile(quantileEstimates, 0.975)

        quantileStats = {
            'quantile': trueQuantile,
            'quantileErr': quantileEstimateStdDev,
            'lowBound': lowBound,
            'highBound': highBound
        }

        return quantileStats
    
#***********************************************************************************#
    def scanIsobutane(self, fieldStrength, plateSeparation, compCF4):
        """
        Iterates through Isobutane concentrations of 0 to 10%, performing
        avalanches simulations in a parallel plate geometry for each gas. 
        The field strength is held constant.

        The results of all of these simulations are combined and written to a .pkl file.

        Args:
            fieldStrength (float): The field strength in kV/cm
            plateSeparation (int): The separation between the parallel plates in microns
            compCF4 (int): The percent concentration of CF4 in the gas
        """
        
        avalancheData = []
        
        for compIsobutane in range(11):
            compAr = 100 - compCF4 - compIsobutane
            gasFraction = [compAr, compCF4, compIsobutane]
            self.setGasComposition('myT2K', gasFraction)

            print(f'Running simulation for: {self.gasID} {self.gasArgs}')

            self._runParallelPlateAvalanches(fieldStrength, plateSeparation)

            runResults = self._readGainFile()
            
            runResults['E Field'] = fieldStrength
            runResults['gasID'] = self.gasID
            runResults['Ar'] = compAr
            runResults['CF4'] = compCF4
            runResults['Isobutane'] = compIsobutane
            
            avalancheData.append(runResults)

        avalancheData = pd.DataFrame(avalancheData)

        self._writeGasScan(avalancheData, scannedGas='isobutane')

        return 
    
#***********************************************************************************#
    def _writeGasScan(self, data, scannedGas=None):
        """
        Writes a pandas dataframe of avalanche simulations from a gas scan to a .pkl file.
        These results are with a constant electric field strength.

        Args:
            data (pd.dataframe): The avalanche data from the simulation scans.
            scannedGas (str): The name of the gas that was iterated through.
        """

        gasScans = [
            'isobutane',
            'CF4'
        ]

        if scannedGas not in gasScans:
            raise ValueError('Error - Invalid gas scan.')
        
        #Choose appropriate name based on scanned gas
        match scannedGas:
            case 'isobutane':
                scanGas = '.scanIsobutane'
                constantGas = f'.with{data['CF4'].iloc[0]}CF4'
            case 'CF4':
                scanGas = '.ScanCF4'
                constantGas = f'.with{data['isobutane'].iloc[0]}isobutane'
            case _:
                raise ValueError('Error - Gas scan error.')
        
        
        gasComposition = f'{data['gasID'].iloc[0]}{scanGas}{constantGas}'
        fieldStrength = f'.at{data['E Field'].iloc[0]}kVcm'

        filePath = '../Data/ParallelPlate'
        fileName = f'{gasComposition}{fieldStrength}.pkl'
        filename = os.path.join(filePath, fileName)

        data.to_pickle(filename)

        return
    
#***********************************************************************************#
    def _readEfficiencyFile(self):
        """
        Reads the simulated avalanches results in "Data/parallelPlateEfficiency.dat".

        Returns;
            pd.DataFrame: Dataframe containing the simulated avalanche statistics.
        """

        try:
            with open('../Data/parallelPlateEfficiency.dat', 'r') as inFile:
                allLines = [inLine.strip() for inLine in inFile.readlines()]

        except FileNotFoundError:
            print(f"Error: File 'parallelPlateEfficiency.dat' not found.")
            return None


        if len(allLines) < 11:
            raise IndexError("Malformed file.")
        
        findEffValues = {}
        try:
            findEffValues['stopCondition'] = allLines[7].strip()

            findEffValues['efficiency'] = float(allLines[9].strip())
            findEffValues['efficiencyErr'] = float(allLines[10].strip())

        except (IndexError, ValueError) as e:
            print(f'Error extracting values - {e}')
            return None
                

        return findEffValues
    
#***********************************************************************************#
    def _findIsobutaneEfficiency(self, plateSeparation=0, verbose=True):
        """
        Performs an iterative search to find the minimum Electric Field strength 
        required to achieve a 95% detection efficiency with a 10-e threshold for 
        single-electron avalanches in a parallel-plate geometry with 2-sigma confidence.

        Args:
            plateSeparation (float): The parallel plate separation distance in microns
            verbose (bool): Optional parameter for displaying the finder's intermediate results

        Returns:
            float: The final field that results in a 95% efficiency.
        """

        iterNo = 0
        iterNoLimit = 25 # Max number of iterations for finder

        efficiencyAtField = {
            'field': [],
            'efficiency': [],
            'efficiencyErr': []
        }        

        validEfficiency = False
        while not validEfficiency:

            iterNo += 1
            if iterNo > iterNoLimit:
                print(f'Warning - Iteration limit reached ({iterNoLimit})')
                break

            if verbose:
                print(f'Begining iteration: {iterNo}')

            newField = self._getNextField(iterNo, efficiencyAtField, verbose=verbose)
            newFieldRounded = float(math.ceil(newField))

            efficiencyAtField['field'].append(newFieldRounded)
            
            self._runParallelPlateEfficiency(newFieldRounded, plateSeparation, verbose=verbose)

            #Get efficiency values from file
            effResults = self._readEfficiencyFile()
            efficiencyAtField['efficiency'].append(effResults['efficiency'])
            efficiencyAtField['efficiencyErr'].append(effResults['efficiencyErr'])

            #Determine stopping condition
            match effResults['stopCondition']:
                
                case 'DID NOT CONVERGE':
                    validEfficiency = False
                    if verbose:
                        print(f'Did not converge at {efficiencyAtField['field'][-1]}')

                case 'CONVERGED':
                    validEfficiency = True
                    if verbose:
                        print(f'Converged at {efficiencyAtField['field'][-1]}')
            
                case 'EXCLUDED':
                    validEfficiency = False
                    if verbose:
                        print(f'Excluded at {efficiencyAtField['field'][-1]}')

                case _:
                    raise ValueError('Error - Malformed stop condition.')      
        #End of find field for efficiency loop

        finalField = efficiencyAtField['field'][-1]

        print(f'At field {finalField}kV/cm: Efficiency = {efficiencyAtField['efficiency'][-1]} +/- {efficiencyAtField['efficiencyErr'][-1]}')

        self._plotEfficiencyConvergence(efficiencyAtField)

        return finalField
    
#***********************************************************************************#
    def _getNextField(self, iterNo, efficiencyAtField, verbose=True):
        """
        Determines the next field for achieving a target efficiency. Utilizes the iteration
        number to choreograph a secant-based root-finding method.
        
        Args:
            iterNo (int): Iteration number.
            efficiencyAtField (dict): Dictionary containing field and efficiency information:
                - 'field': Array of previous field strengths.
                - 'efficiency': Array of previous efficiencies.
                - 'efficiencyErr': Array of previous efficiency errors.
            verbose (bool): Optional parameter for displaying the intermediate results

        Returns:
            float: Calculated field stregnth for target efficiency
        """
        # Determine new field strength
        newField = None

        initialField = 15

        if iterNo == 1:
            newField = initialField

        # Take constant step of 2 for 2nd iteration
        elif iterNo == 2:
            newField = initialField + 2

        # Use secant method to determine new field
        else:
            newField = self._getSecantField(efficiencyAtField, verbose=verbose)
        
        if newField is None:
            raise ValueError('Error: Invalid new field')
        
        return newField
    
#***********************************************************************************#
    def _getSecantField(self, efficiencyAtField, verbose=True):
        """
        Utilizes a modified secant method to determine a target electric field strength.

        Makes a step base on the maximum possible slope, amd assumes that the 
        efficiency is monotonically increasing.

        Args:
            efficiencyAtField (dict): Dictionary containing field and efficiency information:
                - 'field': Array of previous field strengths.
                - 'efficiency': Array of previous efficiencies.
                - 'efficiencyErr': Array of previous efficiency errors.
            verbose (bool): Optional parameter for displaying the intermediate results

        Returns:
            float: Calculated field strength for target efficiency
        """

        curField = efficiencyAtField['field'][-1]
        prevField = efficiencyAtField['field'][-1]
        fieldDiff = curField - prevField

        curEff = efficiencyAtField['efficiency'][-1]
        prevEff = efficiencyAtField['efficiency'][-2]

        curEffMax = curEff + efficiencyAtField['efficiencyErr'][-1]
        prevEffMin = prevEff - efficiencyAtField['efficiencyErr'][-2]

        effDiff = curEffMax - prevEffMin
        targetDiff = .95 - curEff

        if abs(effDiff) < 0.001:
            if verbose:
                print(f'Warning: Slope near zero. Using heuristic step of 5%.')
            return curField*1.05 
        
        damping = 0.8
        fieldStep = damping*targetDiff*fieldDiff/effDiff
        if verbose:
                print(f'Field Step: {fieldStep:.2f}')

        if fieldStep > 5:
            if verbose:
                print(f'Warning: Step size limited to 5 for stability.')
            newField = curField+5

        elif fieldStep < 1:
            if verbose:
                print(f'Warning: Field step small. Using heuristic step of 1.')
            newField = curField+1

        else:
            newField = curField+fieldStep

        return newField

    
#***********************************************************************************#
    def scanIsobutaneEfficiency(self, plateSeparation, compCF4):
        """
        Scans through an Isobutane concentration from 0 to 10% in integer amounts,
        holding the CF4 concentration at a constant value. Ensures a 95% detection
        efficiency with an electron threshold of 10.

        Combines the results of all simulations and writes to a .pkl file.

        Args:
            plateSeparation (int): Parallel plate separation distance in microns.
            compCF4 (int): Constant CF4 concentration.
        """

        if compCF4 > 90:
            raise ValueError('Error - CF4 fraction too high.')
        

        avalancheData = []

        for compIsobutane in range(11):
            compAr = 100 - compCF4 - compIsobutane
            gasFraction = [compAr, compCF4, compIsobutane]
            self.setGasComposition('myT2K', gasFraction)

            print(f'Finding efficiency for: {self.gasID} {self.gasArgs}')

            fieldStrength = self._findIsobutaneEfficiency(plateSeparation, verbose=False)

            self._runParallelPlateAvalanches(fieldStrength, plateSeparation)

            runResults = self._readGainFile()
            runResults['E Field'] = fieldStrength
            runResults['gasID'] = self.gasID
            runResults['Ar'] = compAr
            runResults['CF4'] = compCF4
            runResults['Isobutane'] = compIsobutane

            avalancheData.append(runResults)

        avalancheData = pd.DataFrame(avalancheData)

        self._writeGasEfficiencyScan(avalancheData, scannedGas='isobutane')

        return
    
#***********************************************************************************#
    def _writeGasEfficiencyScan(self, data, scannedGas=None):
        """
        Writes a pandas dataframe of avalanche simulations from a gas scan to a .pkl file.
        These simulations have field strengths necessary to achive 95% efficiency.

        Args:
            data (pd.dataframe): The avalanche data from the simulation scans.
            scannedGas (str): The name of the gas that was iterated through.
        """
        gasScans = [
            'isobutane',
            'CF4'
        ]

        if scannedGas not in gasScans:
            raise ValueError('Error - Invalid gas scan.')
        
        match scannedGas:
            case 'isobutane':
                scanGas = '.scanIsobutane'
                constantGas = f'.with{data['CF4'].iloc[0]}CF4'
            case 'CF4':
                scanGas = '.scanCF4'
                constantGas = f'.with{data['isobutane'].iloc[0]}isobutane'
            case _:
                raise ValueError('Error - Scanned gas incorrect.')
        
        
        gasComposition = f'{data['gasID'].iloc[0]}{scanGas}{constantGas}'

        filePath = '../Data/ParallelPlate'
        fileName = f'{gasComposition}.Efficiency.pkl'
        filename = os.path.join(filePath, fileName)

        data.to_pickle(filename)

        return
    
#***********************************************************************************#
    def _plotEfficiencyConvergence(self, efficiencyAtField):
        """
        Plots the efficiency convergence results of electron avalanches in a parallel 
        plate geometry with a defined gas.
        
        The target efficiency of 95% is shown with each iteration's results.

        Args:
            efficiencyAtField (dict): Dictionary containing field and efficiency information:
                    - 'field': Array of field strengths.
                    - 'efficiency': Array of efficiencies.
                    - 'efficiencyErr': Array of efficiency errors.

        """
        plt.errorbar(
            efficiencyAtField['field'], efficiencyAtField['efficiency'], 
            yerr=efficiencyAtField['efficiencyErr'],
            ls='', marker='x'
        )

        plt.axhline(y=.95, ls=':', c='r')
        
        plt.title(f'Efficiency Convergence - {self.gasID} {self.gasArgs}')
        plt.xlabel('Electic Field Strength (kV/cm)')
        plt.ylabel('Detector Efficiency')

        plt.grid()
        plt.show()

        return
    
    #***********************************************************************************#
    def findOptimalDriftField(self):
        """
        Determines the optimal drift field for maximizing the electron drift velocity.
        Boundaries of 0.1 and 1 kV/cm are used for the optimization.

        Returns:
            result (OptimizeResult): The optimization result represented as a 
                                     'OptimizeResult' object.
            pd.DataFrame: Dataframe containing the Magboltz parameters at the optimal field.
        """

        print(f'Finding optimal drift field for {self.gasFractions}...')

        self._optimalMagboltzData = None

        result = minimize_scalar(
            self._objectiveForDriftField,
            bounds=(0.1, 1), #Boundaries of 0.1 and 1 kV/cm
            method='bounded',
            options={'xatol': 0.01}
        )

        return result, self._optimalMagboltzData
    
    #***********************************************************************************#
    def _objectiveForDriftField(self, eField):
        """
        Objective function for optimizing the electron drift field.

        Returns:
            float: Negative of the current drift velocity to be minimized. (For maximization)
        """
        curField = round(eField * 1000) / 1000.0

        self.runMagboltz(curField)
        magboltzData = self._getMaboltzParam(curField)

        if magboltzData.empty:
            return 1e4 #Large penalty for failure

        curDriftVelocity = magboltzData['driftVelocity'].iloc[0] #Want to maximize this

        if (self._optimalMagboltzData is None or 
            curDriftVelocity > (self._optimalMagboltzData['driftVelocity'].iloc[0])):

            self._optimalMagboltzData = magboltzData

        return -1*curDriftVelocity

    #***********************************************************************************#
    def _getMaboltzParam(self, eField):
        """
        Gets and returns the Magboltz parameters for a given electric field.

        Returns:
            pd.DataFrame: Dataframe containing the Magboltz parameters.
        """

        gasCompString = "-".join(map(str, self.gasFractions))
        gasFieldString = f'{int(eField*1000)}'#filename is in V/cm

        gasFilename = f'magboltz.{self.gasID}-{gasCompString}.{gasFieldString}.dat'
        filePath = os.path.join('../Data/Magboltz', gasFilename)

        dataMagboltz = []

        inData = {"filename": os.path.basename(filePath)}
        try:
            with open(filePath, 'r') as inFile:
                lines = [line.strip() for line in inFile.readlines()]
            
            if len(lines) < 11:
                print(f"Error: File {inFile} has unexpected format.")
                return pd.DataFrame()


            inData['gasComposition'] = lines[3]
            inData['eField'] = float(lines[5])
            inData['driftVelocity'] = float(lines[7])
            inData['driftVelocityErr'] = float(lines[8])
            inData['diffusionLongitudinal'] = float(lines[10])
            inData['diffusionLongitudinalErr'] = float(lines[11])
            inData['diffusionTransverse'] = float(lines[13])
            inData['diffusionTransverseErr'] = float(lines[14])

            dataMagboltz.append(inData)
            
        except IndexError:
            print(f"Parsing Error: Could not find data at expected line index in {filePath}.")
        except ValueError:
            print(f"Parsing Error: Could not convert value to float in {filePath}.")
        except IOError as e:
            print(f"Error opening or reading file {filePath}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filePath}: {e}")

        return pd.DataFrame(dataMagboltz)
    

    #***********************************************************************************#
    def scanT2KForOptimalDriftFields(self):
        """
        Scans the T2K gas composition space for optimal drift fields.

        T2k gas is composed of Ar/CF4/Isobutane in 95/3/2 proportions.
        This method scans CF4 and Isobutane concentrations from 0 to 5 % in integer amounts.
        """
        
        rawMagboltzResults = []
        
        scanCF4 = range(0, 6)
        scanIsobutane = range(0, 6)

        for compCF4, compIsobutane in itertools.product(scanCF4, scanIsobutane):
            compAr = 100 - compCF4 - compIsobutane

            gasComp = [compAr, compCF4, compIsobutane]
            self.setGasComposition('myT2K', gasComp)
            
            optimizerResults, magboltzResults = self.findOptimalDriftField()

            if not magboltzResults.empty:
                magboltzResults['Ar'] = compAr
                magboltzResults['CF4'] = compCF4
                magboltzResults['Isobutane'] = compIsobutane
                magboltzResults['optimalField'] = optimizerResults.x

            rawMagboltzResults.append(magboltzResults)

        allMagboltzResults = pd.concat(rawMagboltzResults, ignore_index=True)
        self.saveDriftFieldScan(allMagboltzResults)

        return
    
    #***********************************************************************************#
    def saveDriftFieldScan(self, newData):
        """
        Saves the optimal drift field scan results to a .pkl file.
        If the file already exists, the new data is appended to the existing data.
        """

        filePath = '../Data/Magboltz'
        fileName = f'OptimalDriftFields.{self.gasID}.pkl'
        filename = os.path.join(filePath, fileName)

        if os.path.exists(filename):
            print(f'File exists - Appending.')
            existingData = pd.read_pickle(filename)
            dataToSave = pd.concat([existingData, newData], ignore_index=True)  
        else:
            print('Creating new file.')
            dataToSave = newData

        dataToSave.to_pickle(filename)

        return