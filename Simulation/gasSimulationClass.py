#######################################
# CLASS DEFINITION FOR GAS SIMULATION #
#######################################
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


class gasSimulation:
    """
    TODO
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
    
        Note: If a degmentation fault occurs, it is most likely that the
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
        TODO
        """
        if gasID == 'T2K':
            gasFractions = [95.0, 3.0, 2.0]

        self._checkGas(gasID, gasFractions)

        self.gasID = gasID
        self.gasFractions = gasFractions
        self.gasArgs = " ".join(map(str, self.gasFractions))

        return

#***********************************************************************************#
    def _checkGas(self, gasID, gasFractions):
        """
        """
        gasOptions = {
            'T2K': 3,
            'ArCO2': 2,
            'myT2K': 3
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
                print(f'Running Magboltz for field of {eField:.3} kV/cm...')
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
        TODO
        """
        eFields = np.logspace(-1, 2, 31)

        print(f'Scanning {len(eFields)} fields...')
        for inField in eFields:
            
            self.runMagboltz(inField)

        return
    

#***********************************************************************************#
    def _runParallelPlateAvalanches(self, fieldStrength=0, plateSeparation=0):
        """
        TODO
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
        TODO
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
        TODO
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
        TODO
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
        TODO
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
        TODO
        """
        gasScans = [
            'isobutane',
            'CF4'
        ]

        if scannedGas not in gasScans:
            raise ValueError('Error - Invalid gas scan.')
        
        if scannedGas == 'isobutane':
            scanGas = '.scanIsobutane'
            constantGas = f'.with{data['CF4'].iloc[0]}CF4'
        if scannedGas == 'CF4':
            scanGas = '.ScanCF4'
            constantGas = f'.with{data['isobutane'].iloc[0]}isobutane'
        
        
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
        TODO
        """

        try:
            with open('../Data/parallelPlateEfficiency.dat', 'r') as inFile:
                allLines = [inLine.strip() for inLine in inFile.readlines()]

        except FileNotFoundError:
            print(f"Error: File 'efficiencyFile.txt' not found.")
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
        TODO
        """

        iterNo = 0
        iterNoLimit = 25 #TODO - 

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
        TODO
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
        TODO
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
        TODO
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
        TODO
        """
        gasScans = [
            'isobutane',
            'CF4'
        ]

        if scannedGas not in gasScans:
            raise ValueError('Error - Invalid gas scan.')
        
        if scannedGas == 'isobutane':
            scanGas = '.scanIsobutane'
            constantGas = f'.with{data['CF4'].iloc[0]}CF4'
        if scannedGas == 'CF4':
            scanGas = '.ScanCF4'
            constantGas = f'.with{data['isobutane'].iloc[0]}isobutane'
        
        
        gasComposition = f'{data['gasID'].iloc[0]}{scanGas}{constantGas}'

        filePath = '../Data/ParallelPlate'
        fileName = f'{gasComposition}.Efficiency.pkl'
        filename = os.path.join(filePath, fileName)

        data.to_pickle(filename)

        return
    
#***********************************************************************************#
    def _plotEfficiencyConvergence(self, efficiencyAtField):
        """
        Docstring for plotEfficiencyComvergence
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