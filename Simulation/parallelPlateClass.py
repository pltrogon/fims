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
    def _runParallelPlate(self, fieldStrength=0, plateSeparation=0):
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
                setupGetEfficiency = (
                    f'./runParallelPlate {plateSeparation} {fieldStrength} {self.gasID} {self.gasArgs}'
                )
                print(f'Running parallel plate with field strength {fieldStrength}kV/cm and gap {plateSeparation}um...')
                subprocess.run(
                    setupGetEfficiency, 
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

            self._runParallelPlate(fieldStrength, plateSeparation)

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
        
        
        gasComposition = f'{data['gasComp'].iloc[0]}{scanGas}{constantGas}'
        fieldStrength = f'.at{data['E Field'].iloc[0]}kVcm'

        filePath = '../Data/ParallelPlate'
        fileName = f'{gasComposition}{fieldStrength}.pkl'
        filename = os.path.join(filePath, fileName)

        data.to_pickle(filename)

        return
    
#***********************************************************************************#
    def _findIsobutaneEfficiency(self, plateSeparation=0):
        """
        TODO
        """

        return 
    
#***********************************************************************************#
    def _scanIsobutaneEfficiency(self, plateSeparation, compCF4):
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

            fieldStrength = self._findIsobutaneEfficiency(plateSeparation)

            self._runParallelPlate(fieldStrength, plateSeparation)

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
        
        
        gasComposition = f'{data['gasComp'].iloc[0]}{scanGas}{constantGas}'

        filePath = '../Data/ParallelPlate'
        fileName = f'{gasComposition}.Efficiency.pkl'
        filename = os.path.join(filePath, fileName)

        data.to_pickle(filename)

        return