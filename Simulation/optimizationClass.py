###################################
# CLASS DEFINITION FOR OPTIMIZATION #
###################################
from __future__ import annotations

import os
import sys
import time
import numpy as np
import uproot
import random
import warnings

from scipy.optimize import Bounds, minimize, differential_evolution

simDir = os.getcwd()
analysisDir = os.path.join(simDir, '..', 'Analysis')
sys.path.append(analysisDir)
from simulationClass import FIMS_Simulation
from runDataClass import runData

#Define a unique warning to terminate the optimizer
class RepeatedInputs(Warning):
    pass

class FIMS_Optimizer:
    """
    ===============================================
    TODO - THIS CLASS NEEDS A DESCRIPTIVE DOCSTRING
    ===============================================


    Class representing the FIMS Optimization.

    Methods defined in FIMS_Optimizer:
        _checkParameters
        _getIBN
        _IBNObjective
        _checkConvergence
        _testObjective
        optimizeForIBN
    """

#***********************************************************************************#
    def __init__(self, params=None):
        """
        Initializes a FIMS_Optimization object.
        """
        self.debug = False
        self.params = params
        self.simFIMS = FIMS_Simulation()
        
        self._checkParameters()

        #Create log file for optimizer
        try:
            with open('log/logOptimizer.txt', 'w') as file:
                pass
        except:
            raise FileNotFoundError('Unable to create log file for optimizer.')
        
        self.iterationNumber = 0
                
#***********************************************************************************#
    #String definition
    def __str__(self):
        """
            Returns a formatted string containing all of the optimization parameters,
        along with their minimum and maximum values.
        """
        singleParam = []

        for line in self.params:
            singleParam.append(f"[{' '.join(map(str, line))}]")
        
        paramList = ' '.join(singleParam)
        
        return paramList

#***********************************************************************************#
    def _checkParameters(self):
        """Checks the input parameters to make sure they have the correct format."""
        
        allowedParams = ['holeRadius', 'gridStandoff', 'padLength', 'pitch']
        
        if self.params is None:
            raise ValueError('Error - No parameters.')

        if not isinstance(self.params, list):
            print('Error: Input not a list')
            return False
        
        for element in self.params:
            if not isinstance(element, list):
                raise ValueError('Error - Parameter element not a list.')
                
            if len(element) != 3:
                raise ValueError('Error - Parameter element does not have three entries.')
                
            if element[1] >= element[2]:
                raise ValueError('Error - Minimum bound is not less than maximum bound.')
                
            if element[0] not in allowedParams:
                raise ValueError('Error - Parameter element is not a valid parameter.')

        return 

#***********************************************************************************#    
    def _getIBN(self):
        """
        Orchestrates the process of running a simulation and calculating
        the Ion Backflow Number (IBN) from the results.
        
        Args:
            None.

        Returns:
            float: The calculated Ion Backflow Number.
        """
        #Acquire list of parameters and the names of the active parameters
        saveParam = self.simFIMS.param.copy()
        activeParams = [line[0] for line in self.params]
        
        print('\n********************************')
        print('Testing Parameters:')
        for element, value in saveParam.items():
            if element in activeParams:
                print(f'{element}: {value}')
        print('********************************\n')
            
        runNumber = self.runForOptimizer()
        
        #Get the IBN
        simData = runData(runNumber)
        IBN = simData.getCalcParameter('Average IBN')

        return IBN

#---------------------------------------------------------
    def _IBNObjective(self, optimizerParam, inputList):
        """
        Objective function to optimize for minimum IBN.

        Updates the simulation's parameter dictionary using the values in the optimizer
        parameter array. Then gets the current IBN, prints the value to monitor convergence,
        and returns it for the optimizer to minimize.
        
        Args:
            optimizerParam (np.array): The flat array of parameters from the optimizer.
            inputList (list): A list of parameter names, matching the order of optimizerParam.
        
        Returns:
            float: The IBN value to be minimized.
        """
        # Unpack the optimizer array into the simulation's parameter dictionary.
        for i, inParam in enumerate(inputList):
            self.simFIMS.param[inParam] = optimizerParam[i]
        #self.simFIMS._writeParam()#This should not be necessary - runing the sim itself will write
        
        # Get the Ion Backflow Number
        resultIBN = self._getIBN()
        
        #Output to monitor convergence
        with open('log/logOptimizer.txt', 'a') as log:
            for line in optimizerParam:
                log.write(f'{line} ')
            log.write(f' {resultIBN}\n')
        print(f'\n******************** IBN = {resultIBN} ********************')
        
        return resultIBN

#***********************************************************************************#
    def _checkConvergence(self, optimizerResult):
        """
        Check previous optimizer parameters to see if optimizer is stuck.
        Terminate the optimizer if the input parameters have been repeated
        five times in a row without change (within 1e-3).
        """
        data = []
        num = 0
        numOfRepeatedParams = 0
        
        #Ensure that at least 5 iterations have occurred before terminating
        if self.iterationNumber < 5:
            return

        #Read data from optimizer log file
        try:
            with open('log/logOptimizer.txt', 'r') as log:
                fullData = [line.rstrip('\n') for line in log]
        except Exception as e:
            print(f'Unable to access file: {e}')
        
        recentData = fullData[-4:]
        
        #Split data into separate, readable lists and determine the number of
        #input parameters
        for line in recentData:
            rawData = map(float, line.split())
            data.append(list(rawData))
        numOfParams = len(data[0]) - 1 #- 1 to remove the IBN
        
        #Calculate the number of input parameters that have not changed
        while num <= numOfParams:
            singleParam = []
            for line in data:
                paramValue = round(line[num], 3)
                singleParam.append(paramValue)
            if singleParam.count(singleParam[0]) == len(singleParam):
                numOfRepeatedParams += 1
            num += 1
        
        #Check the convergence condition
        if numOfRepeatedParams == numOfParams:
            print('Warning: series of identical input parameters detected\n'
            'Terminating optimization...')
            raise StopIteration
            
        return

#***********************************************************************************#
    def _testObjective(self, optimizerParam, inputList):
        # Unpack the optimizer array into the simulation's parameter dictionary.
        for i, inParam in enumerate(inputList):
            self.simFIMS.param[inParam] = optimizerParam[i]
        radius = self.simFIMS.getParam('holeRadius')
        padLength = self.simFIMS.getParam('padLength')
        standoff = self.simFIMS.getParam('gridStandoff')
        pitch = self.simFIMS.getParam('pitch')
        
        #Calculate dummy IBN value
        IBN = 100*(((radius-50)/11.25)**2 - (padLength/225)**2 + abs(standoff - 100)/225)*(.95 + random.random()/10)
        self.iterationNumber += 1
        print(
            'test, radius, standoff, IBN: ', 
            self.iterationNumber, 
            round(radius, 2), 
            round(standoff, 2),
            round(padLength, 2),
            ' | ',
            round(IBN, 4)
            )
        
        #Append iteration values to log
        with open('log/logOptimizer.txt', 'a') as log:
            log.write(f'{radius} {standoff} {padLength} {pitch} {IBN}\n')
        
        return IBN

#***********************************************************************************#
    def optimizeForIBN(self):
        """
        Runs an optimization routine to find the FIMS parameters that minimize 
        the Ion Backflow Number (IBN).

        Returns:
            dict: A dictionary containing:
                - params: Dictionary of optimal FIMS parameters.
                - IBNValue: Final minimum IBN value.
                - success: Boolean representing the success status of minimization.
        """
        activeParameters = self.params
        
        #Get optimizer parameters and bounds
        inputList = [line[0] for line in activeParameters]
        minBounds = [line[1] for line in activeParameters]
        maxBounds = [line[2] for line in activeParameters]

        #Set bounds for variables
        optimizerBounds = Bounds(minBounds, maxBounds)

        #Set initial guess as default values
        optimizerParams = [self.simFIMS.param[parameterName] for parameterName in inputList]
        initialGuess = np.array(optimizerParams)

        print('Beginning optimization...')
        
        if self.debug:
            result = minimize(
                fun=self._testObjective,
                x0=initialGuess,
                args=(inputList),
                method='Nelder-Mead',
                callback=self._checkConvergence,
                bounds=optimizerBounds,
            )
        else:
            result = minimize(
                fun=self._IBNObjective,
                x0=initialGuess,
                args=(inputList),
                method='Nelder-Mead',
                callback=self._checkConvergence,
                bounds=optimizerBounds,
            )

        print('\n*************** Optimization Complete ***************')

        #Put results into simulation instance
        for i, parameterName in enumerate(inputList):
            self.simFIMS.param[parameterName] = result.x[i] 
        
        resultVals = {
            'params': self.simFIMS.param, 
            'ibn_value': result.fun, 
            'success': result.success
        }
        
        print(f"Optimal IBN value = {resultVals['ibn_value']}\n",
        "Parameters for optimal IBN:")
        print(self.simFIMS)

        return resultVals

#***********************************************************************************#

