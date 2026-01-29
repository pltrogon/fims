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

from scipy.optimize import Bounds, minimize, differential_evolution, LinearConstraint

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
        _checkGeometry      <------Renamed (depreciated)
        _getMinField
        _getIBN
        _IBNObjective
        _testObjective
        _getConstraints     <-------New
        _checkConvergence
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
        
        givenParam = self.params.copy()
        allowedParams = ['holeRadius', 'gridStandoff', 'padLength', 'pitch']
        
        if givenParam is None:
            raise ValueError('Error - No parameters.')

        if not isinstance(givenParam, list):
            print('Error: Input not a list')
            return False
        
        for element in givenParam:
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
    def _checkGeometry(self):
        """
        TODO: depreciated
        Checks the geometry parameters to ensure the layout is valid (IE, pads do not
        overlap, holes do not expose the pillars, etc).
        
        Returns:
            False if geometry is invalid
            True otherwise
        """
        
        radius = self.simFIMS.param['holeRadius']
        pitch = self.simFIMS.param['pitch']
        pad = self.simFIMS.param['padLength']
        stand = self.simFIMS.param['gridStandoff']
        pillar = self.simFIMS.param['pillarRadius']
        insulator = self.simFIMS.param['thicknessSiO2']
        
        if pillar + 2*radius > pitch:
            return False
        
        elif (pad*0.866 + pillar)*2 > pitch:
            return False
        
        elif insulator + 2 > stand: #2 chosen as an arbitrary buffer distance
            return False
        
        else:
            return True

#***********************************************************************************#    
    def _getMinField(self):
        """
        Wrapper for minField functions
        """
        savedParams = self.simFIMS.param.copy()
        
        #Calculate an initial guess for the minimum field to use as a baseline
        estimatedMinField = self.simFIMS._calcMinField()
        self.simFIMS.param['fieldRatio'] = estimatedMinField
        
        # Get the minimum field ratio for at least 95% detection efficiency
        timeStart = time.time()
        minField = self.simFIMS.findFieldForEfficiency(targetEfficiency=.95, threshold=10)
        if minField < 0:
            raise ValueError('Failed to find minimum field (efficiency).')
        timeEnd = time.time()
        
        print('********************************\n')
        print('Time to find min field for efficiency: ', timeEnd - timeStart)
        print('********************************\n')

        # Get the minimum field ratio for 100% field transparency
        timeStart = time.time()
        if not self._checkTransparency:
            minField = self.simFIMS.findFieldForTransparency(False)
            if minField < 0:
                raise ValueError('Failed to find minimum field (transparency).')
        timeEnd = time.time()
        
        print('********************************\n')
        print('Time to find min field for transparency: ', timeEnd - timeStart)
        print('********************************\n')
        
        self.simFIMS.param = savedParams
        self.simFIMS.param['fieldRatio'] = minField
        self.simFIMS._writeParam()

        return minField

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
            
        runNumber = self.simFIMS.runForOptimizer()
        
        print('********************************\n')
        print('Time to run avalanche sim: ', timeEnd - timeStart)
        print('********************************\n')

        #Get the IBN
        simData = runData(runNumber)
        IBN = simData.getCalcParameter('Average IBN')

        return IBN

#***********************************************************************************#
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
        
        self.simFIMS._writeParam()
        
        #Check parameters to ensure a valid geometry
        if self._checkGeometry():    
            # Get the Ion Backflow Number
            resultIBN = self._getIBN()
        else:
            print('Bad Geometry. Skipping Test')
            resultIBN = 1000
            

        #Output to monitor convergence
        with open('log/logOptimizer.txt', 'a') as log:
            for line in optimizerParam:
                log.write(f'{line} ')
            log.write(f' {resultIBN}\n')
        print(f'\n******************** IBN = {resultIBN} ********************')
        
        return resultIBN

#***********************************************************************************#
    def _testObjective(self, optimizerParam, inputList):
        """
        An equation that takes the input parameters and calculates a dummy IBN
        value. Used for debugging the optimizer.
        
        Args:
            array of optimizer values
            list of the names of the input variables
        Returns:
            Int, dummy IBN value
        """
        
        # Unpack the optimizer array into the simulation's parameter dictionary.
        for i, inParam in enumerate(inputList):
            self.simFIMS.param[inParam] = optimizerParam[i]
        radius = self.simFIMS._getParam('holeRadius')
        padLength = self.simFIMS._getParam('padLength')
        standoff = self.simFIMS._getParam('gridStandoff')
        pitch = self.simFIMS._getParam('pitch')
        
        #Calculate dummy IBN value
        IBN = (1
                + abs(radius - 6)**2 
                + abs(padLength - 20)**2 
                + abs(standoff - 30)**2 
                + abs(pitch - 30)**2
                )*(.95 + random.random()/10)
        
        self.iterationNumber += 1
        print(
            'test, radius, standoff, pitch, IBN: ', 
            self.iterationNumber, 
            round(radius, 2), 
            round(standoff, 2),
            round(padLength, 2),
            round(pitch, 2),
            ' | ',
            round(IBN, 4)
            )
        
        #Append iteration values to log
        with open('log/logOptimizer.txt', 'a') as log:
            log.write(f'{radius} {standoff} {padLength} {pitch} {IBN}\n')
        
        return IBN

#***********************************************************************************#
    def _getConstraints(self):
        """
        Creates a dictionary of constraints to be used by the optimizer. These
        constraints prevent overlapping geometry.
        
        Note: Assumes that all four possible input parameters are being used.
        
        returns: dictionary of constraints
        """
        givenParams = []
        for list in self.params.copy():
            givenParams.append(list[0])
        
        radNum = givenParams.index('holeRadius')
        standNum = givenParams.index('gridStandoff')
        padNum = givenParams.index('padLength')
        pitchNum = givenParams.index('pitch')
        
        pillar = self.simFIMS.param['pillarRadius']
        SiO2 = self.simFIMS.param['thicknessSiO2']
        
        #constraints formatted for SLSQP method
        constraint = [
            {'type': 'ineq', 'fun': lambda x: x[pitchNum] - 2*x[radNum] - pillar},
            {'type': 'ineq', 'fun': lambda x: x[pitchNum] - (x[padNum]*0.866 + pillar)*2},
            {'type': 'ineq', 'fun': lambda x: x[standNum] - (SiO2 + 2)}
            ]
        
        #matrix elements for the constrain equations
        linCon1 = [0,0,0,0]
        linCon2 = [0,0,0,0]
        linCon3 = [0,0,0,0]
        
        linCon1[radNum] = -2
        linCon1[pitchNum] = 1
        
        linCon2[pitchNum] = 1
        linCon2[padNum] = -1.732
        
        linCon3[standNum] = 1
        
        lowBound = [pillar, 2*pillar, SiO2+2]
        upBound = [np.inf, np.inf, np.inf]
        
        #Constraints formatted for methods other than SLSQP
        altConstraint = LinearConstraint([linCon1, linCon2, linCon3], lowBound, upBound)
        
        return altConstraint

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
                paramValue = round(line[num], 2) #Adjust this number to adjust the sensitivity
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
                method='COBYQA',
                constraints = self._getConstraints(),
                callback=self._checkConvergence,
                bounds=optimizerBounds,
                )
        else:
            result = minimize(
                fun=self._IBNObjective,
                args=(inputList),
                x0=initialGuess,
                method='COBYQA',
                constraints = self._getConstraints(),
                bounds=optimizerBounds,
                callback=self._checkConvergence,
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
        self.simFIMS.resetParam()
        
        return resultVals

