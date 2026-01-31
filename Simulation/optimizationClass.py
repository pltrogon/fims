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

from scipy.optimize import Bounds, minimize, LinearConstraint

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
        
        self._lastOptimizerParams = None
        self._lastOptimizerResults = (None, None, None)
        
        return
                
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
        """
        Checks the input parameters to make sure they have the correct format. The
        input should be a list of lists, where each sub-list should contain the 
        name of a valid parameter to be adjusted, the minimum value for that
        parameter, and the maximum value for it. If the parameter is not a valid
        option, or if the format is not a list of lists, or if the maximum value is
        less than the minimum, then the method will raise an error. Otherwise, it
        will do nothing.
        
        Valid parameter names: holeRadius, gridStandoff, padLength, and pitch
        
        Args: none
        
        returns: None
        """
        
        givenParam = self.params.copy()
        allowedParams = ['holeRadius', 'gridStandoff', 'padLength', 'pitch']
        
        if givenParam is None:
            raise ValueError('Error - No parameters.')

        if not isinstance(givenParam, list):
            print('Error: Input not a list')
            return False
        
        for element in givenParam:
            if not isinstance(element, list):
                raise ValueError('Error - Primary list elements not lists.')
                
            if len(element) != 3:
                raise ValueError('Error - Secondary list is missing an index. '
                                    'List should have the form: parameter name, '
                                    'min value, max value')
                
            if element[1] >= element[2]:
                raise ValueError('Error - Minimum bound is greater than maximum bound.')
                
            if element[0] not in allowedParams:
                raise ValueError('Error - Parameter name is not valid.\n'
                                    'Options are: holeRadius, gridStandoff, '
                                    'padLength, and pitch.')

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
        activeParams = [line[0] for line in self.params]
        
        print('\n********************************')
        print('Testing Parameters:')
        for element, value in saveParam.items():
            if element in activeParams:
                print(f'{element}: {value}')
        print('********************************\n')
        
        runNumber = self.simFIMS.runForOptimizer()

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
        
        #Get the Ion Backflow Number
        resultIBN = self._getIBN()
            
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
        radius = self.simFIMS.param['holeRadius']
        padLength = self.simFIMS.param['padLength']
        standoff = self.simFIMS.param['gridStandoff']
        pitch = self.simFIMS.param['pitch']
        
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
        
        #Get the data from the previous 4 runs
        recentData = fullData[-4:]
        
        #Split data into separate, readable lists and determine the number of
        #input parameters
        for line in recentData:
            rawData = map(float, line.split())
            data.append(list(rawData))
        numOfParams = len(data[0]) - 1 #- 1 to remove the IBN
        
        #-3 to remove the IBN, efficiency, and transparency
        #numOfParams = len(data[0]) - 3 #TODO: check Tanner's change
        
        #Calculate the number of input parameters that have not changed
        while num <= numOfParams:
            singleParam = []
            for line in data:
                paramValue = round(line[num], 2)
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

#***********************************************************************************#
#*****************Methods Using Field Ratio as a Constraint*************************#
#***********************************************************************************#
def _getIBNALT(self):
        """
        Orchestrates the process of running a simulation and calculating
        the Ion Backflow Number (IBN) from the results.
        
        Args:
            None.

        Returns:
            IBN (float): The calculated Ion Backflow Number.
            efficiency (float): The detection efficiency from the simulation.
            transparency (float): The field transparency from the simulation.
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
            
        runNumber, efficiency, transparency = self.simFIMS.runForOptimizerALT()
        
        #Get the IBN
        simData = runData(runNumber)
        IBN = simData.getCalcParameter('Average IBN')

        return IBN, efficiency, transparency
    
#***********************************************************************************#
    def _IBNObjectiveALT(self, optimizerParam, inputList):
        """
        Objective function to optimize for minimum IBN, with the resulting
        efficiency and transparency values for the simulation.

        Args:
            optimizerParam (np.array): The flat array of parameters from the optimizer.
            inputList (list): A list of parameter names, matching the order of optimizerParam.

        Returns:
            IBN (float): The calculated Ion Backflow Number.
            efficiency (float): The detection efficiency from the simulation.
            transparency (float): The field transparency from the simulation.
        """

        # Unpack the optimizer array into the simulation's parameter dictionary.
        for i, inParam in enumerate(inputList):
            self.simFIMS.param[inParam] = optimizerParam[i]
        
        # Get the Ion Backflow Number
        try:
            self.iterationNumber += 1
            resultIBN, efficiency, transparency = self._getIBNALT()

            if resultIBN <= 0 or np.isnan(resultIBN):
                raise ValueError('Invalid Simulation Result')
            
        except Exception as e:
            print(f'Error during simulation run: {e}')
            resultIBN = 1e6
            efficiency = 0.0
            transparency = 0.0
        
        #Output to monitor convergence
        with open('log/logOptimizer.txt', 'a') as log:
            for line in optimizerParam:
                log.write(f'{line} ')
            log.write(f' {resultIBN} {efficiency} {transparency}\n')

        print('********************************************************************************')
        print(f'IBN = {resultIBN}, Efficiency = {efficiency}, Transparency = {transparency}')
        print('********************************************************************************')

        return resultIBN, efficiency, transparency
    
#***********************************************************************************#
    def optimizeForIBNALT(self):
        """
        Optimizes the FIMS geometry for minimal IBN subject to 
        efficiency and transparency constraints using COBYQA.
        """

        activeParameters = self.params
        inputList = [line[0] for line in activeParameters]
        optimizerBounds = Bounds(
            [line[1] for line in activeParameters], 
            [line[2] for line in activeParameters]
        )
        
        initialGuess = np.array([self.simFIMS.param[p] for p in inputList])

        self._lastOptimizerParams = None
        self._lastOptimizerResults = (None, None, None)

        constraints = [
            NonLinearConstraint(lambda x: self._optimizerMaster(x, inputList)[1], 0.95, 1.01),
            NonLinearConstraint(lambda x: self._optimizerMaster(x, inputList)[2], 0.99, 1.01)
        ]

        print('Beginning optimization...')
        result = minimize(
            fun=lambda x, args: self._optimizerMaster(x, args)[0],
            x0=initialGuess,
            args=inputList,
            method='COBYQA',
            callback=self._checkConvergence,
            bounds=optimizerBounds,
            constraints=constraints
        )

        print('\n*************** Optimization Complete ***************')

        for i, parameterName in enumerate(inputList):
            self.simFIMS.param[parameterName] = result.x[i] 
        
        resultVals = {
            'params': self.simFIMS.param.copy(), 
            'ibn_value': result.fun, 
            'success': result.success
        }
        
        print(f'Optimal IBN value = {resultVals['ibn_value']}')
        print('Parameters for optimal IBN:')
        print(self.simFIMS)

        return resultVals
    
#***********************************************************************************#
    def _optimizerMaster(self, x, inputList):
        """
        Helper to ensure Elmer/Garfield only run once per optimizer step.
        Args:
            x (np.array): The flat array of parameters from the optimizer.
            inputList (list): A list of parameter names, matching the order of x.
        Returns:
            tuple: The cached results from the last optimizer run.
        """

        if self._lastOptimizerParams is None or not np.array_equal(x, self._lastOptimizerParams):
            # Run the actual simulation pipeline
            self._lastOptimizerResults = self._IBNObjectiveALT(x, inputList)
            self._lastOptimizerParams = np.copy(x)
        
        return self._lastOptimizerResults
