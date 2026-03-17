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

from scipy.optimize import Bounds, minimize, NonLinearConstraint

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
                file.write('## FIMS Optimization Log ##\n')
        except:
            raise FileNotFoundError('Unable to create log file for optimizer.')
        
        self._optimizerLog = []
        # TODO - Optimizer log should be saved to a file after each iteration, 
        # in case of crashes or early termination. 
        # Should also include timestamp for each entry.

        self._lastRunParams = None
        self._lastRunResults = None

        return
                
#***********************************************************************************#
    #String definition
    def __str__(self):
        """
        String containing all of the optimization parameters,
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
        Checks the input parameters for correct format.
        """
        
        allowedParams = [
            'holeRadius', 
            'gridStandoff', 
            'padLength', 
            'pitch',
            'fieldRatio'
        ]
        
        if self.params is None:
            raise ValueError('Error - No parameters.')

        if not isinstance(self.params, list):
            print('Error: Input not a list')
            return False
        

        #TODO - Im not sure of the format of params.
        ## I think it's a list of lists, where each inner list has three entries:
        ## [parameterName, minValue, maxValue]
        ## Should make this more clear in a docstring somewhere.

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
    def _checkConvergence(self, x):
        """
        Checks for convergence of the optimization by looking 
        for repeated parameter sets.

        Will raise a StopIteration exception if the previous 5
        iterations have had identical parameters (to 3 decimal places).

        Args:
            x: Optimizer parameter array (Unused).
        """

        # Number of iterations to check for convergence
        numIteration = 5
        # Decimal precision for parameter comparison
        precision = 3
        
        #Ensure that at least 5 iterations have occurred before terminating
        if len(self._optimizerLog) < numIteration:
            return
        
        recentData = self._optimizerLog[-numIteration:]

        history = []
        for entry in recentData:
            roundedParam = tuple(round(val, precision) for val in entry['params'].values())
            history.append(roundedParam)

        if len(set(history)) == 1:
            print(f'Warning: {numIteration} consecutive identical parameter sets.')
            raise StopIteration
        
        return

#***********************************************************************************#    
    def _getIBN(self):
        """
        Runs a simulation and calculates
        the Ion Backflow Number (IBN) from the results.

        Returns:
            IBN (float): The calculated Ion Backflow Number.
        """

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
            resultIBN (float): The current IBN value.
        """

        #Upload the optimizer parameters into the simulation
        paramDict = dict(zip(inputList, optimizerParam))
        self.simFIMS.setParameters(paramDict)
        
        # Run sim and get the IBN
        resultIBN = self._getIBN()
        
        self._optimizerLog.append({
            'params': paramDict,
            'IBN': resultIBN
        })

        print(f'Iteration {len(self._optimizerLog)}: IBN = {resultIBN:.6f}')
        
        return resultIBN

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
        
        optimizerResult = minimize(
            fun=self._IBNObjective,
            x0=initialGuess,
            args=(inputList,),
            method='COBYQA', #'Nelder-Mead'
            callback=self._checkConvergence,
            bounds=optimizerBounds,
        )

        print('\n*************** Optimization Complete ***************')
        #Put results into simulation instance
        finalParams = dict(zip(inputList, optimizerResult.x))
        self.simFIMS.setParameters(finalParams)
        
        resultVals = {
            'params': self.simFIMS.getAllParam(), 
            'IBNValue': optimizerResult.fun, 
            'success': optimizerResult.success
        }
        
        print(f'Optimal IBN value = {resultVals["IBNValue"]}')
        print(self.simFIMS)

        return resultVals

#***********************************************************************************#    
    def _getIBNALT(self):
        """
        TODO
        """

        runNo, efficiency, transparency = self.simFIMS.runForOptimizerALT()
        
        #Get the IBN
        simData = runData(runNo)
        IBN = simData.getCalcParameter('Average IBN')

        simResults = {
            'IBN': IBN,
            'efficiency': efficiency,
            'transparency': transparency
        }

        return simResults

#***********************************************************************************#
    def _IBNObjectiveALT(self, optimizerParam, inputList):
        """
        TODO
        """

        #Upload the optimizer parameters into the simulation
        paramDict = dict(zip(inputList, optimizerParam))
        self.simFIMS.setParameters(paramDict)
        
        # Run sim and get the resulting values
        simResults = self._getIBNALT()

        resultIBN = simResults['IBN']
        resultEfficiency = simResults['efficiency']
        resultTransparency = simResults['transparency']
        
        self._optimizerLog.append({
            'params': paramDict,
            'IBN': resultIBN,
            'efficiency': resultEfficiency,
            'transparency': resultTransparency
        })

        print(f"Iteration {len(self._optimizerLog)}: IBN = {resultIBN:.6f}")
        print(f"\tEfficiency = {resultEfficiency:.6f}")
        print(f"\tTransparency = {resultTransparency:.6f}")
        
        return (resultIBN, resultEfficiency, resultTransparency)
    
#***********************************************************************************#
    def optimizeForIBNALT(self):
        """
        TODO
        """
        ## NOTE: Requires field ratio to be an input parameter
        #TODO: saw a suggestion to normalize parameter space to improve convergence. 

        activeParameters = self.params.copy()
        self._lastRunParams = None
        self._lastRunResults = None

        if 'fieldRatio' not in [p[0] for p in activeParameters]:
            raise ValueError('Error - fieldRatio must be an input parameter for this optimization.')
        
        #Get optimizer parameters and bounds
        inputList = [line[0] for line in activeParameters]
        minBounds = [line[1] for line in activeParameters]
        maxBounds = [line[2] for line in activeParameters]
        

        #Set bounds for variables
        optimizerBounds = Bounds(minBounds, maxBounds)

        #Set constraints for efficiency and transparency
        efficiencyTarget = 0.95
        transparencyTarget = 0.99
        constraints = [
            NonLinearConstraint(lambda x: self._optimizerMaster(x, inputList)[1], efficiencyTarget, 1.1),
            NonLinearConstraint(lambda x: self._optimizerMaster(x, inputList)[2], transparencyTarget, 1.1)
        ]
        #TODO: geometry constraints to avoid unphysical parameter combinations?

        #Set initial guess as default values
        optimizerParams = [self.simFIMS.param[parameterName] for parameterName in inputList]
        initialGuess = np.array(optimizerParams)

        print('Beginning optimization...')
        try:
            optimizerResult = minimize(
                fun=lambda x, args: self._optimizerMaster(x, args)[0],
                x0=initialGuess,
                args=(inputList,),
                method='COBYQA',
                callback=self._checkConvergence,
                bounds=optimizerBounds,
                constraints=constraints
            )
            finalParams = optimizerResult.x
            finalFunction = optimizerResult.fun
            finalStatus = optimizerResult.success

        except StopIteration:
            print('Optimization terminated due to convergence of parameters.')
            print(finalParams, finalFunction, finalStatus)


        print('\n*************** Optimization Complete ***************')

        #Put results into simulation instance
        for i, parameterName in enumerate(inputList):
            self.simFIMS.param[parameterName] = optimizerResult.x[i] 
        
        resultVals = {
            'params': self.simFIMS.param, 
            'IBNValue': optimizerResult.fun, 
            'success': optimizerResult.success
        }
        
        print(f'Optimal IBN value = {resultVals["IBNValue"]}')
        print('Parameters:\n', self.simFIMS)

        return resultVals
    
#***********************************************************************************#
    def _optimizerMaster(self, x, inputList):
        """
        Master function for optimizer that checks for repeated parameter sets 
        to avoid repeat simulations.
        """

        # Check if input parameters are the same as the last run
        # If not, run the simulation and get new results. Save them for later.
        if self._lastRunParams is None or not np.array_equal(x, self._lastRunParams):
            self._lastRunResults = self._IBNObjectiveALT(x, inputList)
            self._lastRunParams = np.copy(x)
        
        return self._lastRunResults