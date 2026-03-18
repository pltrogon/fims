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

from scipy.optimize import Bounds, minimize, NonLinearConstraint, LinearConstraint

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

    """

#**********************************************************************#

    def __init__(self, params=None):
        """
        Initializes a FIMS_Optimization object.

        The input parameters should be a list of lists.
        Each inner list must contain:
        - The name of the parameter (string)
        - The minimum value for the parameter (float)
        - The maximum value for the parameter (float)
        
        Args:
            params (list of lists): List of parameters with bounds.
        """
        self.params = params
        self.simFIMS = FIMS_Simulation()
        
        self._checkParameters()

        # Create log file for optimizer
        try:
            with open('log/logOptimizer.txt', 'w') as file:
                file.write('## FIMS Optimization Log ##\n')
        except:
            raise FileNotFoundError('Unable to create log file.')
        
        self._optimizerLog = []
        # TODO - Optimizer log should be saved to a file
        # in case of crashes or early termination. 
        # After each iteration?
        # Should also include timestamp for each entry?

        # Maintain a record of previous trials and results
        self._lastRunParams = None
        self._lastRunResults = None

        return
                
#**********************************************************************#

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

#**********************************************************************#
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

        for inParam in self.params:
            if not isinstance(inParam, list) or len(inParam) != 3:
                raise ValueError(f'Error: {inParam} is invalid.')
                
            name, minVal, maxVal = inParam
            
            if name not in allowedParams:
                raise ValueError(f'Error: {name} not a valid parameter.')
            if minVal >= maxVal:
                raise ValueError(f'Error: Invalid bounds for {name}.')

        return 
    
#**********************************************************************#

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
        
        # Ensure that at least 5 iterations have occurred
        if len(self._optimizerLog) < numIteration:
            return
        
        recentData = self._optimizerLog[-numIteration:]

        history = []
        for entry in recentData:
            roundedParam = tuple(
                round(val, precision) for val in entry['params'].values()
            )
            history.append(roundedParam)

        if len(set(history)) == 1:
            print(f'Warning: {numIteration} identical parameter sets.')
            raise StopIteration
        
        return

#**********************************************************************#
#     
    def _getIBN(self):
        """
        Runs a FIMS simulation and calculates
        the Ion Backflow Number (IBN) from the results.

        Returns:
            IBN (float): The calculated Ion Backflow Number.
        """

        runNumber = self.simFIMS.runForOptimizer()
        
        # Get the IBN
        simData = runData(runNumber)
        IBN = simData.getCalcParameter('Average IBN')

        return IBN

#**********************************************************************#

    def _IBNObjective(self, optimizerParam, inputList):
        """
        Objective function to optimize for minimum IBN.

        Updates the FIMS simulation with the given parameters, 
        runs the simulation, and returns the resulting IBN.

        Assumes that field ratio is not one of the input parameters.
        I.e. The efficiency and transparency conditions are being 
        satified internally by the simulation.

        Note that optimizerParam and inputList must be in the same order.
        
        Args:
            optimizerParam (np.array): Flat array of parameters.
            inputList (list): List of parameter names (in order).
        
        Returns:
            resultIBN (float): The current IBN value.
        """

        # Upload the optimizer parameters into the simulation
        paramDict = dict(zip(inputList, optimizerParam))
        self.simFIMS.setParameters(paramDict)
        
        # Run simulation and get the IBN
        resultIBN = self._getIBN()
        
        # Update the optimizer log
        self._optimizerLog.append({
            'params': paramDict,
            'IBN': resultIBN
        })

        # Print the current IBN value for this iteration
        print(f'Iteration {len(self._optimizerLog)}: IBN = {resultIBN:.6f}')
        
        return resultIBN

#**********************************************************************#

    def optimizeForIBN(self):
        """
        Runs an optimization routine to find the FIMS parameters that 
        minimize the Ion Backflow Number (IBN).

        Utilizes the COBYQA optimization method (derivative-free).
        Bounds are set based on the input parameters. 
        Terminated based on criteria in _checkConvergence.
        Parameters are constrained to prevent unphysical combinations.

        Returns:
            dict: A dictionary containing:
                - params (dict): Optimal FIMS parameters.
                - IBNValue (float): Final minimum IBN value.
                - success (bool): Success status of minimization.
        """
        
        # Get optimizer parameters and bounds
        activeParameters = self.params.copy()
        inputList, minBounds, maxBounds = map(list, zip(*activeParameters))

        # Set bounds for variables
        optimizerBounds = Bounds(minBounds, maxBounds)

        # Set initial guess as default values
        optimizerParams = self.simFIMS.getAllParam()
        initialGuess = np.array(optimizerParams)

        print('Beginning optimization...')
        try:
            optimizerResult = minimize(
                fun=self._IBNObjective,
                x0=initialGuess,
                args=(inputList,),
                method='COBYQA', #or 'Nelder-Mead'
                constraints=self._getGeometryConstraints(),
                callback=self._checkConvergence,
                bounds=optimizerBounds,
            )
            finalParams = optimizerResult.x
            finalFunction = optimizerResult.fun
            finalStatus = optimizerResult.success

        except StopIteration:
            print('Optimization terminated due to convergence.')
            print(finalParams, finalFunction, finalStatus)
            

        print('\n*************** Optimization Complete ***************')
        # Put results into simulation instance
        finalParams = dict(zip(inputList, optimizerResult.x))
        self.simFIMS.setParameters(finalParams)
        
        resultVals = {
            'params': self.simFIMS.getAllParam(), 
            'IBNValue': optimizerResult.fun, 
            'success': optimizerResult.success
        }
        
        print(f"Optimal IBN value = {resultVals['IBNValue']}")
        print(self.simFIMS)

        return resultVals

#**********************************************************************#
#     
    def _getIBNALT(self):
        """
        Runs a FIMS simulation with the current parameters.
        Gets the resulting IBN, efficiency, and transparency.

        Returns:
            dict: A dictionary containing:
                - IBN (float): The calculated Ion Backflow Number.
                - efficiency (float): The calculated efficiency.
                - transparency (float): The calculated transparency.
        """

        runNo, efficiency, transparency = self.simFIMS.runForOptimizerALT()
        
        # Get the IBN
        simData = runData(runNo)
        IBN = simData.getCalcParameter('Average IBN')

        simResults = {
            'IBN': IBN,
            'efficiency': efficiency,
            'transparency': transparency
        }

        return simResults

#**********************************************************************#

    def _IBNObjectiveALT(self, optimizerParam, inputList):
        """
        Objective function to optimize for minimum IBN.

        Updates the FIMS simulation with the given parameters, 
        runs the simulation, and returns the resulting IBN, field 
        transparency, and detection efficiency.

        Assumes that field ratio is one of the input parameters.
        I.e. The efficiency and transparency are external constraints.

        Note that optimizerParam and inputList must be in the same order.
        
        Args:
            optimizerParam (np.array): Flat array of parameters.
            inputList (list): List of parameter names (in order).
        
        Returns:
            Tuple containing:
                - resultIBN (float): The IBN value.
                - resultEfficiency (float): The efficiency value.
                - resultTransparency (float): The transparency value.
        """

        # Upload the optimizer parameters into the simulation
        paramDict = dict(zip(inputList, optimizerParam))
        self.simFIMS.setParameters(paramDict)
        
        # Run sim and get the resulting values
        simResults = self._getIBNALT()

        resultIBN = simResults['IBN']
        resultEfficiency = simResults['efficiency']
        resultTransparency = simResults['transparency']
        
        # Update the optimizer log
        self._optimizerLog.append({
            'params': paramDict,
            'IBN': resultIBN,
            'efficiency': resultEfficiency,
            'transparency': resultTransparency
        })

        # Print the resulting values for this iteration
        print(f"Iteration {len(self._optimizerLog)}: IBN = {resultIBN:.6f}")
        print(f"\tEfficiency = {resultEfficiency:.6f}")
        print(f"\tTransparency = {resultTransparency:.6f}")
        
        return (resultIBN, resultEfficiency, resultTransparency)
    
#**********************************************************************#

    def optimizeForIBNALT(self):
        """
        Runs an optimization routine to find the FIMS parameters that 
        minimize the Ion Backflow Number (IBN).

        Requires that field ratio is one of the input parameters.

        Utilizes the COBYQA optimization method (derivative-free).
        Bounds are set based on the input parameters. 
        Terminated based on criteria in _checkConvergence.
        Parameters are constrained to prevent unphysical combinations. 

        Returns:
            dict: A dictionary containing:
                - params (dict): Optimal FIMS parameters.
                - IBNValue (float): Final minimum IBN value.
                - success (bool): Success status of minimization.
        """

        #TODO: saw a suggestion to normalize parameter space to improve convergence. 

        # Get optimizer parameters and bounds
        activeParameters = self.params.copy()
        if 'fieldRatio' not in [p[0] for p in activeParameters]:
            raise ValueError('Error - fieldRatio must be an input.')
        inputList, minBounds, maxBounds = map(list, zip(*activeParameters))
        
        # Set bounds for variables
        optimizerBounds = Bounds(minBounds, maxBounds)

        # Set constraints for efficiency and transparency
        efficiencyTarget = 0.95
        transparencyTarget = 0.99
        fieldConstraints = [
            NonLinearConstraint(lambda x: self._optimizerMaster(x, inputList)[1], efficiencyTarget, 1.1),
            NonLinearConstraint(lambda x: self._optimizerMaster(x, inputList)[2], transparencyTarget, 1.1)
        ]
        #TODO: Find a way to incorporate the geometry constraints of _getGeometryConstraints here

        # Set initial guess as default values
        optimizerParams = self.simFIMS.getAllParam()
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
                constraints=fieldConstraints
            )
            finalParams = optimizerResult.x
            finalFunction = optimizerResult.fun
            finalStatus = optimizerResult.success

        except StopIteration:
            print('Optimization terminated due to convergence of parameters.')
            print(finalParams, finalFunction, finalStatus)


        print('\n*************** Optimization Complete ***************')

        # Put results into simulation instance
        finalParams = dict(zip(inputList, optimizerResult.x))
        self.simFIMS.setParameters(finalParams)
        
        resultVals = {
            'params': self.simFIMS.getAllParam(), 
            'IBNValue': optimizerResult.fun, 
            'success': optimizerResult.success
        }
        
        print(f'Optimal IBN value = {resultVals["IBNValue"]}')
        print('Parameters:\n', self.simFIMS)

        return resultVals
    
#**********************************************************************#

    def _optimizerMaster(self, x, inputList):
        """
        Master function for optimizer that checks for repeated parameter
        sets to avoid repeat simulations.
        """

        # Check if input parameters are the same as the last run
        # If not, run the simulation and get new results. Save them for later.
        if self._lastRunParams is None or not np.array_equal(x, self._lastRunParams):
            self._lastRunResults = self._IBNObjectiveALT(x, inputList)
            self._lastRunParams = np.copy(x)
        
        return self._lastRunResults

#**********************************************************************#

    def _getGeometryConstraints(self):
        """
        Define the geometry constraints based on the physical 
        requirements of the FIMS design.

        Ensure that the pillars can fit in the space between holes in 
        the grid and the region between pads.
        Ensure that the grid standoff is not too small to prevent arcing.

        Returns:
            LinearConstraint: Object representing the geometry constraints.

        """

        paramName = {p[0]: i for i, p in enumerate(self.params)}
        
        pillarRadius = self.simFIMS.getParam('pillarRadius')
        dielectricThickness = self.simFIMS.getParam('thicknessSiO2')
        numParam = len(self.params)

        minPillar = 5 # Min pillar height

        # Geometry constraints:
        # Ensure that there is enough room for the pillars:
        ##  pitch - 2*holeRadius >= pillarRadius
        ##  pitch - sqrt(3)*padLength >= 2*pillarRadius
        # Ensure that the grid standoff is not too small
        ##  gridStandoff >= dielectricThickness + minPillar

        constraints = [
            ({'pitch': 1, 'holeRadius': -2}, pillarRadius),
            ({'pitch': 1, 'padLength': -1*np.sqrt(3)}, 2*pillarRadius),
            ({'gridStandoff': 1}, dielectricThickness+minPillar)
        ]

        matrix = []
        lowerBound = []
        upperBound = np.inf

        for coeffs, limit in constraints:
            row = np.zeros(numParam)
            for name, value in coeffs.items():
                row[paramName[name]] = value
            
            matrix.append(row)
            lowerBound.append(limit)

        geometryConstraints = LinearConstraint(
            matrix, lowerBound, upperBound
        )

        return geometryConstraints
    
