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
import math

from scipy.optimize import Bounds, minimize, NonlinearConstraint, LinearConstraint

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
    TODO: Review Docstring
    ===============================================
    
    Class representing the FIMS optimization algorithm.
    
    Utilizes scipy.optimize's minimize method with the COBYQA method 
    to minimize a target parameter. Currently only accepts the hole 
    radius, pitch, grid standoff height (amplification distance), and 
    pad length as input parameters. 
    
    Note: Currently only minimizes the IBN.
    
    Private Attributes:
        params (dict of lists):
            first param name: [minimum value, maximum value],
            .
            .
            .
            last param name: [minimum value, maximum value]
        
        initialValues (list): list of initial values for each parameter.
        Note: initial values currently taken from FIMS.Simulation class'
        default parameters (TODO: allow default params as input?)
        
        simFIMS (simulationClass): a simulation class object that 
    represents the simulation pipeline.
        
        optimizerLog (list): input values and the corresponding target
        output value for each iteration of the optimizer.
        
        startTime (float): timestamp of the beginning of the optimizer
        lastRunParams (dictionary): parameters and values from the 
    previous iteration.
        
        lastRunResults (float): the target output value of the 
        previous iteration.
    """

#**********************************************************************#

    def __init__(self, params=None):
        """
        Initializes a FIMS_Optimization object.

        The input parameters should be a dictionary of lists.
        Each list must contain:
        - The minimum value for the parameter (float)
        - The maximum value for the parameter (float)
        
        Args:
            params (list of lists): List of parameters with bounds.
        """
        self.params = params
        self._initialValues = []
        self.simFIMS = FIMS_Simulation()
        
        self._checkParameters()

        # Create log file for optimizer
        try:
            with open('log/logOptimizer.txt', 'w') as file:
                file.write('## FIMS Optimization Log ##\n')
        except:
            raise FileNotFoundError('Unable to create log file.')
        
        # Setup log file and timestamps
        self._optimizerLog = []
        self._startTime = time.perf_counter()

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

        for item in self.params:
            singleParam.append(f'{item} {self.params[item]}') #TODO: double check
            #singleParam.append(f"[{' '.join(map(str, line))}]")
        
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
        paramCopy = self.params.copy()
        
        if paramCopy is None:
            raise ValueError('Error - No parameters.')

        for paramName in paramCopy:
            if not isinstance(paramCopy[paramName], list) or len(paramCopy[paramName]) != 2:
                raise ValueError(f'Error: {paramName} is invalid.')
                
            name = paramName
            minVal = min(paramCopy[paramName])
            maxVal = max(paramCopy[paramName])
            
            if name not in allowedParams:
                raise ValueError(f'Error: {name} not a valid parameter.')

        return 
    
#**********************************************************************#

    def _getGeometryConstraints(self):
        """
        Define the geometry constraints based on the physical 
        requirements of the FIMS design.

        Ensure that the pillars can fit in the space between holes in 
        the grid and the region between pads.
        Ensure that the grid standoff is not too small to prevent arcing.
        
        Note: Assumes input parameters are normalized by their initial
        values.
        
        Returns:
            LinearConstraint: Object representing the geometry constraints.

        """
        # Get input parameters and their initial values
        paramIndex = {p: i for i, p in enumerate(self.params)}
        initialPitch = self._initialValues[paramIndex['pitch']]
        initialRadius = self._initialValues[paramIndex['holeRadius']]
        initialPad = self._initialValues[paramIndex['padLength']]
        initialStand = self._initialValues[paramIndex['gridStandoff']]
        
        radiusRatio = initialPitch/initialRadius
        padRatio = initialPitch/initialPad
        
        # Get Fixed parameter values        
        pillarRadius = self.simFIMS.getParam('pillarRadius')
        dielectricThickness = self.simFIMS.getParam('thicknessSiO2')
        minPillar = 5 # Min pillar height
        buffer = .01 # safety buffer for precision at boundary
        numParam = len(self.params)


        # Geometry constraints:
        # Format: a*x(1) + b*x(2) >= c*x(3)
        # a,b,c are constants of the matrix. x(n) is the parameter.
        # 1. Ensure radius is smaller than the pitch (with sufficient space for pillars)
        # 2. Ensure pads are smaller than the pitch (with sufficient space for pillars)
        # 3. Ensure that the grid is above the SiO2 layer (with a buffer)
        
        constraints = [
            ({'pitch': initialPitch, 'holeRadius': -2*initialRadius}, pillarRadius+buffer),
            ({'pitch': initialPitch, 'padLength': -initialPad}, 2*pillarRadius+buffer),
            ({'gridStandoff': initialStand}, dielectricThickness+minPillar+buffer)
        ]

        matrix = []
        lowerBound = []
        upperBound = np.inf

        for coeffs, limit in constraints:
            row = np.zeros(numParam)
            for name, value in coeffs.items():
                row[paramIndex[name]] = value
            
            matrix.append(row)
            lowerBound.append(limit)
        geometryConstraints = LinearConstraint(
            matrix, lowerBound, upperBound
        )

        return geometryConstraints

#**********************************************************************#

    def _normalizeValues(self, rawValues):
        """
        Normalizes a given list of values to the matching parameter.
        
        Note: assumes that the list of values is given in the same order
        as the input parameters.
        
        Args: 
            rawValues (list): list of values associated with the input
            parameters.
        returns:
            normValues (list): list of values normalized by the initial
            value of the corresponding input parameter.
        """
        initialVals = []
        for value in self._initialValues:
            initialVals.append(value)
        normValues = []
        paramID = 0
        
        # Normalize each value in the list using the initial value of
        # the corresponding parameter as the normalization coefficient.
        for elem in rawValues:
            normValues.append(elem/initialVals[paramID])
            paramID += 1
            
        return normValues

#**********************************************************************#

    def _unNormalizeInputs(self, optimizerDict):
        """
        Converts the optimizer guess to a value readable by simFIMS
       
        Uses the initial guess for each parameter as a normalization
        factor and calculates the raw input value from the current 
        optimizer value.
        
        args:
            inputParams (list): list of the names of each input 
        parameter.
        
        returns:
            paramVals (dict): dictionary of parameters names and values
        """
        paramVals = {}
        paramIndex = {p: i for i, p in enumerate(self.params)}
        for param in optimizerDict:
            val = optimizerDict[param]
            paramVals[param] = val*self._initialValues[paramIndex[param]]
        
        return paramVals
        
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

    def _getIBN(self):
        """
        Runs a FIMS simulation and calculates
        the Ion Backflow Number (IBN) from the results.

        Returns:
            IBN (float): The calculated Ion Backflow Number.
        """
        
        print(f'********** Iteration {len(self._optimizerLog)+1:<3}************')
        allParams = self.simFIMS.getAllParam()
        for elem in self.params:
            print(f'\t{elem}: {allParams[elem]}')
        print('************************************')

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
        satisfied internally by the simulation.

        Note that optimizerParam and inputList must be in the same order.
        
        Args:
            optimizerParam (np.array): Flat array of parameters.
            inputList (list): List of parameter names (in order).
        
        Returns:
            resultIBN (float): The current IBN value.
        """
        runStart = time.perf_counter()
        
        # Unpack and Upload the optimizer parameters into the simulation
        paramDict = dict(zip(inputList, optimizerParam))
        unNormalizedDict = self._unNormalizeInputs(paramDict)
        self.simFIMS.setParameters(unNormalizedDict)
        
        # Calculate and set the ideal hole radius
        #pitch = self.simFIMS.getParam('pitch')
        #gridArea = pitch**2*math.sqrt(3)/2
        #optTrans = 0.15 # Ideal optical transparency is 15%
        #holeRadius = math.sqrt(optTrans*gridArea/math.pi)
        #self.simFIMS.setParameters({'holeRadius': holeRadius})
        
        # Run simulation and get the IBN
        resultIBN = self._getIBN()
        
        # Get time stamps
        runEnd = time.perf_counter()
        runTime = runEnd - runStart
        totalTime = runEnd - self._startTime
        
        # Update the optimizer log
        self._optimizerLog.append({
            'params': unNormalizedDict,
            'IBN': resultIBN
        })
        with open('log/logOptimizer.txt', 'a') as file:
                file.write(f'\nIteration {len(self._optimizerLog)}\n')
                for param, value in paramDict.items():
                    file.write(f'\t{param}: {value}\n')
                file.write(f'\tIBN: {resultIBN}\n')
                file.write(f'Run Time: {runTime}\n')
                file.write(f'Total Time: {totalTime}')
                
        # Print the current IBN value for this iteration
        print(f'\tIteration {len(self._optimizerLog)}: IBN = {resultIBN:.6f}\n')
        
        
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
        inputList = []
        minBounds = []
        maxBounds = []
        activeParameters = self.params.copy()
        
        for paramName in activeParameters:
            inputList.append(paramName)
            minBounds.append(min(activeParameters[paramName]))
            maxBounds.append(max(activeParameters[paramName]))
        
        # Set initial guess as simFIMS default values
        initialGuess = np.empty(0)
        for param in inputList:
            self._initialValues.append(self.simFIMS.getParam(param))
            initialGuess = np.append(initialGuess, 1) # All inputs initially normalized to 1
        
        # Set bounds for variables
        normMinBounds = self._normalizeValues(minBounds)
        normMaxBounds = self._normalizeValues(maxBounds)
        optimizerBounds = Bounds(normMinBounds, normMaxBounds)

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
                options = {'initial_tr_radius': .2} # initial step of 20%
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
    
