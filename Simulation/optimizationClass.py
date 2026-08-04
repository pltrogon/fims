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
        
        initialGeometry (dict): dictionary of geometry values to be used
        as initial values for the optimizer.
        
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
        self.simFIMS = FIMS_Simulation()
        
        # Set geometry configuration and values
        self.params = params
        self.initialGeometry = {
            'padLength': 25.,
            'pitch': 55.,
            'amplificationGap': 30.,
            'gridThickness': 1.,
            'holeRadius': 16.,
            'driftLength': 200.,
            'thicknessSiO2': 5.,
            'pillarRadius': 5.,
        }
        self.geoConfig = self.simFIMS._geoConfiguration
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
            'amplificationGap', 
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

    def _setInitialParameters(self, initialGuess={}):
        """
        initial value for each parameter. Dimensions in microns.
        
        args: 
            initialGuess (dict): dictionary of initial values to be used
            in the optimizer.
        """
        # Verify format
        if not isinstance(initialGuess, dict):
            raise ValueError('Error: Initial guess is not a dictionary')

        for param in initialGuess:
            if not isinstance(initialGuess[param], (int, float)):
                raise ValueError(f'Error: {param} value is not valid. Must be a number.')
        
        # Update default values with given values, if any provided
        for geo in initialGuess:
            self.initialGeometry[geo] = initialGuess[geo]
        
        return
#**********************************************************************#

    def _makeConstraintEquation(self, keys, variables, constants):
        """
        Creates a single constraint equation.
        
        args:
            keys (list): names of each parameter present in the equation.
            variables (dict): parameters on the left side of the equation.
            constants (dict): parameters on the right side of the equation.
        
        returns:
            constraintEquation (tuple): constraint equation with constants summed
        on the right and free variables as a dictionary on the left.
        """
        # Get the names of the parameters and distinguish which are constant.
        constKeys = set(keys) - set(variables.keys())
        actKeys = set(keys) - set(constants.keys())
        
        # Create constraint equation
        buffer = .01 # safety buffer for precision at boundary  
        
        if len(actKeys) <= 0:
            return []

        eqDict = {key: variables[key] for key in actKeys}
        eqConstant = sum([-1*constants[key] for key in constKeys]) + buffer
        constraintEquation = (eqDict, eqConstant)
        
        return constraintEquation

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
        '''
        Note: assumes initially that all parameters are on the left side of the equation.
        Shape constants:
            Hex unit cell: circle = 2, square = 4/math.sqrt(3), hexagon = math.sqrt(3), octagon = 2.0173
            square unit cell: circle = 2, square = 1, hexagon = 2, octagon = 2*cos(67.5)
            Note: octagon length defined as distance from center to vertex. 
        '''
        
        # Shape scaling factors lookup
        octagonFactor = 2 * math.cos(math.radians(67.5))
        kikiFactor = math.sqrt(3)

        holeShapeFactors = {
            'circle':  (-2, -2),
            'hexagon': (-math.sqrt(3), -2),
            'octagon': (-2.0173, -octagonFactor),
            'triangle': (-2, -2),
            'kiki': (-math.sqrt(3), -2),
            'nesteggs': (-7.1, -7.1),
            'trivialpursuit': (-4.4, -4.4)
        }
    
        padShapeFactors = {
            'square':  (-4 / math.sqrt(3), -1),
            'hexagon': (-math.sqrt(3), -2),
            'octagon': (-1.9601, -octagonFactor),
        }
    
        # Get all geometry values
        initGeometry = self.initialGeometry.copy()
        hexCell = 'Hexagonal' in self.geoConfig.unitCell
        hexID = 0 if hexCell else 1
        
        # Apply geometry-dependent multipliers
        holeShape = self.geoConfig.holeShape
        if holeShape in holeShapeFactors:
            initGeometry['holeRadius'] *= holeShapeFactors[holeShape][hexID]
    
        padShape = self.geoConfig.padShape
        if padShape in padShapeFactors:
            initGeometry['padLength'] *= padShapeFactors[padShape][hexID]
        
        # Set Other geometry parameters
        initGeometry['pillarRadius'] *= -1
        initGeometry['thicknessSiO2'] *= -1
        
        # Active vs fixed parameters
        activeParams = list(self.params.keys())
        numParams = len(activeParams)
        paramIndex = {p: i for i, p in enumerate(activeParams)}
    
        initVals = {p: initGeometry[p] for p in activeParams}
        fixVals = {k: initGeometry[k] for k in set(initGeometry) - set(activeParams)}
        fixVals['zBuffer'] = -5  # Min pillar height
    
        # Constraint key definitions
        constraintKeySets = [
            ['pitch', 'holeRadius', 'pillarRadius'],                      # Hole clearance
            ['pitch', 'padLength', 'pillarRadius'],                       # Pad clearance
            ['amplificationGap', 'pillarRadius', 'thicknessSiO2', 'zBuffer']  # SiO2 standoff
        ]

        # Build constraint matrix
        matrixRows = []
        lowerBounds = []
    
        for keys in constraintKeySets:
            eq = self._makeConstraintEquation(keys, initVals, fixVals)
            if not eq:
                continue
    
            coeffs, limit = eq
            row = np.zeros(numParams)
            for name, val in coeffs.items():
                if name in paramIndex:
                    row[paramIndex[name]] = val
    
            matrixRows.append(row)
            lowerBounds.append(limit)
    
        if not matrixRows:
            return LinearConstraint(np.empty((0, numParams)), [], [])
        
        geometryConstraints = LinearConstraint(
            A=np.array(matrixRows),
            lb=np.array(lowerBounds),
            ub=np.full(len(lowerBounds), np.inf)
        )

        return geometryConstraints

#**********************************************************************#

    def _normalizeValues(self, initialValues, rawValues):
        """
        Normalizes a given list of values to the matching parameter.
        
        Note: assumes that the list of values is given in the same order
        as the input parameters.
        
        args: 
            initialValues (list): list of initial values
            rawValues (list): list of current values
        
        returns:
            normValues (list): list of normalized values.
        """
        valueList = list(zip(rawValues, initialValues))
        normValues = [raw/initial for raw, initial in valueList]
    
        return normValues

#**********************************************************************#

    def _unNormalizeInputs(self, optimizerDict):
        """
        Converts the optimizer guess to a value readable by simFIMS
       
        Uses the initial guess for each parameter as a normalization
        factor and calculates the raw input value from the current 
        optimizer value.
        
        args:
            inputParams (dict): names and normalized values of each 
        input parameter.
        
        returns:
            paramVals (dict): parameter names and values
        """
        paramVals = {}
        for param in optimizerDict:
            paramVals[param] = optimizerDict[param]*self.initialGeometry[param]
        
        return paramVals
        
#**********************************************************************#

    def _checkConvergence(self, x):
        """
        Checks for convergence of the optimization by looking 
        for repeated parameter sets.

        Will raise a StopIteration exception if the previous 4
        iterations have had identical parameters (to 2 decimal places).

        Args:
            x: Optimizer parameter array (Unused).
        """

        # Number of iterations to check for convergence
        numIteration = 4
        # Decimal precision for parameter comparison
        precision = 2
        
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
        self.simFIMS.setGeometry(self.geoConfig)
        runNumber = self.simFIMS.runForIBNOptimizer()
        
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

    def _getEff(self):
        """
        Runs a FIMS simulation and determines the minimum field ratio
        needed for 95% efficiency.

        returns:
            minField (float): The minimum field ratio needed for 95% efficiency.
        """
        
        print(f'********** Iteration {len(self._optimizerLog)+1:<3}************')
        allParams = self.simFIMS.getAllParam()
        for elem in self.params:
            print(f'\t{elem}: {allParams[elem]}')
        print('************************************')
        self.simFIMS.setGeometry(self.geoConfig)
        minField = self.simFIMS.runForEffOptimizer()
        
        return minField

#**********************************************************************#

    def _effObjective(self, optimizerParam, inputList):
        """
        Objective function to optimize for efficiency.

        Updates the FIMS simulation with the given parameters, 
        runs the simulation, and returns the minimum field ratio.

        Assumes that field ratio is not one of the input parameters.
        I.e. The efficiency conditions are being satisfied internally
        by the simulation.

        Note that optimizerParam and inputList must be in the same order.
        
        Args:
            optimizerParam (np.array): Flat array of parameters.
            inputList (list): List of parameter names (in order).
        
        Returns:
            resultField (float): The minimum field ratio value.
        """
        runStart = time.perf_counter()
        
        # Unpack and Upload the optimizer parameters into the simulation
        paramDict = dict(zip(inputList, optimizerParam))
        unNormalizedDict = self._unNormalizeInputs(paramDict)
        self.simFIMS.setParameters(unNormalizedDict)
        
        # Run simulation and get the minimum field ratio
        fieldRatio = self._getEff()
        
        # Get time stamps
        runEnd = time.perf_counter()
        runTime = runEnd - runStart
        totalTime = runEnd - self._startTime
        
        # Update the optimizer log TODO: does this actually do anything?
        self._optimizerLog.append({
            'params': unNormalizedDict,
            'fieldRatio': fieldRatio
        })
        
        with open('log/logOptimizer.txt', 'a') as file:
                file.write(f'\nIteration {len(self._optimizerLog)}\n')
                for param, value in unNormalizedDict.items():
                    file.write(f'\t{param}: {value}\n')
                file.write(f'\tMinimum Field Ratio: {fieldRatio}\n')
                file.write(f'Run Time: {runTime}\n')
                file.write(f'Total Time: {totalTime}')
                
        # Print the current field ratio for this iteration
        print(f'\tIteration {len(self._optimizerLog)}: field ratio = {fieldRatio:.6f}\n')
        
        return fieldRatio


#**********************************************************************#

    def setGeometry(self, geometry):
        """
        Sets the shape of the geometry.
         
        args:
            geoConfig (dataClass: parameters defining the geometry to be generated.
                unitCell: shape of the unit cell.
                padShape: shape of the readout pad.
                holeShape: shape of the grid holes.
                scaleOption: scale of the geometry generated.
        """
        
        self.geoConfig = geometry

        return

#**********************************************************************#

    def optimizeForIBN(self, initialGuess={}):
        """
        Runs an optimization routine to find the FIMS parameters that 
        minimize the Ion Backflow Number (IBN).

        Utilizes the COBYQA optimization method (derivative-free).
        Bounds are set based on the input parameters. 
        Terminated based on criteria in _checkConvergence.
        Parameters are constrained to prevent unphysical combinations.
        
        Args:
            initialGuess (dict): dictionary of initial optimizer values
        
        Returns:
            dict: A dictionary containing:
                - params (dict): Optimal FIMS parameters.
                - IBNValue (float): Final minimum IBN value.
                - success (bool): Success status of minimization.
        """
        # Unpack optimizer parameters and bounds
        activeParameters = self.params.copy()        
        inputList = [name for name in activeParameters]
        minBounds = [min(activeParameters[name]) for name in inputList]
        maxBounds = [max(activeParameters[name]) for name in inputList]
        
        # Verify and set the initial guess
        self._setInitialParameters(initialGuess)
        initNormGuess = np.array([1 for elem in inputList])
        self.simFIMS.setParameters(self.initialGeometry)
        
        # Set the bounds for each variable
        initialValues = [self.initialGeometry[param] for param in inputList]
        normMinBounds = self._normalizeValues(initialValues, minBounds)
        normMaxBounds = self._normalizeValues(initialValues, maxBounds)
        optimizerBounds = Bounds(normMinBounds, normMaxBounds)

        print('Beginning optimization...')

        try:
            optimizerResult = minimize(
                fun=self._IBNObjective,
                x0=initNormGuess,
                args=(inputList,),
                method='COBYQA', #or 'Nelder-Mead'
                constraints=self._getGeometryConstraints(),
                callback=self._checkConvergence,
                bounds=optimizerBounds,
                options = {'initial_tr_radius': .2} # initial step of 20%
            )
            
            # Unpack optimizer output
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

    def optimizeForEfficiency(self, initialGuess={}):
        """
        Runs an optimization routine to find the FIMS parameters that 
        minimize the field ratio needed for 95% efficiency.

        Utilizes the COBYQA optimization method (derivative-free).
        Bounds are set based on the input parameters. 
        Terminated based on criteria in _checkConvergence.
        Parameters are constrained to prevent unphysical combinations.
        
        Args:
            initialGuess (dict): dictionary of initial optimizer values
        
        Returns:
            dict: A dictionary containing:
                - params (dict): Optimal FIMS parameters.
                - fieldValue (float): Final field ratio value.
                - success (bool): Success status of minimization.
        """
        # Unpack optimizer parameters and bounds
        activeParameters = self.params.copy()        
        inputList = [name for name in activeParameters]
        minBounds = [min(activeParameters[name]) for name in inputList]
        maxBounds = [max(activeParameters[name]) for name in inputList]
        
        # Verify and set the initial guess
        self._setInitialParameters(initialGuess)
        initNormGuess = np.array([1 for elem in inputList])
        self.simFIMS.setParameters(self.initialGeometry)
        
        # Set the bounds for each variable
        initialValues = [self.initialGeometry[param] for param in inputList]
        normMinBounds = self._normalizeValues(initialValues, minBounds)
        normMaxBounds = self._normalizeValues(initialValues, maxBounds)
        optimizerBounds = Bounds(normMinBounds, normMaxBounds)

        print('Beginning optimization...')

        try:
            optimizerResult = minimize(
                fun=self._effObjective,
                x0=initNormGuess,
                args=(inputList,),
                method='COBYQA', #or 'Nelder-Mead'
                constraints=self._getGeometryConstraints(),
                callback=self._checkConvergence,
                bounds=optimizerBounds,
                options = {'initial_tr_radius': .2} # initial step of 20%
            )
            
            # Unpack optimizer output
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
            'fieldValue': optimizerResult.fun, 
            'success': optimizerResult.success
        }
        
        print(f"Optimal IBN value = {resultVals['fieldValue']}")
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
    
