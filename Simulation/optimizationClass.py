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
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

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
        
    Class representing the FIMS optimization algorithm.
    
    Utilizes scipy.optimize's minimize method with the COBYQA method 
    to minimize a target parameter. 
    
    Note - Currently only accepts as inputs:
        hole radius, 
        pitch, 
        amplification distance (grid standoff height), 
        pad lengthand 
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
            params (dict): Dictionary where keys are parameter names 
                           and values are lists of [Minimum, Maximum].
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

        self._setupScalings()
        
        # Create log file for optimizer
        try:
            with open('log/logOptimizer.txt', 'w') as file:
                file.write('## FIMS Optimization Log ##\n')
        except OSError as e:
            raise FileNotFoundError(f'Unable to create log file: {e}') from e
        
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
        if self.params is None:
            raise ValueError('Error - No parameters provided.')
            
        allowedParams = [
            'holeRadius', 
            'amplificationGap', 
            'padLength', 
            'pitch',
            'fieldRatio'
        ]
        paramCopy = self.params.copy()

        for paramName, bounds in paramCopy.items():
            if paramName not in allowedParams:
                raiseValueError(f'Error: {name} not a valid parameter.')
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f'Error: Invalid bounds {bounds}.')

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

        for param, value in initialGuess.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f'Error: {param} value is not valid. Must be a number.')
        
        # Update default values with given values, if any provided
        for geo, value in initialGuess.items():
            self.initialGeometry[geo] = value
        
        return

#**********************************************************************#
    def _setupScalings(self):
        """
        Set up scaling factors for various geometries
        """
        octagonFactor = 2 * math.cos(math.radians(67.5))
        kikiFactor = math.sqrt(3)

        self.holeShapeFactors = {
            'circle':  (-2, -2),
            'hexagon': (-math.sqrt(3), -2),
            'octagon': (-2.0173, -octagonFactor),
            'triangle': (-2, -2),
            'kiki': (-math.sqrt(3), -2),
            'nesteggs': (-7.1, -7.1),
            'trivialpursuit': (-4.4, -4.4)
        }
    
        self.padShapeFactors = {
            'square':  (-4 / math.sqrt(3), -1),
            'hexagon': (-math.sqrt(3), -2),
            'octagon': (-1.9601, -octagonFactor),
        } 
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
    
        # Get all geometry values
        initGeometry = {key: self.initialGeometry[key] for key in self.initialGeometry}
        hexCell = 'Hexagonal' in self.geoConfig.unitCell
        hexID = 0 if hexCell else 1
        
        # Apply geometry-dependent multipliers
        holeShape = self.geoConfig.holeShape
        if holeShape in self.holeShapeFactors:
            initGeometry['holeRadius'] *= self.holeShapeFactors[holeShape][hexID]
    
        padShape = self.geoConfig.padShape
        if padShape in self.padShapeFactors:
            initGeometry['padLength'] *= self.padShapeFactors[padShape][hexID]
        
        # Set Other geometry parameters
        initGeometry['pillarRadius'] = 0 # TODO: include pillar
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
            return [LinearConstraint(np.empty((0, numParams)), [], [], keep_feasible=True)]
        
        geometryConstraints = [LinearConstraint(
            A=np.array(matrixRows),
            lb=np.array(lowerBounds),
            ub=np.full(len(lowerBounds), np.inf),
            keep_feasible=True
        )]
        
        return geometryConstraints

#**********************************************************************#

    def _projectValue(self, paramName, scale, valDict, varDict):
        """
        Takes a given parameter and projects it into a feasible region.
        
        args:
            paramName (str): name of the parameter to be checked.
            scale (float): how the parameter scales.
            valDict (dict): dictionary of values for all parameters.
            varDict (dict): dictionary of variabels.
        
        returns:
            valDict: the new values for the parameters.
        """
        buffer = 0.98
        maxVal = buffer * valDict['pitch'] / scale
        
        if valDict[paramName] <= maxVal:
            return valDict

        if paramName in varDict:
            valDict[paramName] = maxVal
            print(f'Warning: {paramName} larger than cell. Changed to {maxVal}')
        
        elif 'pitch' in varDict:
            newPitch = valDict[paramName] * scale * (2-buffer)
            valDict['pitch'] = newPitch
            print(f'Warning: {paramName} larger than cell. Pitch set to {newPitch}')
        
        else:
            raise ValueError(
                f'Error: {paramName} larger than cell AND constant. '
                f'Adjust value and then restart the optimizer.'
            )
        
        return valDict

    #**********************************************************************#

    def _constraintFailSafe(self, geoDict):
        """
        Ensures geometry values are feasible. Adjusts them if not.

        args:
            geoDict (dict): dictionary of optimizer values

        returns: 
            verifiedDict (dict): dictionary of optimizer values that are
        verified to be feasible.
        """
        # Get all geometry values
        hexID = 0 if 'Hexagonal' in self.geoConfig.unitCell else 1
        parameters = ['pitch', 'holeRadius', 'padLength']
        valuesDict = {
            key: geoDict[key] if key in geoDict else self.simFIMS.getParam(key)
            for key in parameters
        }    

        # Apply geometry-dependent multipliers
        holeShape = self.geoConfig.holeShape
        if holeShape in self.holeShapeFactors:
            holeScale = -1 * self.holeShapeFactors[holeShape][hexID]
        
        padShape = self.geoConfig.padShape
        if padShape in self.padShapeFactors:
            padScale = -1 * self.padShapeFactors[padShape][hexID]
        
        # Grid hole must be smaller than the cell size
        verifiedDict = self._projectValue('holeRadius', holeScale, valuesDict, geoDict)    
        
        # Pad size must be smaller than the cell size
        verifiedDict = self._projectValue('padLength', padScale, verifiedDict, geoDict) 
    	    
        return verifiedDict

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
        verifiedGeo = self._constraintFailSafe(unNormalizedDict)
        self.simFIMS.setParameters(verifiedGeo)
        
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
        verifiedGeo = self._constraintFailSafe(unNormalizedDict)
        self.simFIMS.setParameters(verifiedGeo)
        
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

        print('Beginning BoTorch optimization...')

        # Prepare normalized bounds for BoTorch (torch tensors)
        lower = torch.tensor(normMinBounds, dtype=torch.double)
        upper = torch.tensor(normMaxBounds, dtype=torch.double)

        # Objective wrapper for BoTorch: accepts 2D tensor (q x d) and returns 2D tensor (q x 1)
        def _torchObj(xTorch: torch.Tensor) -> torch.Tensor:
            # xTorch is shape (q, d)
            xNp = xTorch.detach().cpu().numpy()
            ys = []
            for row in xNp:
                try:
                    y = float(self._IBNObjective(row, inputList))
                except Exception:
                    y = float('nan')
                ys.append([y])
            return torch.tensor(ys, dtype=torch.double)

        # Run a simple sequential BO loop
        try:
            best_x = None
            best_y = float('inf')

            # initial design: random samples
            nInit = max(5, min(20, len(inputList)*3))
            xInit = lower + (upper - lower) * torch.rand(nInit, len(inputList), dtype=torch.double)
            yList = []
            for i in range(xInit.shape[0]):
                y = float(self._IBNObjective(xInit[i].numpy(), inputList))
                yList.append([y])
                if y < best_y:
                    best_y = y
                    best_x = xInit[i].clone()

            X = xInit.clone()
            Y = torch.tensor(yList, dtype=torch.double)

            n_iter = 25
            for it in range(n_iter):
                # fit GP
                gp = SingleTaskGP(X, Y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

                # acquisition
                ei = ExpectedImprovement(model=gp, best_f=Y.min())

                # optimize acquisition over bounds
                candidate, _ = optimize_acqf(
                    acq_function=ei,
                    bounds=torch.stack([lower, upper]),
                    q=1,
                    num_restarts=5,
                    raw_samples=20,
                )

                # ensure candidate respects geometry constraints by projecting
                candNp = candidate.detach().cpu().numpy().ravel()
                candUn = self._unNormalizeInputs(dict(zip(inputList, candNp)))
                candProj = self._constraintFailSafe(candUn)
                # re-normalize
                candNorm = torch.tensor([candProj[p] / self.initialGeometry[p] for p in inputList], dtype=torch.double)

                yNew = float(self._IBNObjective(candNorm.numpy(), inputList))
                X = torch.cat([X, candNorm.unsqueeze(0)], dim=0)
                Y = torch.cat([Y, torch.tensor([[yNew]], dtype=torch.double)], dim=0)

                if yNew < best_y:
                    best_y = yNew
                    best_x = candNorm.clone()

                # simple convergence check
                if len(Y) > 10 and torch.isclose(Y[-1], Y[-5], rtol=1e-4, atol=1e-6):
                    break

            finalDict = dict(zip(inputList, best_x.numpy()))
            finalParams = self._unNormalizeInputs(finalDict)
            finalFunction = best_y
            finalStatus = True

        except Exception as e:
            print('BoTorch optimization failed:', e)
            lastLog = self._optimizerLog[-1] if self._optimizerLog else {'params': self.initialGeometry, 'IBN': None}
            finalParams = lastLog['params']
            finalFunction = lastLog.get('IBN', None)
            finalStatus = False

        print('\n*************** Optimization Complete ***************')
        # Put results into simulation instance
        self.simFIMS.setParameters(finalParams)
        
        resultVals = {
            'params': self.simFIMS.getAllParam(),
            'IBNValue': finalFunction,
            'success': finalStatus,
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

        print('Beginning BoTorch optimization...')

        # Prepare normalized bounds for BoTorch (torch tensors)
        lower = torch.tensor(normMinBounds, dtype=torch.double)
        upper = torch.tensor(normMaxBounds, dtype=torch.double)

        def _torchEffObj(xTorch: torch.Tensor) -> torch.Tensor:
            xNp = xTorch.detach().cpu().numpy()
            ys = []
            for row in xNp:
                try:
                    y = float(self._effObjective(row, inputList))
                except Exception:
                    y = float('nan')
                ys.append([y])
            return torch.tensor(ys, dtype=torch.double)

        try:
            best_x = None
            best_y = float('inf')

            nInit = max(5, min(20, len(inputList)*3))
            xInit = lower + (upper - lower) * torch.rand(nInit, len(inputList), dtype=torch.double)
            yList = []
            for i in range(xInit.shape[0]):
                y = float(self._effObjective(xInit[i].numpy(), inputList))
                yList.append([y])
                if y < best_y:
                    best_y = y
                    best_x = xInit[i].clone()

            X = xInit.clone()
            Y = torch.tensor(yList, dtype=torch.double)

            n_iter = 25
            for it in range(n_iter):
                gp = SingleTaskGP(X, Y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

                ei = ExpectedImprovement(model=gp, best_f=Y.min())
                candidate, _ = optimize_acqf(
                    acq_function=ei,
                    bounds=torch.stack([lower, upper]),
                    q=1,
                    num_restarts=5,
                    raw_samples=20,
                )

                candNp = candidate.detach().cpu().numpy().ravel()
                candUn = self._unNormalizeInputs(dict(zip(inputList, candNp)))
                candProj = self._constraintFailSafe(candUn)
                candNorm = torch.tensor([candProj[p] / self.initialGeometry[p] for p in inputList], dtype=torch.double)

                yNew = float(self._effObjective(candNorm.numpy(), inputList))
                X = torch.cat([X, candNorm.unsqueeze(0)], dim=0)
                Y = torch.cat([Y, torch.tensor([[yNew]], dtype=torch.double)], dim=0)

                if yNew < best_y:
                    best_y = yNew
                    best_x = candNorm.clone()

                if len(Y) > 10 and torch.isclose(Y[-1], Y[-5], rtol=1e-4, atol=1e-6):
                    break

            finalDict = dict(zip(inputList, best_x.numpy()))
            finalParams = self._unNormalizeInputs(finalDict)
            finalFunction = best_y
            finalStatus = True

        except Exception as e:
            print('BoTorch optimization failed:', e)
            lastLog = self._optimizerLog[-1] if self._optimizerLog else {'params': self.initialGeometry, 'IBN': None}
            finalParams = lastLog['params']
            finalFunction = lastLog.get('IBN', None)
            finalStatus = False
        print('\n*************** Optimization Complete ***************')
        # Put results into simulation instance
        self.simFIMS.setParameters(finalParams)
        
        resultVals = {
            'params': self.simFIMS.getAllParam(),
            'fieldValue': finalFunction,
            'success': finalStatus,
        }
        
        print(f"Optimal Field value = {resultVals['fieldValue']}")
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
    
