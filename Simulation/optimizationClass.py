###################################
# CLASS DEFINITION FOR OPTIMIZATION #
###################################
from __future__ import annotations

import os
import sys
import time
import numpy as np
import math
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf

# Torch precision
dtype = torch.double

simDir = os.getcwd()
analysisDir = os.path.join(simDir, '..', 'Analysis')
sys.path.append(analysisDir)

from simulationClass import FIMS_Simulation
from runDataClass import runData

#**********************************************************************#
#**********************************************************************#
class FIMS_Optimizer:
    """
        
    Class representing the FIMS optimization algorithm.
    
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
        self._iterationLimit = 25
        self._precisionLimit = 0.5
        
        # Set geometry configuration and values
        self.params = params
        self._setupScalings()

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
        self.curGeometry = self.initialGeometry.deepcopy()

        self.geoConfig = self.simFIMS._geoConfiguration
        self._checkParameters()
        
        # Create log file for optimizer
        os.makedirs('log', exist_ok=True)
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
        paramList = [f'{k}: {v}' for k, v in self.params.items()]
        paramString = ' '.join(paramList)
        return paramString

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
                raise ValueError(f'Error: {paramName} not a valid parameter.')
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
            self.initialGeometry[param] = value
        
        return

#**********************************************************************#
    def setIterationLimit(self, iterLim):
        """
        Sets the optimization iteration limit.
        Defaults to 25.
        """
        setLimit = -1
        if isinstance(iterLim, (int, float)):
            setLimit = round(iterLim)

        self._iterationLimit = setLimit if setLimit >= 1 else 25
        return

#**********************************************************************#
    def setPrecisionLimit(self, geoLimit):
        """
        Sets the precision limit of geometry values.
        """
        self._precisionLimit = geoLimit
        return

#**********************************************************************#
    def _setupScalings(self):
        octagonFactor = 2 * math.cos(math.radians(67.5))

        self.holeShapeFactors = {
            'circle': (-2, -2),
            'hexagon': (-math.sqrt(3), -2),
            'octagon': (-2.0173, -octagonFactor),
            'triangle': (-2, -2),
            'kiki': (-math.sqrt(3), -2),
            'nesteggs': (-7.1, -7.1),
            'trivialpursuit': (-4.4, -4.4)
        }

        self.padShapeFactors = {
            'square': (-4 / math.sqrt(3), -1),
            'hexagon': (-math.sqrt(3), -2),
            'octagon': (-1.9601, -octagonFactor),
        }

        return
        
#**********************************************************************#
    def _getBoTorchConstraints(self, inputList):
        """
        Constructs BoTorch linear inequality constraints for unit cell geometries.
        
        Enforces:
            pitch >= hole
            pitch >= pad size
        """

        constraints = []

        #Buffer distance - TODO can restrict for pillars here
        buffer = 1 #1 um buffer

        #Get unit cell geometry
        hexCell = 'Hexagonal' in getattr(self.geoConfig, 'unitCell', 'Hexagonal')
        hexID = 0 if hexCell else 1

        holeShape = getattr(self.geoConfig, 'holeShape', 'circle')
        padShape = getattr(self.geoConfig, 'padShape', 'square')
        holeScale = -1.0 * self.holeShapeFactors.get(holeShape, (-2.0, -2.0))[hexID]
        padScale = -1.0 * self.padShapeFactors.get(padShape, (-2.0, -2.0))[hexID]

        hasPitch = 'pitch' in inputList
        hasHole = 'holeRadius' in inputList
        hasPad = 'padLength' in inputList

        pitchID = inputList.index('pitch') if hasPitch else -1
        holeID = inputList.index('holeRadius') if hasHole else -1
        padID = inputList.index('padLength') if hasPad else -1


        ##### Hole Size Constraint #####
        if hasPitch and hasHole: #Pitch and hole
            constraints.append((
                torch.tensor([pitchID, holeID], dtype=torch.long),
                torch.tensor([1.0, -holeScale], dtype=torch.double),
                buffer
            ))
        elif hasHole and not hasPitch: #fixed pitch
            pitch = self.initialGeometry['pitch']
            constraints.append((
                torch.tensor([holeID], dtype=torch.long),
                torch.tensor([-holeScale], dtype=torch.double),
                buffer - pitch
            ))
        elif hasPitch and not hasHole: #fixed hole
            holeRadius = self.initialGeometry['holeRadius']
            constraints.append((
                torch.tensor([pitchID], dtype=torch.long),
                torch.tensor([1.0], dtype=torch.double),
                holeScale*holeRadius + buffer
            ))


        ##### Pad Constraint #####
        if hasPitch and hasPad: #Pitch and pad
            constraints.append((
                torch.tensor([pitchID, padID], dtype=torch.long),
                torch.tensor([1.0, -padScale], dtype=torch.double),
                buffer
            ))
        elif hasPad and not hasPitch: #fixed pitch
            pitch = self.initialGeometry['pitch']
            constraints.append((
                    torch.tensor([padID], dtype=torch.long),
                    torch.tensor([-padScale], dtype=torch.double),
                    buffer - pitch
                ))
        elif hasPitch and not hasPad: #fixed pitch
            padLength = self.initialGeometry['padLength']
            constraints.append((
                torch.tensor([pitchID], dtype=torch.long),
                torch.tensor([1.0], dtype=torch.double),
                padScale*padLength + buffer
            ))


        return constraints if len(constraints) > 0 else None

#**********************************************************************#
    def _getIBN(self):
        """
        Runs a FIMS simulation and calculates
        the Ion Backflow Number (IBN) from the results.

        Returns:
            Average IBN and its standard error.
        """
        
        self.simFIMS.setGeometry(self.geoConfig)
        self.simFIMS.setParameters(self.curGeometry)
        runNumber = self.simFIMS.runForIBNOptimizer()
        
        # Get the IBN
        simData = runData(runNumber)
        IBN = simData.getCalcParameter('Average IBN')
        IBNError = simData.getCalcParameter('IBN Error')

        return IBN, IBNError

#**********************************************************************#
    def _IBNObjective(self, optimizerParam, inputList):
        """
        Objective function to optimize for minimum IBN.

        Updates the FIMS simulation with the given parameters, 
        runs the simulation, and returns the resulting IBN.

        Assumes that field ratio is not one of the input parameters.

        Note that optimizerParam and inputList must be in the same order.
        
        Args:
            optimizerParam (np.array): Flat array of parameters.
            inputList (list): List of parameter names (in order).
        
        Returns:
            resultIBN (float): The current IBN value.
            resultIBNError (flaot): The SEM of the IBN.
        """
        runStart = time.perf_counter()

        # Unpack and Upload the optimizer parameters
        paramDict = dict(zip(inputList, optimizerParam))
        self.curGeometry.update(paramDict)

        try:
            resultIBN, resultIBNError = self._getIBN()
        except Exception as e:
            print(f'Simulation failed for params {paramDict}: {e}')
            # Failure penalty: high mean IBN, large variance
            resultIBN, resultIBNError = 100.0, 0.1
        
        # Get time stamps
        runEnd = time.perf_counter()
        runTime = runEnd - runStart
        totalTime = runEnd - self._startTime
        
        # Update the optimizer log
        self._optimizerLog.append({
            'params': paramDict,
            'IBN': resultIBN, 
            'IBNError': resultIBNError
        })

        with open('log/logOptimizer.txt', 'a') as file:
                file.write(f'\nIteration {len(self._optimizerLog)}\n')
                for param, value in paramDict.items():
                    file.write(f'\t{param}: {value}\n')
                file.write(f'\tIBN: {resultIBN}\n')
                file.write(f'Run Time: {runTime}\n')
                file.write(f'Total Time: {totalTime}')
                
        # Print the current IBN value for this iteration
        print(f'\tIteration {len(self._optimizerLog)}: IBN = {resultIBN:.6f} +/- {resultIBNError:.6f}\n')
        
        return resultIBN, resultIBNError

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
    def _snapToPrecision(self, geoValues):
        """
        Round geometry values to the nearest step size
        """
        stepSize = self._precisionLimit
        roundedValues = np.round(geoValues/stepSize)*stepSize
        return roundedValues

#**********************************************************************#
    def optimizeForIBN(self, initialGuess={}):
        """
        Runs an optimization routine to find the FIMS parameters that 
        minimize the Ion Backflow Number (IBN).

        Uses Bayesian optimization via BoTorch.
        
        Args:
            initialGuess (dict): dictionary of initial optimizer values
        
        Returns:
            dict: A dictionary containing:
                - params (dict): Optimal FIMS parameters.
                - IBNValue (float): Final minimum IBN value.
                - IBNError (float): Error of IBN.
                - success (bool): Success status of minimization.
        """
        # Unpack optimizer parameters and bounds
        inputList = list(self.params.keys())
        minBounds = [self.params[name][0] for name in inputList]
        maxBounds = [self.params[name][1] for name in inputList]
        
        # Verify and set the initial guess
        self._setInitialParameters(initialGuess)
        self.simFIMS.setParameters(self.initialGeometry)
        
        # Set the constraints
        lower = torch.tensor(minBounds, dtype=dtype)
        upper = torch.tensor(maxBounds, dtype=dtype)
        boundsTensor = torch.stack([lower, upper])
        inequalityConstraints = self._getBoTorchConstraints(inputList)

        print('Beginning BoTorch optimization...')

        #Initial Design
        numInit = max(5, len(inputList)*3)
        rawValues = lower + (upper - lower) * torch.rand(numInit, len(inputList), dtype=dtype)
        snapValues = self._snapToPrecision(rawValues)

        inValues = torch.tensor(snapValues, dtype=dtype)

        IBNList, IBNVarList = [], []
        for i in range(inValues.shape[0]):
            valueIBN, errorIBN = self._IBNObjective(inValues[i].numpy(), inputList)
            IBNList.append([-valueIBN]) #BoTorch tries to maximize, so invert
            IBNVarList.append([errorIBN**2])

        inResults = torch.tensor(IBNList, dtype=dtype)
        inResultsVar = torch.tensor(IBNVarList, dtype=dtype)

        finalParams = self.initialGeometry
        finalIBN = None
        finalIBNErr = None
        finalStatus = False

        try:
            # Optimization steps
            for inIter in range(self._iterationLimit):
                gp = SingleTaskGP(inValues, inResults, train_Yvar=inResultsVar)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

                aqcFunction = qLogNoisyExpectedImprovement(
                    model=gp,
                    X_baseline=inValues,
                    prune_baseline=True
                )

                # Optimize continuous surrogate acquisition function
                candidateContinuous, _ = optimize_acqf(
                    acq_function=aqcFunction,
                    bounds=boundsTensor,
                    inequality_constraints=inequalityConstraints,
                    q=1,
                    num_restarts=10,
                    raw_samples=512,
                )

                # Snap candidate point to precision
                candidates = candidateContinuous[0].detach().cpu().numpy()
                snapCandidates = self._snapToPrecision(candidates)

                #Get new IBN values
                newIBN, newIBNErr = self._IBNObjective(snapCandidates, inputList)

                # Append grid tensor back into dataset
                candidateTensor = torch.tensor([snapCandidates], dtype=dtype)
                inValues = torch.cat([inValues, candidateTensor], dim=0)
                inResults = torch.cat([inResults, torch.tensor([[-newIBN]], dtype=dtype)], dim=0) #BoTorch tries to maximize, so invert
                inResultsVar = torch.cat([inResultsVar, torch.tensor([[newIBNErr**2]], dtype=dtype)], dim=0)

            bestIDx = torch.argmax(inResults)
            finalParams = dict(zip(inputList, inValues[bestIDx].numpy()))
            finalIBN = -inResults[bestIDx].item()
            finalIBNErr = math.sqrt(inResultsVar[bestIDx].item())
            finalStatus = True

        except Exception as e:
            print(f'BoTorch optimization failed: {e}')
            lastLog = self._optimizerLog[-1] if self._optimizerLog else {'params': self.initialGeometry, 'fieldRatio': None}
            finalParams = lastLog['params']
            finalStatus = False


        print('\n*************** Optimization Complete ***************')
        fullFinalParams = self.initialGeometry.copy()
        fullFinalParams.update(finalParams)
        self.simFIMS.setParameters(fullFinalParams)
        
        resultVals = {
            'params': self.simFIMS.getAllParam(),
            'IBNValue': finalIBN,
            'IBNError': finalIBNErr,
            'success': finalStatus,
        }
        
        print(f"Optimal IBN value = {resultVals['IBNValue']} +/- {resultVals['IBNError']}")
        print(self.simFIMS)
        
        return resultVals
    
