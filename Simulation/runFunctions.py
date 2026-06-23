import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import pandas as pd
import time
import itertools

simDir = os.getcwd()
analysisDir = os.path.join(simDir, '..', 'Analysis')
sys.path.append(analysisDir)

from simulationClass import FIMS_Simulation
from gasSimulationClass import gasSimulation
from runDataClass import runData

#********************************************************************************#   
def timeSimulation():
    """
    Executes a simulation with the default parameters for runtime checks.
    """

    print('Timing FIMS simulation with default parameters...')
    startTime = time.monotonic()
    defaultSim = FIMS_Simulation()
    defaultSim.runSimulation()
    endTime = time.monotonic()
    totalTime = endTime - startTime
    print(f'\Completed in: {totalTime} s.')

    return 
    
#********************************************************************************#   
def parameterScan(parameterDefinitions):
    """
    Performs an N-dimensional parameter sweep, running the full simulation pipeline
    for each combination of specified parameters.

    Args:
        parameterDefinitions (list): List of elements with form: ['parameter', [values]]
        - parameter (string): Simulation parameter name to be iterated.
        - values (iterable): Values of interest for the given parameter.

    Returns:
        list: List of the run numbers for the executed simulations
    """

    simFIMS = FIMS_Simulation()
    allParams = simFIMS.getAllParam()

    params = []
    values = []
    numRuns = 1

    for parameter, targetValues in parameterDefinitions:
        
        if parameter not in allParams:
            raise ValueError(f'Invalid parameter - {parameter}')
        
        inRange = list(targetValues)
        params.append(parameter)
        values.append(inRange)
        numRuns *= len(inRange)

    print(f'Beginning {numRuns} simulations...')
    runNos = []
    for inValues in itertools.product(*values):
        simFIMS.resetParam(verbose=False)

        inParam = dict(zip(params, inValues))
        simFIMS.setParameters(inParam)
        runNumber = simFIMS.runSimulation()
        runNos.append(runNumber)

    return runNos





    

    


