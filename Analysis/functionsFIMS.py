import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import pandas as pd

from scipy.special import gammaincc



"""
Functions:
    getAnalysisNumbers
    plotGeneralPolya
    plotPolya
    plotPolyaEfficiency
    plotThreshold
    plotPolyaExamples
    withinHex
    withinNeighborHex
    xyInterpolate
    getSetData
    plotDataSets
    getDiffusionData            <-----NEW
"""

#********************************************************************************#   
def getAnalysisNumbers():
    """
    Reads a list of run numbers to analyzer from a file.

    Assumes filename is 'analysisRunNumbers'.
    If file does not exist, it is created and initialized with a '-1'.

    Returns:
        list[int]: List of integers representing the run number to be analyzed.
                   Empty if an error occurs.
    """
    filename = 'analysisRunNumbers'

    if not os.path.exists(filename):
        with open(filename, "w") as file:
            file.write('-1')
            print(f"File '{filename}' created with default -1.")
            return []

    allRunnos = []
    try:
        with open(filename, 'r') as file:
            for line in file:

                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    runNo = int(line.strip())
                    if runNo == -1:
                        continue
                    allRunnos.append(runNo)   
                    
                except ValueError:
                    print(f"Warning. Skipping non-integer line in '{filename}'.")
                    
    except Exception as e:
        print(f"An unexpected error occurred while reading '{filename}': {e}")
        return []

    return allRunnos
    
#********************************************************************************#   
def plotGeneralPolya(theta):
    """
    Plots the general Polya distribution for a given set of theta values.

    The x-axis represents the normalized avalanche size ($n/\bar{n}$),
    and the y-axis represents the normalized probability ($\bar{n}$ x Probability).

    Args:
        theta (float): List or numpy array of values to use as 
                       theta in Polya calculations.
    """
    from polyaClass import myPolya

    n = np.linspace(0, 4, 101)
    plt.figure(figsize=(6, 4))
    
    for t in theta:
        generalPolya = myPolya(1, t)
        polyaProb = generalPolya.calcPolya(n)
        plt.plot(n, polyaProb,
                 label=r'$\theta$'+f' = {t:.2f}')
        
    plt.title(f"General Polya Distribution")
    plt.xlabel(r'Avalanche Size ($n/\bar{n}$)')
    plt.ylabel(r'$\bar{n}$ x Probability')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    return

#********************************************************************************#   
def plotPolya(theta):
    """
    Generates and plots Polya distributions for various gain and theta values.

    Each subplot corresponds to a single theta with various gain values.

    Args:
        theta (float): List or numpy array of values to use as 
                       theta in Polya calculations.
    """
    from polyaClass import myPolya

    gain = [10, 25, 50, 75, 100]

    n = np.arange(0, 101, 1)
    
    numPlots = len(theta)
    numRows = int(np.ceil(numPlots/2))
    
    # Create the figure and add subplots
    fig, axes = plt.subplots(nrows=numRows, ncols=2, figsize=(12, 5*numRows))
    axesFlat = axes.flatten()

    fig.suptitle(f'Polya Avalanches')

    for i, t in enumerate(theta):
        for nBar in gain:
            plotPolya = myPolya(nBar, t)
            polyaProb = plotPolya.calcPolya(n)
            axesFlat[i].plot(n, polyaProb,
                             label=r'$\bar{n}$'+f' = {nBar:.0f}')

        axesFlat[i].set_title(r'$\theta$'+f' = {t:0.2f}')
        axesFlat[i].set_xlabel('Avalanche size')
        axesFlat[i].set_ylabel('Probability')
        axesFlat[i].legend()
        axesFlat[i].grid(True, alpha=0.5)

    for j in range(numPlots, len(axesFlat)):
        fig.delaxes(axesFlat[j])
    plt.show()
    return

#********************************************************************************#   
def plotPolyaEfficiency(theta):
    """
    Plots the efficiency of the Polya distribution as a function of the
    threshold-to-gain ratio (threshold / gain).

    Includes reference lines for 95% efficiency for the theta=0 case.
    
    Args:
        theta (float): List or numpy array of values to use as 
                       theta in Polya calculations.
    """
    k = np.linspace(0, 1, 101) #Ratio: Threshold/Gain

    plt.figure(figsize=(12, 5))

    for t in theta:
        efficiency = gammaincc(t+1, k*(t+1))
        plt.plot(k, efficiency,
                 label=r"$\theta$"+f" = {t:0.2f}")

    targetEfficiency = 0.95
    plt.axhline(y=targetEfficiency,
                c='r', ls='--', label=f'{targetEfficiency*100:.0f}% Efficiency')
    plt.axvline(x=-np.log(targetEfficiency),
                c='r', ls=':', label=r'$\theta = 0$ Limit: '+f'{-np.log(targetEfficiency):.3f}')

    plt.title(f"Parameterized Efficiency: "
              +r"$\eta = \frac{\Gamma\left(\theta+1, (\theta+1)*n_{t}/\bar{n}\right)}{\Gamma\left(\theta+1\right)}$")
    plt.xlabel("Threshold / Gain Fraction: "
               +r"$n_{t} / \bar{n}$")
    plt.ylabel(f"Efficiency")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

    return

#********************************************************************************#   
def plotThreshold():
    """
    Plots the minimum gain required to achieve specific target efficiencies
    as a function of detector threshold. 
    
    Include the theta=0 case as the maximum, and several other low-theta results.
    Efficiencies are 95% and 90%.
    """
    from polyaClass import myPolya

    threshold = np.linspace(0, 16, 11)
    efficiency = [.95, .9]

    colors = ['b', 'r', 'g']

    plt.figure(figsize=(6, 4))
    
    for i, eff in enumerate(efficiency):
        gain = -threshold/np.log(eff)
        plt.plot(threshold, gain,
                 c=colors[i], label=r'$\theta$ = 0.0 '+f'(Efficiency = {eff*100:.0f}%)')

        polya5 = myPolya(1, 0.5)
        polya5.solveForGain(targetEff=eff, threshold=1)
        theta5 = threshold*polya5.gain
        plt.plot(threshold, theta5,
                 c=colors[i], ls=':', label=r'$\theta$ = 0.5')

        polya1 = myPolya(1, 1)
        polya1.solveForGain(targetEff=eff, threshold=1)
        theta1 = threshold*polya1.gain
        plt.plot(threshold, theta1,
                 c=colors[i], ls='--', label=r'$\theta$ = 1.0')

        polya2 = myPolya(1, 2)
        polya2.solveForGain(targetEff=eff, threshold=1)
        theta2 = threshold*polya2.gain
        plt.plot(threshold, theta2,
                 c=colors[i], ls='-.', label=r'$\theta$ = 2.0')
                 

    plt.title(f'Minimum Gain Required to Achieve Efficiency')
    plt.xlabel('Detector Threshold')
    plt.ylabel('Gain')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

    return

#********************************************************************************#   
def plotPolyExamples(thetaStart=0, thetaEnd=5, numSteps=6):
    """
    Generates a series of plots illustrating various aspects of 
    the Polya distribution for a given range of theta-values.

    Args:
        thetaStart (int or float): The starting value for the theta range.
        thetaEnd (int or float): The ending value for the theta range.
        numSteps (int): The number of steps to generate within the theta range. 
    """
    theta = np.linspace(thetaStart, thetaEnd, numSteps)

    plotGeneralPolya(theta)
    plotPolya(theta)
    plotPolyaEfficiency(theta)
    plotThreshold()

    return
    
#********************************************************************************#   
def withinHex(xVal, yVal, sideLength):
    """
    Determines if a coordinate lies within a regular hexagon.
    Assumes a flat-top geometry centered at the origin.

    Args:
        xVal (float): The x-coordinate to check.
        yVal (float): The y-coordiante to check.
        sideLength (float): The length of a side of the hexagon.

    Returns:
        bool: True if the coordiate is within the hexagon, False otherwise.
    """
    #Use symmetry of regular hexagon
    x = np.abs(xVal)
    y = np.abs(yVal)

    #Check if below flat top
    checkTop = y <= sideLength*math.sqrt(3)/2.

    #Check if the point is within the sloped edge
    checkSlope = x+y/math.sqrt(3) <= sideLength

    #Combine conditions
    inHex = np.logical_and(checkTop, checkSlope)

    return inHex

#********************************************************************************#   
def withinNeighborHex(xVal, yVal, sideLength, pitch):
    """
    Determines if a coordinate lies within a hexagonal region in hexagonal tiling.
    Assumes a flat-top geometry. 
    Possible uses: 
        sideLength = side length of the unit cell - Determines if in neighbor cell.
        sideLength = side length of the pad - Determines if in neighbor pad.

    Args:
        xVal (float): The x-coordinate to check.
        yVal (float): The y-coordiante to check.
        sideLength (float): The length of a side of the hexagon to check.
        pitch (float): The spacing between the hexagonal tiling.
    
    Returns:
        bool: True if is in neighboring region, otherwise False.
    """
    # Use symmetry of tiling - Only need to check above and top-right
    x = np.abs(xVal)
    y = np.abs(yVal)

    #Unit cell dimensions
    inRadius = pitch/2.
    outRadius = 2*inRadius/math.sqrt(3)
    
    #Centers of neighboring cells
    neighborX = 3./2.*outRadius*np.array([0, 1])
    neighborY = inRadius*np.array([2, 1])

    #Check
    checkTop = withinHex(x - neighborX[0], y - neighborY[0], sideLength)
    checkTopRight = withinHex(x - neighborX[1], y - neighborY[1], sideLength)

    #Combine conditions
    isInNeighborHex = np.logical_or(checkTop, checkTopRight)

    return isInNeighborHex


#********************************************************************************#   
def xyInterpolate(point1, point2, zTarget):
    """
    Linear interpolation between two points for a target z-value.

    Args:
        point1 (tuple): x,y,z coordinates of the first point.
        point2 (tuple): x,y,z coordinates of the second point.
        zTarget (float): The target z-value for the interpolation.

    Returns:
        tuple: Interpolated x,y,z coordinates. None if points are at the same z.
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # Cannot interpolate if z-values are the same
    if z1 == z2:
        return None
    
    if not (z1 <= zTarget <= z2):
        raise ValueError('Target is outside of interpolation range.')

    #Interpolation requires points to be increasing
    if z1 > z2:
        z1, z2 = z2, z1
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    x = np.interp(zTarget, (z1, z2), (x1, x2))
    y = np.interp(zTarget, (z1, z2), (y1, y2))

    return (x, y, zTarget)

#********************************************************************************#
def getSetData(runList, xVal, yVal):
    """
    Retrieves and organizes parameter data from a list of runs.

    Args:
        runList (list): A list of run numbers for a given trial.
        xVal (str): The name of the parameter to use for the x-axis.
        yVal (str): The name of the parameter to use for the y-axis.

    Returns:
        tuple: A tuple containing two lists: (xData, yData).
               Each list contains the parameter values for the specified runs.
    """
    from runDataClass import runData

    xData = []
    yData = []
    for inRun in runList:
        simData = runData(inRun)

        xData.append(simData.getRunParameter(xVal))
        yData.append(simData.getRunParameter(yVal))

    return xData, yData


#********************************************************************************#
def plotDataSets(dataSets, xVal, yVal, savePlot=False):
    """
    Generates a scatter plot comparing multiple simulation trials.

    Each data set is plotted on the same figure for direct comparison. 
    The plot can be optionally saved to a 'Plots' directory.

    Args:
        dataSets (dict): A dictionary where keys are trial labels (strings) and
                         values are lists of corresponding run numbers.
        xVal (str): Parameter name for the x-axis.
        yVal (str): Parameter name for the y-axis.
        savePlot (bool): Saves plot as a PNG file if True.
    """
    from simulationClass import FIMS_Simulation

    if savePlot and not os.path.exists('./Plots'):
        os.makedirs('./Plots')

    #Check if valid parameters
    simulation = FIMS_Simulation()
    allParams = simulation.defaultParam()
    if xVal not in allParams or yVal not in allParams:
        raise ValueError(f'Error: Invalid parameter specified.')

    #Add units to axis labels if dimensional
    dimensionalParam = [
        'Pad Length',
        'Pitch',
        'Grid Standoff',
        'Grid Thickness',
        'Hole Radius',
        'Cathode Height',
        'Thickness SiO2',
        'Field Bundle Radius'
    ]
    xLabel = xVal + ' (um)' if xVal in dimensionalParam else xVal
    yLabel = yVal + ' (um)' if yVal in dimensionalParam else yVal

    # Make plot and add data
    fig, ax = plt.subplots()
    
    for inTrial, runList in dataSets.items():
        xData, yData = getSetData(runList, xVal, yVal)
        ax.scatter(
            xData,
            yData,
            label=inTrial, 
        )

    ax.set_title(f'{yVal} vs. {xVal}')
    ax.set_xlabel(f'{xLabel}')
    ax.set_ylabel(f'{yLabel}')
    ax.legend()
    ax.grid()
    fig.tight_layout()

    #Save plot
    if savePlot:
        filename = f'{yVal}_vs_{xVal}.png'
        fig.savefig(os.path.join('./Plots', filename))
        
    plt.show()
    return


def getDiffusionData(gasComp):
    """
    TODO
    """

    gasCompOptions = [
            'ArCO2-80-20',
            'T2K'
        ]

    if gasComp not in gasCompOptions:
        raise ValueError(f"Error: Invalid gas composition '{gasComp}'.")
    
    gasFilenames = f'diffusion.{gasComp}*.dat'
    dataPath = os.path.join('../Data/Diffusion', gasFilenames)
    fileList = glob.glob(dataPath)

    if not fileList:
        print(f"No files found matching pattern '{dataPath}'.")
        return None

    dataMagboltz = []

    for inFile in fileList:
        inData = {"filename": os.path.basename(inFile)}
        
        try:
            with open(inFile, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
            
            if len(lines) < 11:
                print(f"Error: File {inFile} has unexpected format - Skipping.")
                continue

            inData['gasComposition'] = lines[3]
            inData['eField'] = float(lines[5])
            inData['driftVelocity'] = float(lines[7])
            inData['driftVelocityErr'] = float(lines[8])
            inData['diffusionLongitudinal'] = float(lines[10])
            inData['diffusionLongitudinalErr'] = float(lines[11])
            inData['diffusionTransverse'] = float(lines[13])
            inData['diffusionTransverseErr'] = float(lines[14])

            dataMagboltz.append(inData)
            
        except IndexError:
            print(f"Parsing Error: Could not find data at expected line index in {inFile}.")
        except ValueError:
            print(f"Parsing Error: Could not convert value to float in {inFile}.")
        except IOError as e:
            print(f"Error opening or reading file {inFile}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {inFile}: {e}")

    rawData = pd.DataFrame(dataMagboltz)

    sortedData = rawData.sort_values(by='eField').reset_index(drop=True)

    return sortedData
    
