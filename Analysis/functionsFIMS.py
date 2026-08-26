import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
import pandas as pd
import json

from scipy.special import gammaincc
from scipy.stats import beta
from scipy.interpolate import griddata

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
    getDiffusionData
    getFullFieldData
    plotFullField
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
def getSetData(runList, xVal, yVal):#TODO account for calcData vs runData here
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
        yData.append(simData.getCalcParameter(yVal))

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

    if savePlot and not os.path.exists('./Plots'):
        os.makedirs('./Plots')

    #Add units to axis labels if dimensional
    dimensionalParam = [
        'Pad Length',
        'Pitch',
        'Amplification Gap',
        'Grid Thickness',
        'Hole Radius',
        'Drift Length',
        'Thickness SiO2',
        'Field Bundle Radius'
    ]
    xLabel = xVal + r' ($\mu$m)' if xVal in dimensionalParam else xVal
    yLabel = yVal + r' ($\mu$m)' if yVal in dimensionalParam else yVal

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

#********************************************************************************#
def plot2DGasScan(allData, plotParams):
    """
    Generates a series of 2D color mesh plots for various gas mixture parameters.

    Args:
        allData (pd.DataFrame): DataFrame containing simulation results
        plotParams (dict): Dictionary where keys are column names in allData
                           and values are tuples of (units, scale).
                           - units (str): Units for the color bar.
                           - scale (str): Scale for color bar ('log', 'logit'), 
                                            otherwise is linear.
    """

    numPlots = len(plotParams)
    numCols = 2 if numPlots > 1 else 1
    numRows = int(np.ceil(numPlots/2))

    fig, axes = plt.subplots(
        numRows, numCols, 
        figsize=(12, 5*numRows), 
        constrained_layout=True, squeeze=False
    )

    fig.suptitle(f'Simulation Results for Ar/CF4/Isobutane Mixtures', fontweight='bold', fontsize=16)

    axesFlat = axes.flatten()

    for ax, (inData, (units, scale)) in zip(axesFlat, plotParams.items()):

        # Pivot the data for pcolormesh
        plotData = allData.pivot(index='Gas Comp: CF4', columns='Gas Comp: Isobutane', values=inData)
        
        plotScale = 'linear'
        if scale == 'log' or scale == 'logit':
            plotScale = scale
        
        dataMesh = ax.pcolormesh(
            plotData.columns, 
            plotData.index, 
            plotData.values, 
            shading='nearest', 
            cmap='viridis',
            norm=plotScale
        )

        ax.scatter(2, 3, marker='$T2K$', color='r', s=1000, label='T2K Gas')
        ax.scatter(0, 0, marker='$Pure~Ar$', color='r', s=1000, label='Pure Ar')

        ax.set_title(f'{inData.strip()}', fontweight='bold')
        ax.set_xlabel(f'Isobutane Concentration (%)')
        ax.set_ylabel(f'CF4 Concentration (%)')

        fig.colorbar(dataMesh, ax=ax, label=f'{inData.strip()} ({units})')

    # Hide any unused subplots
    for i in range(numPlots, len(axesFlat)):
        axesFlat[i].axis('off')

    return fig

#********************************************************************************#
def getGasData(runNoList):
    """
    Retrieves and compiles metadata and calculated data for a list of runs
    into a single DataFrame.

    Args:
        runNoList (list): List of run numbers to retrieve data for.
    Returns:
        pd.DataFrame: DataFrame containing combined metadata and calculated data
                      for all specified runs, sorted by run number.
    """

    from runDataClass import runData
    
    allRunData = []

    # Get the data from each run
    for inRun in runNoList:
        simData = runData(inRun)

        metaData = simData.getMetaData()
        calcData = simData.getCalculatedData()

        inRunData = {'runNo': inRun} | metaData | calcData

        allRunData.append(inRunData)

    #Combine all reults into a single dataframe, sorted by run number
    allRunData = pd.DataFrame(allRunData)
    allRunData = allRunData.sort_values(by='runNo').reset_index(drop=True)

    return allRunData
 
#********************************************************************************#
def getFullFieldData(runNumber):
    """
    Gets the field data points for each drift line created by runFullField.

    Args:
        runNumber (int): Run number of the dataset
        
    returns:
        fieldData (dict): dictionary of numpy arrays for each spacial coordinate.
    """

    filePath = f'../Data/sim{runNumber}fullFieldLines.dat'

    xData, yData, zData = np.loadtxt(filePath, delimiter=',', unpack=True)

    CMTOMICRON = 1e4
    fieldData = {
        'xComp': xData*CMTOMICRON,
        'yComp': yData*CMTOMICRON,
        'zComp': zData*CMTOMICRON
    }
    
    return fieldData

#********************************************************************************#
def plotFullField(runNum, zTarget=0):
    """
    Plots a 2D slide of the full electric field.
    
    args:
        runNumber (int): Run number of the dataset
        zTarget (float): height of 2D field slice
    
    returns:
        figure
    """
    #Get all field line data
    fieldData = getFullFieldData(runNum)
    xData = fieldData['xComp']
    yData = fieldData['yComp']
    zData = fieldData['zComp']

    #Find data within slice
    ##Note - this selects ALL data within a slice.
    #Todo - if looking to plot this as density plot, ensure exactly 1 datapoint per field line.
    sliceWidth = 0.5
    sliceRegion = (zData > (zTarget - sliceWidth)) & (zData < (zTarget + sliceWidth))

    xSlice = xData[sliceRegion]
    ySlice = yData[sliceRegion]
    
    # Plot the x,y components of the field at the given height
    fieldFig = plt.figure(figsize=(14,6))
    xySlice = fieldFig.add_subplot(121)
    xzSlice = fieldFig.add_subplot(122)

    xySlice.scatter(xSlice, ySlice, s=.1, c='r')
    xySlice.grid()
    xySlice.set_xlabel(r'x Position ($\mu$m)')
    xySlice.set_ylabel(r'y Position ($\mu$m)')
    
    # Plot the x,z components of the field along with a line showing the target height
    xzSlice.scatter(xData, zData, s=.1, c='r')
    xzSlice.axhline(zTarget, c='y', lw=3, label='Target Height')
    xzSlice.grid()
    xzSlice.legend(loc='lower left')
    xzSlice.set_xlabel(r'x Position ($\mu$m)')
    xzSlice.set_ylabel(r'z Position ($\mu$m)')
    
    fieldFig.suptitle('2D Field Slice', fontsize = 20)
    
    return fieldFig

#********************************************************************************#
def plotFullFieldMapping(runNum):
    """
    Plots a map of the initial radius of a field line vs its final radius.
    
    args:
        runNumber (int): Run number of the dataset
    
    returns:
        figure
    """
    # Get all field line data
    fieldData = getFullFieldData(runNum)
    xData = np.asarray(fieldData['xComp'])
    yData = np.asarray(fieldData['yComp'])
    zData = np.asarray(fieldData['zComp'])

    #Calaculate all radii
    radii = np.hypot(xData, yData)

    #Identify jumps between lines
    lineID = np.where(np.diff(zData) <= 0)[0]
    
    #Initial indices - Beginning and each point after a jump
    initialID = np.insert(lineID+1, 0, 0)

    #Final indices - First jump point and last
    finalID = np.append(lineID, len(zData)-1)

    #Make plot
    mapFig = plt.figure()
    plt.scatter(
        radii[initialID], radii[finalID], 
        c='b', s=.4, label='Data'
    )
    plt.xlabel(r'Initial Radius ($\mu$m)')
    plt.ylabel(r'Final Radius ($\mu$m)')
    plt.title('Field Line Mapping')
    #plt.legend()
    plt.grid()
    
    return mapFig

#********************************************************************************#
def getAsymErrs(numSuccess, numTotal):
    """
    TODO
    """

    if numTotal <= 0:
        return 0.0, 0.0

    # Beta posterior (Laplace prior)
    a = numSuccess + 1
    b = (numTotal - numSuccess) + 1

    meanEff = a / (a+b)

    # 1-sigma bounds
    pLow = (1 - .6827) / 2
    pHigh = 1 - pLow

    lowerBound = beta.ppf(pLow, a, b)
    upperBound = beta.ppf(pHigh, a, b)

    #Errors on mean
    errorLow = meanEff - lowerBound
    errorHigh = upperBound - meanEff

    return errorLow, errorHigh

#********************************************************************************#
def plotAllEfficiencies():
    """
    Plot the efficiencies (net, detect, collect) as a fucntion of field ratio.
    """

    runData = pd.read_csv('../Data/allEfficiencyResults.csv')
    runData.sort_values(by='fieldRatio', inplace=True)

    fig = plt.figure(figsize=(8, 5))

    plt.errorbar(
        runData['fieldRatio'], runData['collectionEff'], 
        yerr=[runData['collectionErrLow'], runData['collectionErrHigh']],
        c='r', ls='-', marker='x',
        label='Collection Efficiency'
    )

    plt.errorbar(
        runData['fieldRatio'], runData['detectionEff'], 
        yerr=[runData['detectionErrLow'], runData['detectionErrHigh']],
        c='b', ls='-', marker='x',
        label='Detection Efficiency'
    )

    plt.errorbar(
        runData['fieldRatio'], runData['netEff'], 
        yerr=[runData['netErrLow'], runData['netErrHigh']],
        c='g', ls='-', marker='x',
        label='Net Efficiency'
    )

    plt.axhline(.95, c='m', ls=':', label='Target Efficiency = 95%')
    plt.title('Minimum Field Finder')
    plt.xlabel('Field Ratio')
    plt.ylabel('Efficiency (%)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return fig

#********************************************************************************#
def getOT(hole, pitch):
    holeArea = math.pi*hole**2
    inRadius = pitch/2
    hexArea = 2*math.sqrt(3)*inRadius**2

    return holeArea / hexArea

#********************************************************************************#
def readScanData(filename):
    """"""

    rawLines = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                rawLines.append(json.loads(line))

    flatData = []
    for run in rawLines:       
        params = run.get('params', {})
        results = run.get('results', [])
        
        for result in results:
            simResults = result.get('simResults', {})
            
            flatData.append({
                'pitch': params.get('pitch', -1),
                'holeRadius': params.get('holeRadius', -1),
                'standoff': params.get('gridStandoff', 50),
                'fieldRatio': result.get('fieldRatio', -1),
                'runNumber': result.get('runNumber', -1),
                'meanGain': simResults.get('averageGain', 0),
                'netEfficiency': simResults.get('netEff', 0),
                'colEfficiency': simResults.get('collectionEff', 0),
                'detEfficiency': simResults.get('detectionEff', 0),
                'gainHist': simResults.get('gainHist', []),
                'rawGains': simResults.get('rawGains', [])
            })

    scanData = pd.DataFrame(flatData)

    return scanData

#********************************************************************************#
def getPolyaData(scanData):
    """
    Get the Polya information from scanned runs,
    """
    from runDataClass import runData

    allSimData = []
    for _, inRow in scanData.iterrows():
        inRun = int(inRow['runNumber'])
        inField = inRow['fieldRatio']
        inColEff = inRow['colEfficiency']
    
        simData = runData(inRun)
        inGain = simData.getCalcParameter('Trimmed Gain')
        inTheta = simData.getCalcParameter('Polya Theta')
        inThetaErr = simData.getCalcParameter('Polya Theta Error')
        inPGain = simData.getCalcParameter('Polya Gain')
        inPGainErr = simData.getCalcParameter('Polya Gain Error')
    
        if not math.isnan(inGain):
            inGain = int(inGain)
        else:
            inGain = 10
    
        collectEff = float(inColEff)

        for inThresh in range(1, 101):
            inDetectEff = simData._getEfficiency(threshold=inThresh)
            detectEff = inDetectEff['efficiency']
            
            if detectEff < 0.01:
                detectEff = 0.0
                
            netEfficiency = collectEff*detectEff
    
            allSimData.append({ 
                'runNumber': inRun,
                'fieldRatio': inField,
                'gain': inGain,
                'threshold': inThresh,
                'netEfficiency': netEfficiency,
                'detectEff': detectEff,
                'collectEff': collectEff,
                'theta': inTheta,
                'thetaErr': inThetaErr,
                'pGain': inPGain,
                'pGainErr': inPGainErr
            })
    
    allSimData = pd.DataFrame(allSimData)

    return allSimData
    
#********************************************************************************#
def getDataToPlot(dataFile):
    scanData = readScanData(dataFile)
    allData = getPolyaData(scanData)

    return allData

#********************************************************************************#
def plotPolyaData(datasets, absField=False, vsGain=False):
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for name, data in datasets.items():
        # Determine field scale factor
        fieldScale = 1.0 if (not absField or name == 'ArCO2') else 0.280

        # Extract and process core dataset
        plotData = (
            data[['fieldRatio', 'pGain', 'pGainErr', 'theta', 'thetaErr']]
            .drop_duplicates()
            .sort_values('fieldRatio')
        )
        
        # Extract core dataset and minimum GTR at >= 95% efficiency
        plotData = data[['fieldRatio', 'pGain', 'pGainErr', 'theta', 'thetaErr']].drop_duplicates().sort_values('fieldRatio')
        aboveEff = data[data['detectEff'] >= 0.95].copy()
        aboveEff['GTR'] = aboveEff['gain'] / aboveEff['threshold']
        minGTR = aboveEff.loc[aboveEff.groupby('fieldRatio')['GTR'].idxmin()]

        # Define raw physical quantities
        field = (plotData['fieldRatio'] * fieldScale, None)
        gain = (plotData['pGain'], plotData['pGainErr'])
        theta = (plotData['theta'], plotData['thetaErr'])
        gtr = (minGTR['GTR'], None)
        gtrField = (minGTR['fieldRatio'] * fieldScale, None)
        gtrGain = (minGTR['gain'], None)

        # Configure and make plots
        if vsGain:
            panels = [(gain, field), (gain, theta), (gtrGain, gtr)]
        else:
            panels = [(field, gain), (field, theta), (gtrField, gtr)]

        for inAx, ((x, xerr), (y, yerr)) in zip(ax, panels):
            inAx.errorbar(
                x, y, 
                xerr=xerr, yerr=yerr, 
                marker='x', ls='-', label=name
            )

    # Configure Labels and Formatting
    fieldLabel = r'Amplification Field: $E_{\text{Amp}}$ (kV/cm)' if absField else r'Field Ratio: $E_{\text{Amp}}~/~E_{\text{Drift}}$'
    gainLabel = r'Gas Gain: $\bar{n}$'
    
    xLabel = gainLabel if vsGain else fieldLabel
    yLabels = [
        fieldLabel if vsGain else gainLabel,
        r'Polya Shape: $\theta$',
        r'GTR for $\epsilon_{\text{d}}$=95%'
    ]

    # Set log scale and Y limits for gain and field plot
    if vsGain:
        ax[0].set_xscale('log')
        ax[0].set_xlim(1, None)
        ax[0].set_ylim(10, None)
    else:
        ax[0].set_yscale('log')
        ax[0].set_xlim(10, None)
        ax[0].set_ylim(1, None)

    # Apply remaining panel formatting
    for inAx, label in zip(ax, yLabels):
        inAx.grid()
        inAx.set_ylabel(label, fontsize=14)
        if inAx != ax[0]:
            inAx.set_ylim(0, None)

    ax[2].set_xlabel(xLabel, fontsize=14)
    ax[0].legend(fontsize=14, loc='upper left')

    plt.tight_layout()
    plt.show()
    
#********************************************************************************#
def plotEfficiencyContours(allData, xLabel):
    """
    Plot the efficiency data across 2D scans wiht contours indicated.
    """
    
    fontsize = 14

    x = np.array(allData['xData'])
    y = np.array(allData['yData'])
    z = np.array(allData['zData'])

    fig = plt.figure(figsize=(10, 6))

    # Plot the data
    contour = plt.tricontourf(
        x, y, z,
        levels=np.linspace(0, 1, 101),
        cmap="viridis",
    )
    cbar = plt.colorbar(contour)
    cbar.set_ticks(np.linspace(0, 1, 11))
    cbar.set_label('Net Efficiency', rotation=270, labelpad=15, fontsize=fontsize)

    # Plot the contour lines
    effLines = [.95, .90, .75, .50]
    effLineStyle = ['-', '--', '-.', ':']
    for inLevel, inLine in zip(effLines, effLineStyle):
        contourLine = plt.tricontour(
            x, y, z, 
            levels=[inLevel], 
            colors='m', 
            linestyles=inLine,
            linewidths=2.5
        )
        plt.clabel(contourLine, inline=True, fontsize=fontsize, fmt=f"{inLevel*100:.0f} %%")
        plt.plot([], [], c='m', ls=inLine, lw=2.5, label=r"$\epsilon_{\text{Net}}$"+f" = {inLevel*100:.0f}%")

    # Plot breakdown region
    xBreakdown = allData['xBreakdown']
    yBreakdown = allData['yBreakdown']
    plt.fill_between(
        xBreakdown, yBreakdown, y.max()*np.ones(len(yBreakdown)),
        color='r', alpha=0.4, hatch='//')
    plt.plot(
        xBreakdown, yBreakdown, 
        c='r', label='Breakdown Region', ls='-', lw=2.5
    )

    plt.xlabel(xLabel, fontsize=fontsize)
    plt.ylabel('Field Ratio', fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.yscale('log')

    plt.xlim([x.min(), x.max()])
    plt.ylim([y.min(), y.max()])

    plt.tight_layout()
    plt.show()

    return fig


#********************************************************************************#
def makePWL(runNumber, averageSignal=True, avalancheID=None):
    """
    Reads a run's Parquet signal file and exports an LTSpice-compatible PWL file.

    Default is to export the average signal. 
    If a single signal is desired, averageSignal must be False.
    If no specific avalanche ID is given, one is chosen at random.

    Args:
        runNumber (int): The simulation run number.
        averageSignal (bool): If true, exports the averae signal.
        avalancheID (int): Specific avalanche ID to export.
    """

    # Read in data from file
    filepath = '../Data/' # Ensure path to data (output will also be written here)
    filename = f'allSignalsRun{runNumber}.parquet'
    dataFile = os.path.join(filepath, filename)

    allData = pd.read_parquet(dataFile)

    # Time is common for all signals
    relativeTime = allData['commonTime'].values

    #Get average signal
    if averageSignal:
        rawSignal = allData['meanCurrent'].values
        signalLabel = 'AVERAGE'

    # Get individual signal
    else:
        individualColumns = [
            col for col in allData.columns if col.startswith('AvalancheID_')
        ]

        #Ensure data exists
        if not individualColumns:
            raise ValueError('No single signals in file')

        #Get chosen avalancheID or select randomly
        if avalancheID is not None:
            targetColumn = f'AvalancheID_{avalancheID}'
            if targetColumn not in individualColumns:
                raise ValueError('Invalid avalanche ID')
            chosenColumn = targetColumn                
        else:
            chosenColumn = np.random.choice(individualColumns)

        rawSignal = allData[chosenColumn].values
        signalLabel = chosenColumn

    # Format time and amplitude
    numPoints = len(relativeTime)
    timeStep = (relativeTime[-1] - relativeTime[0]) / (numPoints - 1)
    
    cleanTime = np.arange(numPoints) * timeStep * 1e-9  # ns -> s
    cleanSignal = rawSignal * 1e-6                     # uA -> A

    # Export PWl file
    pwlData = np.column_stack((cleanTime, cleanSignal))
    outFilename = f'signalFileRun{runNumber}.txt'
    outputFilename = os.path.join(filepath, outFilename)

    np.savetxt(outputFilename, pwlData, fmt='%.8e', delimiter=' ')
    print(f'Exported LTSpice PWL to: {outputFilename}')
    print(f'\tContains the {signalLabel} signal.')

    return

#********************************************************************************#
def plotEfficiencies(dataFull=None, dataScan=None, vsGain=False):
    '''TODO'''
    # TODO - Currently hardcoded for T2K and gridpix geometry

    if dataFull is not None:
        dataFull['netEff'] = dataFull['Charge Collection Eff'] * dataFull['Efficiency (10e)']
        dataFull['netEff (Low)'] = dataFull['Charge Collection Eff Err (Low)'] * dataFull['Efficiency Error (Low)']
        dataFull['netEff (High)'] = dataFull['Charge Collection Eff Err (High)'] * dataFull['Efficiency Error (High)']

    fig, ax = plt.subplots(figsize=(10, 6))

    if dataFull is not None:
        configsFull = [
            {'data': 'netEff', 'error': 'netEff ', 'label': 'Net', 'c': 'g', 'ls': '-'},
            {'data': 'Charge Collection Eff', 'error': 'Charge Collection Eff Err ', 'label': 'Collection', 'c': 'r', 'ls': '-'},
            {'data': 'Efficiency (10e)', 'error': 'Efficiency Error ', 'label': 'Detection', 'c': 'b', 'ls': '-'},
        ]
        xDataFull = dataFull['Trimmed Gain'] if vsGain else dataFull['ampField']/280
        for cfg in configsFull:
            inData = cfg['data']
            ax.errorbar(
                xDataFull, dataFull[inData],
                xerr=dataFull['Gain Error'] if vsGain else None,
                yerr=[dataFull[cfg['error']+'(Low)'], dataFull[cfg['error']+'(High)']],
                label=cfg['label']+' (Full)', c=cfg['c'], ls=cfg['ls']
            )

    if dataScan is not None:
        configsScan = [
            {'data': 'netEff', 'label': 'Net', 'c': 'g', 'ls': '--'},
            {'data': 'collectionEff', 'label': 'Collection', 'c': 'r', 'ls': '--'},
            {'data': 'detectionEff', 'label': 'Detection', 'c': 'b', 'ls': '--'},
        ]
        xDataScan = dataScan['averageGain'] if vsGain else dataScan['fieldRatio']
        for cfg in configsScan:
            inData = cfg['data']
            ax.errorbar(
                xDataScan, dataScan[inData],
                xerr=dataScan['averageGainErr'] if vsGain else None,
                yerr=dataScan[f'{inData}Err'],
                label=cfg['label']+' (Check)', c=cfg['c'], ls=cfg['ls']
            )
    
    optTrans = math.pi*(17.5/55)**2
    ax.axhline(optTrans, ls='-', c='c', label=f'OT ({optTrans:.3f})')
    ax.axhline(1-optTrans, ls='--', c='c', label='1 - OT')
    if vsGain:
        ax.axvline(10, ls='--', c='m', label='Threshold')

    xLabel = r'Gas Gain: $\overline{n}$' if vsGain else r'Field Ratio: $E_{\text{Amp}}~/~E_{\text{Drift}}$'
    ax.set_xlabel(xLabel, fontsize=14)
    ax.set_ylabel(r'Efficiency: $\epsilon$', fontsize=14)
    ax.set_xscale('log' if vsGain else 'linear')
    
    ax.grid()
    ax.legend(fontsize=14)
    
    plt.tight_layout()

    return fig
