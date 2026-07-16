#############################################
# CLASS DEFINITION FOR EVENT RECONSTRUCTION #
#############################################
import os
import sys
import math
import glob
import random
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class Reconstruction:
    """
    Class enabling particle event reconstruction approximations.
    
    Notes: All reconstruction plots assume the following
    
    -25 cm drift distance before amplification.
    
    -320 micron/sqrt(cm) and 200 micron/sqrt(cm) transverse and longitudinal 
    diffusion coefficients in the drift region, respectively.
    
    -190 micron/sqrt(cm) and 150 micron/sqrt(cm) transverse and longitudinal
    diffusion coefficients in the amplification region, respectively (GEM 
    based readouts do not use this value as they have functionally zero 
    amplification region length).
    
    -25 micron vertical spatial resolution (based on time resolution of readout converted
    into distance). This is different from the integration time of the readout, which is
    an input parameter.
    
    Functions:
        ## Base functions ##
        Private:
            _checkInput
            _getDataFrames
            _getCoordinates
            _groupData
            _convertToSignal
            _approximateToT
            _format3DPlot
        Public:
            diffuseData
            discretizeData
            approximateGain
            approximateReadout

        ## Wrapper functions ##
        plotRaw
        reconstructFIMS
        reconstructBEAST
        reconstructMigdal
        reconstructGridPix
    """
    #********************************************************************************#
    def __init__(self, reconInfo=None):
        """Initializes Reconstruction class."""
        # Validate Input
        self.reconInfo = reconInfo
        self._checkInput()
        
        # Get Data
        dataFrame = self._getDataFrames()
        self.rawData = self._getCoordinates(dataFrame)
        
        # Set constant values
        self.timeRez = 25 # microns
        self.initialDriftDistance = 10 # cm
        
        # Values from Tanner sim
        self.transDriftDifCoef = 320 # microns/sqrt(cm)
        self.lonDriftDifCoef = 200 # microns/sqrt(cm)
        
        self.transAmpDifCoef = 190 # microns/sqrt(cm)
        self.lonAmpDifCoef = 150 # microns/sqrt(cm)
        
        # Values from Majd paper
        #self.transDriftDifCoef = 136 # microns/sqrt(cm)
        #self.lonDriftDifCoef = 114 # microns/sqrt(cm)
        
        
        return
    
    #********************************************************************************#

    def _checkInput(self):
        """Checks input to ensure all keys are present."""
        paramKeys = [
            'Gain',
            'Avalanche Sigma',
            'Hole Pitch',
            'Pixel Pitch',
            'Standoff',
            'Signal Decay Rate',
            'Signal Threshold',
            'File Location',
            'Tree Name',
        ]
        
        if self.reconInfo is None:
            raise(KeyError('Reconstruction dictionary is empty.'))

        #Check that all parameters are present
        for key in paramKeys:
            if key not in self.reconInfo:
                raise KeyError(f"Key '{key}' is absent from reconstruction dictionary.")
            
        return
    
    #********************************************************************************#

    def _getDataFrames(self):
        """
        Unpacks a root file from a given filename
        
        returns:
            dataframes (pandas dataframe): unpacked root file
        """
        filePath = self.reconInfo['File Location']
        with uproot.open(filePath) as rootFile:
            dataframes = {}
            for treeKey in rootFile.keys():
                # Strip the tree number ID
                strippedName = treeKey.split(';')[0]
                
                if isinstance(rootFile[treeKey], uproot.behaviors.TTree.TTree):
                    tree = rootFile[treeKey]
                    try:
                        df = tree.arrays(library='pd')
                        dataframes[strippedName] = df
                    except Exception as e:
                        print(f"Error reading tree '{treeKey}': {e}")
        
        return dataframes

    #********************************************************************************#
    
    def _getCoordinates(self, dataframes):
        """
        Takes a given dataframe and extracts the x,y,z coordinates from a specified branch.
        
        Note: Assumes coordinates are given in cm and converts them to microns.
        
        Args:
            dataframes: pandas dataframe
        
        Returns:
            rawData (dataframe): the x,y,z coordinates of every electron
        """
        treeName = self.reconInfo['Tree Name']
        
        # Get data of a single trial
        trialData = dataframes[treeName][['x', 'y', 'z']].iloc[0]
        # convert to proper formating
        rawData = pd.DataFrame(
            zip(trialData['x']*10000, trialData['y']*10000, trialData['z']),
            columns=['x','y','z']
        )
        
        return rawData
    
    #********************************************************************************#

    def _groupData(self, inputData):
        """
        Takes the x,y,z coordinates and groups the amount of charge by location.
        
        Args:
            coordinates (dataframe): list of x,y,z coordinates of each electron.
        
        returns:
            groupedData (dataframe): z-coordinates with their corresponding 
            pixel location and amount of charge. 
        """
        countedData = inputData.groupby(['x', 'y', 'z']).size().reset_index(name='q')
        groupedData = countedData.groupby(['x','y']).agg(z=('z', list), q=('q',list)).reset_index()

        return groupedData

    #********************************************************************************#
        
    def _convertToSignal(self, pixel):
        """
        Takes data of a single pixel and calculates the voltage signal.
        
        args:
            pixel (dataframe): coordinates and charges of a single pixel.
        
        returns:
            signalData (dataframe): voltage signal data.
        """
        threshold = self.reconInfo['Signal Threshold']
        decayRate = self.reconInfo['Signal Decay Rate']
        
        # Generate a signal range that guarantees that the full signal is included
        rangeList = np.linspace(
            min(pixel['z']),
            max(pixel['z']) - math.log(threshold/sum(pixel['q']))*decayRate,
            1000
        )

        # Calculate the net signal of all charges
        netSignal = np.array([sum(pixel['q']*math.e**(-(height-pixel['z'])/decayRate)) for height in rangeList])

        # Identify which points are above and below threshold
        isAbove = netSignal >= threshold
        aboveRange = rangeList[isAbove]
        belowRange = rangeList[~isAbove]
        aboveID = np.flatnonzero(aboveRange)
        belowID = np.flatnonzero(belowRange)

        # Find the upwards crossing points
        upCrossMask = aboveID[1:]-aboveID[:-1] > 1
        upCrossMask = np.append(True, upCrossMask)
        upCrossPoints = aboveRange[upCrossMask]

        # Find all the downwards crossing points, using the last point if none are found
        if len(belowID > 0):
            downCrossMask = belowID[1:]-belowID[:-1] > 1
            downCrossMask = np.append(True, downCrossMask)
            downCrossPoints = belowRange[downCrossMask]
        else:
            downCrossPoints = [rangeList[-1]]

        ToTList = [end-start for start,end in list(zip(upCrossPoints, downCrossPoints))]
        
        return upCrossPoints, ToTList
    
    #********************************************************************************#

    def _approximateToT(self, signalPlot):
        """
        Approximates the TOT of a given group of electrons.
        
        args:
            signalPlot (dataframe): dataframe of signal points 
            
        returns:
            ToTDF (dataframe): dataframe of initial crossing times and ToTs
        """
        return
    
    #********************************************************************************#

    def diffuseData(self, coordinates, diffusionWidths):
        """
        Applies a Gaussian smear to a given data set
        
        Args: 
            coordinates (dataframe): x,y,z coordinates of each electron prior to
            diffusion.
            diffusionWidths (tuple): standard deviation of the Gaussian smear for each
            coordinate.
        
        Returns:
            diffusedData (list): list of all data points after being diffused
        """
        size = len(coordinates['x'])
        diffusionAmount = pd.DataFrame({
            'x': np.random.normal(0, diffusionWidths[0], size),
            'y': np.random.normal(0, diffusionWidths[1], size),
            'z': np.random.normal(0, diffusionWidths[2], size)
        })
        
        diffusedData = coordinates.add(diffusionAmount, fill_value=0)

        return diffusedData
        
    #********************************************************************************#

    def discretizeData(self, inputArray, binSize):
        """
        Bins a given dataset
        
        Note: assumes units are given in microns
        
        Args:
            inputArray (dataframe): list of data points to be binned.
            binSize (dict): widths bins for each coordinate axis.
        
        returns:
            discreteData (list): list of discretized coordinates.
        """
        # Get bounds
        boundMin = round(np.min(inputArray)/10)*10 - 100
        boundMax = round(np.max(inputArray)/10)*10 + 100

        discreteDataFrame = pd.DataFrame()
        for column in inputArray:
            # Check if data has a bin size
            if binSize[column] == 0 or binSize[column] == None:
                discreteData = inputArray[column]
            else:
                binEdges = np.arange(boundMin, boundMax, binSize[column])
                binnedData = pd.cut(inputArray[column], binEdges, labels=binEdges[:-1])
                discreteData = binnedData.astype(int) + int(binSize[column]/2)
            discreteDataFrame[column] = discreteData

        return discreteDataFrame

    #********************************************************************************#
    
    def avalancheData(self, coord, difWidths):
        """
        Takes the x,y,z coordinates of an electron dataframe and approximates an avalanche.
        
        Note: also applies diffusion to the new electrons
        
        args:
            coord (dataframe): the x,y,z coordinates of each initial electron
            difWidths (tuple, floats): diffusion values for each axis
        
        returns:
            avalData (dataframe): list of x,y,z coordinates for each new electron
        """
        sigma = self.reconInfo['Avalanche Sigma']
        gain = self.reconInfo['Gain']
    
        newElec = coord.apply(lambda x: np.random.normal(
            loc=(x['x'], x['y'], x['z']), scale=difWidths, 
            size=(abs(int(random.gauss(gain, sigma)))+1, 3)
            ), axis=1
        )
        avalData = pd.concat([pd.DataFrame(elem, columns=['x','y','z']) for elem in newElec], ignore_index=True)

        return avalData
    
    #********************************************************************************#
    
    def approximateGain(self, discreteDataFrame):
        """
        Takes the x,y,z coordinates of an electron dataframe and approximates an avalanche.
        
        Note: does not apply any diffusion to the new electrons.
        
        args:
            coordinates (dataframe): the x,y,z coordinates of each initial electron
        
        returns:
            avalData (dataframe): list of x,y,z coordinates for each new electron
        """
        gain = self.reconInfo['Gain']
        sigma = self.reconInfo['Avalanche Sigma']
        # convert to np.array for easier manipulation
        preAvalancheElectrons = np.array(list(zip(
            discreteDataFrame['x'],
            discreteDataFrame['y'],
            discreteDataFrame['z']
        )))
        
        postAvalancheElectrons = np.empty((1,3))
        for elem in preAvalancheElectrons:
            newElectrons = [elem]*int(random.gauss(gain, sigma))
            # Check if random.gauss was positive and ensure initial electron survives
            if len(newElectrons):
                postAvalancheElectrons = np.concatenate((postAvalancheElectrons, newElectrons))
            else:
                postAvalancheElectrons = np.concatenate((postAvalancheElectrons, [elem]))
        
        # Convert back to dataframe and remove first index (blank index from np.empty)
        avalData = pd.DataFrame(postAvalancheElectrons, columns=['x','y','z'])
        avalData = avalData.drop(avalData.index[0])
        
        return avalData

    #********************************************************************************#

    def approximateReadout(self, inputData):
        """
        Takes a charge distribution and approximates the readout values for each pixel.

        Args:
            inputData (dataframe): x,y,z coordinates of each charge.
        returns:
            readoutData (dataframe): x,y,z coordinates of the charge bundles as well as the time over threshold.
        """
        threshold = self.reconInfo['Signal Threshold']

        # Group data by pixel
        groupedData = self._groupData(inputData)

        # Filter pixels with total charge less than the threshold
        mask = groupedData['q'].apply(lambda q: sum(q) > threshold and sum(q) > len(q))
        filteredData = groupedData[mask].reset_index(drop=True)
        
        print('Calculating ToT...')
        # Convert charge to voltage signal
        signalData = filteredData.apply(self._convertToSignal, axis=1)
        crossing, ToT = list(zip(*signalData))
        filteredData['ToT'] = ToT
        filteredData['crossing'] = crossing

        # Remove depreciated columns
        filteredData.drop(['z', 'q'], axis=1, inplace=True)

        # Expand data for easier use
        readoutData = filteredData.explode('ToT', ignore_index=True)
        readoutData = readoutData.explode('crossing', ignore_index=True)
 
        return readoutData
    
    #********************************************************************************#
    
    def _format3DPlot(self, plotData, title='', charge=False):
        """
        Creates a 3D and 2D plot of a given dataset.
        
        args:
            plotData (pd.array): pandas array of data.
            title (str): Name of the data set
            charge (bool): boolean indicating if charge density is used as a color
            map.
        returns:
            fig3D (figure): matplotlib figure
        """
        
        # Create figures
        fig3D = plt.figure(figsize=(10, 5), dpi=200)
        sub3D = fig3D.add_subplot(121, projection='3d')
        sub2D = fig3D.add_subplot(122)
        
        # Assign point color, if given
        if charge:
            color = plotData['q']
        else:
            color = 'g'
        
        # Plot data in 2D and 3D
        sub3DRef = sub3D.scatter(
            plotData['x'], plotData['y'], plotData['z'],
            s=.2, c=color, label=f'{title} Readout Data', cmap='viridis'
        )
        
        sub2DRef = sub2D.scatter(
            plotData['x'], plotData['y'],
            s=.3, c=color, label=f'{title} Readout Data', cmap='viridis'
        )
        
        # Add color bar
        if charge:
            colorBar = plt.colorbar(sub2DRef, pad=.2)
            colorBar.set_label('Charge')

        # Add labels and adjust formatting
        sub3D.set_xlabel('x pixels')
        sub3D.set_ylabel('y pixels')
        sub3D.set_zlabel('Height')
        sub3D.set_title(f'{title} 3D Event Reconstruction')
        
        sub2D.set_xlabel('x pixels')
        sub2D.set_ylabel('y pixels')
        sub2D.set_title(f'{title} 2D Event Reconstruction')
        sub2D.yaxis.set_label_position("right")
        sub2D.yaxis.tick_right()
        sub2D.grid(True, alpha=.5)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        
        return fig3D

    #********************************************************************************#
    ############## Reconstruction Wrapper Functions for Specific Setups ##############
    #********************************************************************************#
    
    def plotRaw(self):
        """
        Plots the raw data from an event.
        
        returns:
            rawFig: matplotlib figure
        """
        rawFig = self._format3DPlot(self.rawData, title='Raw Data')
        
        return rawFig
        
    #********************************************************************************#
    
    def reconstructFIMS(self):
        """
        Approximates an event reconstruction using a FIMS readout.
        
        Amplification produced by a thin aluminum mesh that induces amplification 
        below it. Uses a pixel-pad readout with instant reset time, enabling clear 
        distinction of electrons in the vertical direction.
        
        returns:
            FIMSfig: matplotlib figure
        """
        # Extract relevant data from dictionary and set constant values
        holePitch = self.reconInfo['Hole Pitch']
        pixPitch = self.reconInfo['Pixel Pitch']
        timeRez = self.timeRez
        
        transDif = self.transDriftDifCoef*math.sqrt(self.initialDriftDistance)
        lonDif = self.lonDriftDifCoef*math.sqrt(self.initialDriftDistance)
        firstDifWidths = (transDif, transDif, lonDif)
        
        # Apply Gaussian smear to approximate diffusion
        smearData = self.diffuseData(self.rawData, firstDifWidths)

        # Discretize data to approximate falling into grid holes
        bins = {'x': holePitch, 'y': holePitch, 'z': 0}
        discreteData = self.discretizeData(smearData, bins)
        
        # Approximate avalanches
        # Diffusion is smaller than the pitch between pixels, so there
        # is zero net diffusion in the amplification region.
        numBelowThresh = int(len(discreteData)*0.05)
        belowID = np.random.choice(discreteData.index, size=numBelowThresh, replace=False)
        avalData = discreteData.drop(belowID).reset_index(drop=True)
        
        # Plot data
        FIMSfig = self._format3DPlot(avalData, title='FIMS')
        
        return FIMSfig
        
    #********************************************************************************#
    
    def reconstructBEAST(self):
        """
        Approximates an event reconstruction using a BEAST readout.
        
        Uses GEM amplification structure. Post avalanche electrons are then drifted 
        again before reaching a pixel readout. Readout uses infinite integration time,
        so final readout is purely 2D (x,y).
        
        returns:
            beastFig: matplotlib figure
        """
        # Extract relevant data from dictionary
        holePitch = self.reconInfo['Hole Pitch']
        pixPitch = self.reconInfo['Pixel Pitch']
        standoff = self.reconInfo['Standoff']
        timeRez = self.timeRez
        
        transDif = self.transDriftDifCoef*math.sqrt(self.initialDriftDistance)
        lonDif = self.lonDriftDifCoef*math.sqrt(self.initialDriftDistance)
        firstDifWidths = (transDif, transDif, lonDif)
        
        secondTransDif = self.transDriftDifCoef*math.sqrt(standoff/10000.) # Convert to cm
        secondLonDif = self.lonDriftDifCoef*math.sqrt(standoff/10000.)
        secondDifWidths = (secondTransDif, secondTransDif, secondLonDif)
        
        # Convert net avalanche stats to stats of an individual GEM (double GEM stack)
        self.reconInfo['Gain'] = int(math.sqrt(self.reconInfo['Gain']))
        self.reconInfo['Avalanche Sigma'] = int(math.sqrt(self.reconInfo['Avalanche Sigma']))
        
        # Apply Gaussian smear to approximate diffusion
        smearData = self.diffuseData(self.rawData, firstDifWidths)

        # Discretize data to approximate falling into first GEM holes
        holeBins = {'x': holePitch, 'y': holePitch, 'z': 0}
        discreteData1 = self.discretizeData(smearData, holeBins)
        
        # Approximate first set of avalanches
        avalData1 = self.avalancheData(discreteData1, secondDifWidths)
        
        # Discretize data to approximate falling into second GEM holes
        discreteData2 = self.discretizeData(avalData1, holeBins)
        
        # Approximate second set of avalanches
        avalData2 = self.avalancheData(discreteData2, secondDifWidths)
        
        # Discretize data to approximate pixels readout
        pixBins = {'x': pixPitch, 'y': pixPitch, 'z': timeRez}
        readoutData = self.discretizeData(avalData2, pixBins)
        
        # Configure data for plotting
        plotData = inputData.groupby(['x', 'y', 'z']).size().reset_index(name='q')
        
        # Plot data
        beastFig = self._format3DPlot(plotData, title='BEAST', charge=True)
        
        return beastFig

    #********************************************************************************#
    
    def reconstructMigdal(self):
        """
        Approximates an event reconstruction using the Migdal experiment readout.
        
        Uses a THGEM-esque amplification structure. Post avalanche electrons are then
        drifted again before reaching a pixel readout. Readout uses long integration
        time, so final readout is functionally 2D (x,y).
        
        returns:
            migdalfig: matplotlib figure
        """
        # Extract and calculate relevant data
        holePitch = self.reconInfo['Hole Pitch']
        pixPitch = self.reconInfo['Pixel Pitch']
        standoff = self.reconInfo['Standoff']
        timeRez = self.timeRez
        
        transDif = self.transDriftDifCoef*math.sqrt(self.initialDriftDistance)
        lonDif = self.lonDriftDifCoef*math.sqrt(self.initialDriftDistance)
        firstDifWidths = (transDif, transDif, lonDif)
        
        secondTransDif = self.transDriftDifCoef*math.sqrt(standoff/10000.) # Convert to cm
        secondLonDif = self.lonDriftDifCoef*math.sqrt(standoff/10000.)
        secondDifWidths = (secondTransDif, secondTransDif, secondLonDif)
        
        # Apply Gaussian smear to approximate initial drift diffusion
        smearData = self.diffuseData(self.rawData, firstDifWidths)

        # Discretize data to approximate falling into grid holes
        holeBins = {'x': holePitch, 'y': holePitch, 'z': 0}
        discreteData = self.discretizeData(smearData, holeBins)
        
        # Approximate avalanches
        avalData = self.avalancheData(discreteData, secondDifWidths)
        
        # Discretize data to approximate pixels readout
        pixBins = {'x': pixPitch, 'y': pixPitch, 'z': timeRez}
        padData = self.discretizeData(avalData, pixBins)

        # Approximate Signal Readout
        #readoutData = self.approximateReadout(padData)
        readoutData = padData.copy()
        
        # Plot data
        # Extract Data
        totalXWidth = max(readoutData['x']) - min(readoutData['x'])
        totalYWidth = max(readoutData['y']) - min(readoutData['y'])
        numXBins = int(totalXWidth/pixPitch)
        numYBins = int(totalYWidth/pixPitch)
        
        # Create figure
        migdalFig = plt.figure()
        plt.hist2d(
            readoutData['x'],
            readoutData['y'],
            bins=(numXBins, numYBins)
        )
        
        # Add plot elements
        plt.colorbar().set_label('Charge')
        plt.xlabel('x pixels')
        plt.ylabel('y pixels')
        plt.title('Migdal Experiment Event Reconstruction')
        
        return migdalFig

    #********************************************************************************#
    
    def reconstructGridPix(self):
        """
        Approximates an event reconstruction using the GridPix readout.
        
        Drifted electrons are amplified by a single, thin, aluminum mesh. The full
        amplification occurs below the mesh. Avalanched electrons are read out by a
        pixel readout. The pixel ID gives the x,y position, the threshold crossing
        time gives z, and the time over threshold gives the total charge. This allows
        for a full 3D reconstruction.
        
        returns:
            gridpixFig: matplotlib figure
        """
        # Extract and calculate relevant data
        holePitch = self.reconInfo['Hole Pitch']
        pixPitch = self.reconInfo['Pixel Pitch']
        standoff = self.reconInfo['Standoff']
        timeRez = self.timeRez
        
        transDif = self.transDriftDifCoef*math.sqrt(self.initialDriftDistance)
        lonDif = self.lonDriftDifCoef*math.sqrt(self.initialDriftDistance)
        firstDifWidths = (transDif, transDif, lonDif)
        
        secondTransDif = self.transAmpDifCoef*math.sqrt(standoff/10000.) # Convert to cm
        secondLonDif = self.lonAmpDifCoef*math.sqrt(standoff/10000.)
        secondDifWidths = (secondTransDif, secondTransDif, secondLonDif)
        
        # Apply Gaussian smear to approximate initial drift diffusion
        smearData = self.diffuseData(self.rawData, firstDifWidths)

        # Discretize data to approximate falling into grid holes
        holeBins = {'x': holePitch, 'y': holePitch, 'z': 0}
        discreteData = self.discretizeData(smearData, holeBins)
        
        # Approximate avalanche
        avalData = self.avalancheData(discreteData, secondDifWidths)

        # Discretize data to approximate pixels readout
        pixBins = {'x': pixPitch, 'y': pixPitch, 'z': timeRez}
        padData = self.discretizeData(avalData, pixBins)

        # Approximate Signal Readout
        readoutData = self.approximateReadout(padData)
        
        # Format data for plotting
        plotData = readoutData.rename(columns={'crossing': 'z', 'ToT': 'q'})
        
        # Plot data
        gridPixFig = self._format3DPlot(plotData, title='GridPix', charge=True)
        
        return gridPixFig

    #********************************************************************************#

