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
            _format3DPlot
        
        Public:
            diffuseData
            discretizeData
            avalancheData
            approximateReadout

        ## Wrapper functions ##
        TODO: add z-distribution wrapper to quantify typical spacing between hits
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
        self.timeRez = 25 # ns
        self.driftVelocity = 1 # microns/ns
        self.zRez = self.timeRez*self.driftVelocity # microns
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
        
    def _convertToSignal(self, zLocs, charges):
        """
        Takes data of a single pixel and calculates ToT and threshold crossing time.
        
        args:
            pixel (dataframe): coordinates and charges of a single pixel.
        
        returns:
            upCrossPoints (list): list of threshold crossing times
            ToTList (list): list of ToT times
        """
        threshold = self.reconInfo['Signal Threshold']
        decayRate = self.reconInfo['Signal Decay Rate']

        z = np.asarray(zLocs)
        charge = np.asarray(charges)

        chargeSum = charge.sum()
        zMin= z.min()
        zMax = z.max() - np.log(threshold/chargeSum)*decayRate

        if zMin >= zMax:
            return [], []

        rangeList = np.linspace(zMin, zMax, 1000)

        #Matrix multiplication for signals
        dz = rangeList[:, np.newaxis] - z[np.newaxis, :]
        decay = np.where(dz >= 0, np.exp(-dz/decayRate), 0.0)
        netSignal = decay @ charge

        #Find crossing times
        isAbove = netSignal >= threshold
        diff = np.diff(isAbove.astype(np.int8))

        riseID = np.flatnonzero(diff == 1) + 1
        fallID = np.flatnonzero(diff == -1) + 1

        if isAbove[0]: #If signal is above threshold at t=0
            riseID = np.insert(riseID, 0, 0)

        if len(riseID) > len(fallID): #If signal is above threshold at tmax
            fallID = np.append(fallID, len(rangeList)-1)

        if len(riseID) == 0:
            return [], []

        # Calculate ToT
        upCrossPoints = rangeList[riseID].tolist()
        downCrossPoints = rangeList[fallID].tolist()
        ToTList = (np.array(downCrossPoints) - np.array(upCrossPoints)).tolist()
            
        return upCrossPoints, ToTList
    
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
        
        diffusionAmount = pd.DataFrame(
            np.random.normal(0, diffusionWidths, size=(size, 3)),
            columns=['x', 'y', 'z']
        )
        
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
        discreteDataFrame = pd.DataFrame(index=inputArray.index)

        for inColumn in inputArray.columns:
            size = binSize.get(inColumn)

            if not size:
                discreteDataFrame[inColumn] = inputArray[inColumn]
            else:
                #Floor divide to snap to lower bin edge. Add half to center in bin.
                binnedData = (inputArray[inColumn] // size) * size + (size/2)
                discreteDataFrame[inColumn] = binnedData.astype(int)

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
        # Get parameters
        sigma = self.reconInfo['Avalanche Sigma']
        gain = self.reconInfo['Gain']
        numInitial = len(coord)

        # Get gain for each initial electron from normal dist.
        allGains = np.random.normal(gain, sigma, size=numInitial)
        allGains = np.abs(allGains.astype(int)) + 1

        # Duplicate coordinates based on individual gains
        allElectrons = np.repeat(coord[['x', 'y', 'z']].to_numpy(), allGains, axis=0)
        numNewElectrons = len(allElectrons)

        # Get diffusion amounts and add to initial locations
        diffusion = np.random.normal(0, difWidths, size=(numNewElectrons, 3))
        avalData = pd.DataFrame(allElectrons + diffusion, columns=['x', 'y', 'z'])
        
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

        chargeSum = [sum(q) for q in groupedData['q']]
        chargeLen = [len(q) for q in groupedData['q']]
        chargeMask = [(s > threshold) and (s > l) for s, l in zip(chargeSum, chargeLen)]

        filteredData = groupedData[chargeMask].copy()
        if filteredData.empty:
            raise ValueError('Empty Dataframe.')
        
        # Calculate ToT by converting charge to voltage
        print('Calculating ToT...')
        signals = [
            self._convertToSignal(z, q)
            for z, q in zip(filteredData['z'], filteredData['q'])
        ]

        # Unpack results and remove depreciated columns
        crossings, tots = zip(*signals)
        filteredData['crossing'] = crossings
        filteredData['ToT'] = tots
        filteredData.drop(columns=['z', 'q'], inplace=True)

        readoutData = filteredData.explode(['crossing', 'ToT'], ignore_index=True)
        
        return readoutData
    
    #********************************************************************************#
    
    def _format3DPlot(self, plotData, title=''):
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
        try:
            color = plotData['q']
        except:
            color = 'g'
        
        # Plot data in 2D and 3D
        sub3DRef = sub3D.scatter(
            plotData['x'], plotData['y'], plotData['z'],
            s=.1, c=color, label=f'{title} Readout Data', cmap='viridis'
        )
        
        sub2DRef = sub2D.scatter(
            plotData['x'], plotData['y'],
            s=.3, c=color, label=f'{title} Readout Data', cmap='viridis'
        )
        
        # Add color bar    
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
    
    def _calcAverage(self, values):
        """
        Takes a list and calculates the average along with the error.
        
        args:
            values (list): list of values.
        
        returns:
            average (tuple): average value along with its uncertainty.
        """
        total = len(values)
        mean = sum(values)/total
        variance = sum([(elem - mean)**2 for elem in values]) / (total - 1)
        error = variance ** .5
        
        average = (mean, error)
        
        return average
    
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
    
    def getFIMSPileup(self, drift=10, reset=25, numTrials=100):
        """
        Determines the efficiency for a FIMS readout based on given input parameters.
        
        args:
            drift (float): initial drift distance of the electron in centimeters.
            reset (float): the time for the reset signal in nanoseconds.
            numTrials (int): number of data sets to sample.
            
        returns:
            efficiency (float): detection efficiency, measured as # initial/# counted.
        """
        # Extract relevant data from dictionary and set constant values
        holePitch = self.reconInfo['Hole Pitch']
        pixPitch = self.reconInfo['Pixel Pitch']
        zRez = reset*self.driftVelocity
        
        transDif = self.transDriftDifCoef*math.sqrt(drift)
        lonDif = self.lonDriftDifCoef*math.sqrt(drift)
        firstDifWidths = (transDif, transDif, lonDif)
        efficiencies = []
        
        trialNum = 0
        while trialNum < numTrials: 
            # Apply Gaussian smear to approximate diffusion
            smearData = self.diffuseData(self.rawData, firstDifWidths)
            totalElecNum = len(smearData['z'])
            
            # Discretize data to approximate falling into grid holes.
            bins = {'x': holePitch, 'y': holePitch, 'z': 0}
            discreteData = self.discretizeData(smearData, bins)
            
            # Convert the z position to arrival time
            minZ = abs(min(discreteData['z']))
            discreteData['t'] = (discreteData['z'] + minZ)/self.driftVelocity

            # Determine how many electrons are seen by the readout.
            groupedData = discreteData.groupby(['x','y']).agg(t=('t', list)).reset_index()

            check = [pixel for pixel in groupedData['t'] if len(pixel) > 1] #TODO: improve
            numDrop = 0
            for pixel in check:
                elecID = 1
                pixel.sort()
                while elecID < len(pixel):
                    if abs(pixel[elecID] - pixel[elecID-1]) < reset:
                        numDrop += 1
                    elecID += 1
            
            singleEff = (totalElecNum - numDrop)/totalElecNum
            efficiencies.append(singleEff)

            trialNum += 1
        
        efficiency = self._calcAverage(efficiencies)
        
        return efficiency
    
    #********************************************************************************#
    
    def getGridPixPileup(self, drift=10, reset=25, numTrials=100):
        """
        Determines the efficiency for a GridPix readout based on given input parameters.
        
        args:
            drift (float): initial drift distance of the electron in centimeters.
            reset (float): the time for the reset signal in nanoseconds.
            numTrials (int): number of data sets to sample.
            
        returns:
            efficiency (float): detection efficiency, measured as # initial/# counted.
        """
        # Extract and calculate relevant data
        holePitch = self.reconInfo['Hole Pitch']
        pixPitch = self.reconInfo['Pixel Pitch']
        standoff = self.reconInfo['Standoff']
        zRez = reset*self.driftVelocity
        
        transDif = self.transDriftDifCoef*math.sqrt(drift)
        lonDif = self.lonDriftDifCoef*math.sqrt(drift)
        firstDifWidths = (transDif, transDif, lonDif)
        
        secondTransDif = self.transAmpDifCoef*math.sqrt(standoff/10000.) # Convert to cm
        secondLonDif = self.lonAmpDifCoef*math.sqrt(standoff/10000.)
        secondDifWidths = (secondTransDif, secondTransDif, secondLonDif)
        
        efficiencies = []
        
        trialNum = 0
        while trialNum < numTrials: 
            totalElecNum = len(self.rawData['z'])
            
            # Apply Gaussian smear to approximate initial drift diffusion
            smearData = self.diffuseData(self.rawData, firstDifWidths)
            
            # Discretize data to approximate falling into grid holes
            holeBins = {'x': holePitch, 'y': holePitch, 'z': 0}
            discreteData = self.discretizeData(smearData, holeBins)
            
            # Approximate diffusion from avalanche
            avalData = self.diffuseData(discreteData, secondDifWidths)

            # Discretize data to approximate pixels readout
            pixBins = {'x': pixPitch, 'y': pixPitch, 'z': 0}
            padData = self.discretizeData(avalData, pixBins)
            
            # Convert the z position to arrival time
            minZ = abs(min(padData['z']))
            padData['t'] = (padData['z'] + minZ)/self.driftVelocity
            
            # Determine how many electrons are seen by the readout. #TODO: improve
            groupedData = padData.groupby(['x','y']).agg(t=('t', list)).reset_index()
            check = [pixel for pixel in groupedData['t'] if len(pixel) > 1]
            
            numDrop = 0
            for pixel in check:
                elecID = 1
                pixel.sort()
                while elecID < len(pixel):
                    if abs(pixel[elecID] - pixel[elecID-1]) < reset:
                        numDrop += 1
                    elecID += 1
            
            singleEff = (totalElecNum - numDrop)/totalElecNum
            efficiencies.append(singleEff)

            trialNum += 1
        
        efficiency = self._calcAverage(efficiencies)
        
        return efficiency
    
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
        zRez = self.zRez
        
        transDif = self.transDriftDifCoef*math.sqrt(self.initialDriftDistance)
        lonDif = self.lonDriftDifCoef*math.sqrt(self.initialDriftDistance)
        firstDifWidths = (transDif, transDif, lonDif)
        
        # Apply Gaussian smear to approximate diffusion
        smearData = self.diffuseData(self.rawData, firstDifWidths)

        # Discretize data to approximate falling into grid holes and being read by the readout.
        bins = {'x': holePitch, 'y': holePitch, 'z': zRez}
        discreteData = self.discretizeData(smearData, bins)
        
        # Approximate avalanches
        # Diffusion is smaller than the pitch between pixels, so there
        # is zero net diffusion in the amplification region.
        numBelowThresh = int(len(discreteData)*0.05)
        belowID = np.random.choice(discreteData.index, size=numBelowThresh, replace=False)
        avalData = discreteData.drop(belowID).reset_index(drop=True)
        
        plotData = avalData.groupby(['x', 'y', 'z']).size().reset_index(name='q')

        # Plot data
        FIMSfig = self._format3DPlot(plotData, title='FIMS')
        
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
        zRez = self.zRez
        
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
        pixBins = {'x': pixPitch, 'y': pixPitch, 'z': zRez}
        readoutData = self.discretizeData(avalData2, pixBins)
        
        # Group Data by pixel
        groupedData = self._groupData(readoutData)
        
        # Configure data for plotting
        plotData = pd.DataFrame()
        plotData[['x', 'y']] = groupedData[['x', 'y']]
        plotData['z'] = groupedData['z'].apply(min)
        plotData['q'] = groupedData['q'].apply(sum)

        # Plot data
        beastFig = self._format3DPlot(plotData, title='BEAST')
        
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
        zRez = self.zRez
        
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
        pixBins = {'x': pixPitch, 'y': pixPitch, 'z': zRez}
        padData = self.discretizeData(avalData, pixBins)
        
        # Plot data
        # Extract Data
        totalXWidth = max(padData['x']) - min(padData['x'])
        totalYWidth = max(padData['y']) - min(padData['y'])
        numXBins = int(totalXWidth/pixPitch)
        numYBins = int(totalYWidth/pixPitch)
        
        # Create figure
        migdalFig = plt.figure()
        plt.hist2d(
            padData['x'],
            padData['y'],
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
        zRez = self.zRez
        
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
        pixBins = {'x': pixPitch, 'y': pixPitch, 'z': zRez}
        padData = self.discretizeData(avalData, pixBins)

        # Approximate Signal Readout
        readoutData = self.approximateReadout(padData)
        readoutData.dropna(inplace = True)
        
        # Format data for plotting
        plotData = readoutData.rename(columns={'crossing': 'z', 'ToT': 'q'})
        
        # Plot data
        gridPixFig = self._format3DPlot(plotData, title='GridPix')
        
        return gridPixFig

    #********************************************************************************#

