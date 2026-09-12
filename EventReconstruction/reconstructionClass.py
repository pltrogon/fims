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
            _getData
            _getCoordinates
            _calcAverage
            _approximateMIP
            _convertToSignal
        
        Public:
            diffuseData
            discretizeData
            avalancheData
            approximateReadout

        ## Wrapper functions ##
        getPileup
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
        self.trialID = 0
        self.allTrials = self._getData()
        self.rawData = self._getCoordinates()

        # Values for T2K gas assuming 0.28 KV/cm drift field and ~140 KV/cm amplification field
        self.driftVelocity = 80 # microns/ns 
        
        self.transDriftDifCoef = 320 # microns/sqrt(cm)
        self.lonDriftDifCoef = 200 # microns/sqrt(cm)
        
        self.transAmpDifCoef = 190 # microns/sqrt(cm)
        self.lonAmpDifCoef = 150 # microns/sqrt(cm)
        
        # Set constant values
        self.timeRez = 25 # ns
        self.zRez = self.timeRez*self.driftVelocity # microns
        self.initialDriftDistance = 10 # cm
        
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

    def _getData(self):
        """
        Unpacks a root file from a given filename and tree name.
        
        Note: Assumes coordinates are given in cm and converts them to microns.
        
        returns:
            allTrials (pandas dataframe): unpacked root file
        """
        filePath = self.reconInfo['File Location']
        treeName = self.reconInfo['Tree Name']
        
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
        
        # Get data of a single trial
        fileData = dataframes[treeName][['x', 'y', 'z']]
        
        # convert units to microns
        allTrials = fileData*10000
        
        # z is relative, so the minimum value is set to zero for simplicity.
        allTrials['z'] = allTrials['z'].apply(lambda row: np.array(row) - min(row))
        
        return allTrials

    #********************************************************************************#
    
    def _getCoordinates(self):
        """
        Takes a given dataframe and extracts the x,y,z coordinates from a single trial
        
        
        returns:
            rawData (dataframe): the x,y,z coordinates of every electron
        """
        singleTrial = self.allTrials.iloc[self.trialID]
        rawData = pd.DataFrame(zip(*singleTrial), columns=['x', 'y', 'z'])
        
        return rawData
    
    #********************************************************************************#
    
    def _approximateMIP(self):
        """
        Creates a line of approximate MIP interaction electrons.
        
        Returns: 
            rawData (dataframe): the x,y,z coordinates of every electron
        """
        dEdX = .2525 # eV/micron
        dx = 10400 # microns
        minE = 26.0 # eV
        numElec = int(dEdX * dx / minE)
        
        # Randomly assign a location for each point along the line.
        start = np.array([-5700, -5200, 0], dtype=float)
        end = np.array([0, 0, 7000], dtype=float)
        points = np.random.rand(numElec)
        data = start + np.outer(points, (end - start))
        rawData = pd.DataFrame(data, columns=['x', 'y', 'z'])
        
        return rawData

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
        
    def _convertToSignal(self, tLocs, charges):
        """
        Takes data of a single pixel and calculates ToT and threshold crossing time.
        
        args:
            tLocs (list of floats): time of arrival of the charges
            charges (list of ints): amount of charge
        
        returns:
            upCrossPoints (list): list of threshold crossing times
            ToTList (list): list of ToT times
        """
        threshold = self.reconInfo['Signal Threshold']
        decayRate = self.reconInfo['Signal Decay Rate']

        times = np.asarray(tLocs)
        charge = np.asarray(charges)

        chargeSum = charge.sum()
        tMin= times.min()
        tMax = times.max() - np.log(threshold/chargeSum)*decayRate

        if tMin >= tMax:
            return [], []

        rangeList = np.linspace(tMin, tMax, 1000)

        #Matrix multiplication for signals
        dt = rangeList[:, np.newaxis] - times[np.newaxis, :]
        decay = np.where(dt >= 0, np.exp(-dt/decayRate), 0.0)
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
        
        # Convert the z position to arrival time
        inputData['t'] = inputData['z']/self.driftVelocity
        
        # Group data by pixel
        countedData = inputData.groupby(['x', 'y', 't']).size().reset_index(name='q')
        groupedData = countedData.groupby(['x','y']).agg(t=('t', list), q=('q',list)).reset_index()

        chargeSum = [sum(q) for q in groupedData['q']]
        chargeLen = [len(q) for q in groupedData['q']]
        chargeMask = [(s > threshold) and (s > l) for s, l in zip(chargeSum, chargeLen)]

        filteredData = groupedData[chargeMask].copy()
        if filteredData.empty:
            raise ValueError('Empty Dataframe.')
        
        # Calculate ToT by converting charge to voltage
        print('Calculating ToT...')
        signals = [
            self._convertToSignal(t, q)
            for t, q in zip(filteredData['t'], filteredData['q'])
        ]

        # Unpack results and remove depreciated columns
        crossings, tots = zip(*signals)
        filteredData['crossing'] = crossings
        filteredData['ToT'] = tots
        filteredData.drop(columns=['t', 'q'], inplace=True)

        readoutData = filteredData.explode(['crossing', 'ToT'], ignore_index=True)
        
        return readoutData

    #********************************************************************************#
    ############## Reconstruction Wrapper Functions for Specific Setups ##############
    #********************************************************************************#
    
    def getPileup(self, drift=10, reset=25, numTrials=100, MIP=False):
        """
        Determines the efficiency for a readout based on given input parameters.
        
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
            # Get new set of coordinates
            if MIP:
                trialData = self._approximateMIP()
            else:
                self.trialID = trialNum
                trialData = self._getCoordinates()
            totalElecNum = len(trialData['z'])
            
            # Apply Gaussian smear to approximate diffusion
            smearData = self.diffuseData(trialData, firstDifWidths)
            
            # Discretize data to approximate falling into grid holes.
            bins = {'x': holePitch, 'y': holePitch, 'z': 0}
            discreteData = self.discretizeData(smearData, bins)
            
            # Convert the z position to arrival time and sort by that time
            discreteData['t'] = discreteData['z']/self.driftVelocity
            discreteData.sort_values(by='t', inplace=True)
            
            # Group data by pixel
            groupedData = discreteData.groupby(['x','y']).agg(t=('t', list), q=('t', lambda z: len(z))).reset_index()
            filteredData = groupedData[groupedData['q'] > 1] # remove pixels with only 1 electron
            
            # Determine how many electrons are NOT seen by the readout.
            dropped = []

            # Loop through every pixel group
            for pixel in filteredData['t']:
                elecID = 0
                
                # Loop through all electron IDs
                while elecID+1 < len(pixel): 
                    if pixel[elecID+1] - pixel[elecID] < reset:
                        dropped.append(pixel.pop(elecID+1))
                        continue
                    elecID += 1
            numDrop = len(dropped)
            
            # Calculate the efficiency of this trial
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

        # Discretize data to approximate falling into grid holes
        bins = {'x': holePitch, 'y': holePitch, 'z': 0}
        discreteData = self.discretizeData(smearData, bins)
        
        # Approximate avalanches
        # Diffusion is smaller than the pitch between pixels, so there
        # is zero net diffusion in the amplification region.
        numBelowThresh = int(len(discreteData)*0.05)
        belowID = np.random.choice(discreteData.index, size=numBelowThresh, replace=False)
        avalData = discreteData.drop(belowID).reset_index(drop=True)
        
        # Discretize in z by removing pileup electrons
        avalData.sort_values(by='z', inplace=True)
        groupedData = avalData.groupby(['x','y']).agg(z=('z', list)).reset_index()
        groupedData.sort_values(by=['x','y'], inplace=True)
        
        dropped = []
        # Loop through all pixels
        for x, y, height in zip(groupedData['x'], groupedData['y'], groupedData['z']):
            elecID = 0
            
            # Loop through all electron IDs
            while elecID+1 < len(height): 
                if height[elecID+1] - height[elecID] < zRez:
                    dropped.append([x,y, height.pop(elecID+1)])
                    continue
                elecID += 1
        plotData = groupedData.explode(['z'], ignore_index=True)
        droppedData = pd.DataFrame(dropped, columns = ['x','y','z'])
        
        return plotData, droppedData
        
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
        
        # Configure data for plotting
        plotData = readoutData.groupby(['x','y']).agg(z=('z', 'min'), q=('q', 'sum')).reset_index()
        
        return plotData

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
        
        return padData

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
        pixBins = {'x': pixPitch, 'y': pixPitch, 'z': 1}
        padData = self.discretizeData(avalData, pixBins)
            
        # Approximate Signal Readout
        readoutData = self.approximateReadout(padData)
        readoutData.dropna(inplace = True)
        
        # Convert the crossing time to z position and ToT to charge
        chargeConvConst = .075 # ns/electron
        readoutData['crossing'] *= self.driftVelocity
        readoutData['ToT'] /= chargeConvConst
        readoutData.rename(columns={'crossing': 'z', 'ToT': 'q'}, inplace=True)
        readoutData.sort_values(by=['z'], inplace=True)
        
        # Remove charge lost due to sensor dead time
        filteredData = readoutData.groupby(['x','y']).agg(z=('z', list), q=('q', list)).reset_index()
        filterZip = zip(filteredData['x'], filteredData['y'], filteredData['z'], filteredData['q'])
        
        dropped = []
        # Loop through all pixels
        for x, y, height, q in filterZip:
            elecID = 0
            
            # Loop through all electron IDs
            while elecID+1 < len(height): 
                if height[elecID+1] - height[elecID] < zRez:
                    dropped.append([x,y, height.pop(elecID+1), q.pop(elecID+1)])
                    continue
                elecID += 1
        droppedData = pd.DataFrame(dropped, columns = ['x','y','z', 'q'])
        
        plotData = filteredData.explode(['z', 'q'], ignore_index=True)
        
        return plotData, droppedData

    #********************************************************************************#

