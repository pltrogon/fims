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
            _sortData
            _decayEquation
            _convertToSignal
            _approximateToT
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

    def _groupData(self, coordinates):
        """
        Takes the x,y,z coordinates and groups it by the x,y coordinate.
        
        Args:
            coordinates (dataframe): list of x,y,z coordinates of each electron.
        
        returns:
            groupedData (dataframe): z-coordinates with their corresponding 
            pixel location. 
        """
        groupedArray = list(coordinates.groupby(['x', 'y'])['z'])
        
        groupedData = []
        for elem in groupedArray:
            groupedData.append((elem[0], elem[1].tolist()))
        groupedDataFrame = pd.DataFrame(groupedData, columns=['pixel id','z'])

        return groupedDataFrame

    #********************************************************************************#
    
    def _sortData(self, inputData):
        """
        Condenses the data down to only the x,y coordinates and the charge profile.
        
        Args:
            inputData (dataframe): pixel locations (x,y) with associated electron heights.
            
        Returns:
            inputData (dataframe): original dataframe with an extra column for the charge
            profile.
        """
        #TODO: improve efficiency
        def getChargeProfile(val):
            charges = []
            for electron in val:
                Q = val.count(electron)
                # Avoid duplicates
                if (electron, Q) not in charges:
                    charges.append((electron, Q))
            
            if len(charges) > 0:
                return np.array(sorted(charges))
            else:
                return 0

        inputData['charge profile'] = inputData['z'].map(lambda z: getChargeProfile(z))
        
        return inputData
    
    #********************************************************************************#
    
    def _decayEquation(self, charge, time, startTime):
        """Equation modeling the decay rate of a signal."""
        
        decayRate = self.reconInfo['Signal Decay Rate']
        chargeHeight = charge*math.e**(-(time-startTime)/decayRate)
        
        return chargeHeight
    
    #********************************************************************************#
        
    def _convertToSignal(self, chargeProfile):
        """
        Takes the charge profile and converts it to a signal.
        
        args:
            chargeProfile (dataframe): heights of each electron with the amount of
            charge at that height.
        
        returns:
            signalPlot (np.array): array of signal points.
        """
        #TODO: improve efficiency
        threshold = self.reconInfo['Signal Threshold']
        decayRate = self.reconInfo['Signal Decay Rate']
        
        # Generate a signal range that guarantees that the full signal is included
        minPos = min(list(zip(*chargeProfile))[0])

        maxPos = max(list(zip(*chargeProfile))[0])
        maxCharge = max(list(zip(*chargeProfile))[1])
        maxTime = maxPos - math.log(threshold/maxCharge)*decayRate
        
        minRange = int(minPos)
        maxRange = int(max(maxTime, maxPos)+500)*2
        rangeList = np.arange(minPos, maxRange, self.timeRez)

        decayLists = []
        for pos, amount in chargeProfile:
            plotDecay = []
            for elem in rangeList:
                if elem < pos:
                    plotDecay.append(0)
                else:
                    plotDecay.append(self._decayEquation(amount, elem, pos))
                
            decayLists.append(plotDecay)
        
        netSignal = [sum(group) for group in zip(*decayLists)]
        
        signalPlot = np.array([rangeList, netSignal])
        
        return signalPlot
    
    #********************************************************************************#

    def _approximateToT(self, signalPlot):
        """
        Approximates the TOT of a given group of electrons.
        
        args:
            signalPlot (np.array): array of signal points 
            
        returns:
            ToTDF (dataframe): dataframe of initial crossing times and ToTs
        """
        ToTStart = []
        ToTEnd = []
        ToTList = []
        threshold = self.reconInfo['Signal Threshold']
        
        # Identify all the points above threshold
        aboveCheck = signalPlot[1] >= threshold
        above = signalPlot[0]*aboveCheck
        aboveID = np.flatnonzero(above)
        
        # Verify that the signal crosses threshold at least once
        if len(aboveID) == 0:
            return np.nan
        
        # Find all the upwards crossing points
        ID = 0
        while ID < len(aboveID):
            check = aboveID[ID]-aboveID[ID-1]
            if check != 1:
                ToTStart.append(above[aboveID[ID]])
            ID+=1

        # Identify all the points below threshold after the first upwards crossing
        belowCheck = signalPlot[1][aboveID[0]:] < threshold
        below = signalPlot[0][aboveID[0]:]*belowCheck
        belowID = np.flatnonzero(below)

        # Find all the downwards crossing points
        ID = 0
        while ID < len(belowID):
            check = belowID[ID]-belowID[ID-1]
            if check != 1:
                ToTEnd.append(below[belowID[ID]])
            ID+=1
        # Ensure every start has an end. Pad lists with edge values if not.
        if len(ToTStart) != len(ToTEnd):
            max_len = max(len(ToTStart), len(ToTEnd))
            ToTStart += [min(signalPlot[0])] * (max_len - len(ToTStart))
            ToTEnd += [max(signalPlot[0])] * (max_len - len(ToTEnd))
        
        # Take the list of start and end times and calculate the ToTs
        if ToTStart:
            ToTList = [end-start for start,end in list(zip(ToTStart,ToTEnd))]
    
        return (ToTStart, ToTList)
    
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
        diffusedData = pd.DataFrame({
            'x': coordinates['x'].map(lambda x: x + random.gauss(0, diffusionWidths[0])),
            'y': coordinates['y'].map(lambda y: y + random.gauss(0, diffusionWidths[1])),
            'z': coordinates['z'].map(lambda z: z + random.gauss(0, diffusionWidths[2]))
        })

        return diffusedData
        
    #********************************************************************************#

    def discretizeData(self, inputArray, binSize):
        """
        Bins a given dataset
        
        Note: assumes units are given in microns
        
        Args:
            inputArray (dataframe): list of data points to be binned.
            binSize (dataframe): width of each bin.
        
        returns:
            discreteData (list): list of discretized coordinates.
        """
        # Get bounds
        boundMin = abs(np.min(inputArray))
        boundMax = abs(np.max(inputArray))
        rawBound = max(boundMin, boundMax) + 100
        bound = round(rawBound/10)*10
        
        discreteDataFrame = pd.DataFrame({
            'x': [],
            'y': [],
            'z': []
        })
        binID = 0
        for column in inputArray:
            # Check if data has a bin size
            if binSize[binID] == 0 or binSize[binID] == None:
                discreteData = list(inputArray[column])
            else:
                binEdges = np.arange(-bound, bound, binSize[binID])
                binnedData = pd.cut(inputArray[column], binEdges)
                discreteData = [electron.left+int(binSize[binID]/2) for electron in binnedData]
            discreteDataFrame[column] = discreteData
            binID += 1

        return discreteDataFrame

    #********************************************************************************#
    
    def avalancheData(self, coordinates, difWidths):
        """
        Takes the x,y,z coordinates of an electron dataframe and approximates an avalanche.
        
        Note: also applies diffusion to the new electrons
        
        args:
            coordinates (dataframe): the x,y,z coordinates of each initial electron
            difWidths (tuple, floats): diffusion values for each axis
        
        returns:
            avalData (dataframe): list of x,y,z coordinates for each new electron
        """
        sigma = self.reconInfo['Avalanche Sigma']
        meanGain = self.reconInfo['Gain']
        index = 0
        
        postAvalElec = []
        while index < len(coordinates['x']): 
            gain = int(random.gauss(meanGain, sigma))
            if gain > 0:
                postAvalElec.append(np.random.normal(
                    loc=(coordinates['x'][index], coordinates['y'][index], coordinates['z'][index]),
                    scale=difWidths,
                    size=(gain,3)
                ))
            index+=1
        
        postDataFrame = pd.concat([pd.DataFrame(elem, columns=['x','y','z']) for elem in postAvalElec])
        
        return postDataFrame
    
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
        # Group data by pixel
        groupedData = self._groupData(inputData)
        
        # Sort and compress data by pixel
        readoutData = self._sortData(groupedData)
        
        # Convert charge to voltage signal
        vectSignalFunction = np.vectorize(self._convertToSignal, otypes=[np.ndarray])
        readoutData['signal'] = pd.DataFrame(
            vectSignalFunction(readoutData['charge profile'].values), 
            columns=['signal']
        )
        
        # Use voltage signal to calculate ToT
        vectToTFunction = np.vectorize(self._approximateToT, otypes=[list])
        readoutData['ToT'] = pd.DataFrame(
            vectToTFunction(readoutData['signal'].values),
            columns = ['ToT']
        )
        
        # Remove depreciated columns and rows with no ToT
        readoutData.drop(columns=['signal','charge profile', 'z'], inplace=True)
        readoutData = readoutData.dropna(how='any')
        
        return readoutData
    
    #********************************************************************************#
    
    def format3DPlot(self, plotData, title='', charge=False):
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
            plotData['x'], 
            plotData['y'],
            plotData['z'],
            s=.2,
            c=color,
            cmap='viridis',
            label=f'{title} Readout Data'
        )
        
        sub2DRef = sub2D.scatter(
            plotData['x'], 
            plotData['y'],
            s=.3,
            c=color,
            cmap='viridis',
            label=f'{title} Readout Data'
        )
        
        # Add color bar
        if charge:
            colorBar = plt.colorbar(
                sub2DRef, 
                pad=.2
            )
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
        rawFig = self.format3DPlot(self.rawData, title='Raw Data')
        
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
        pitch = self.reconInfo['Hole Pitch']
        pixPitch = self.reconInfo['Pixel Pitch']
        timeRez = self.timeRez
        
        transDif = self.transDriftDifCoef*math.sqrt(self.initialDriftDistance)
        lonDif = self.lonDriftDifCoef*math.sqrt(self.initialDriftDistance)
        firstDifWidths = (transDif, transDif, lonDif)
        
        # Apply Gaussian smear to approximate diffusion
        smearData = self.diffuseData(self.rawData, firstDifWidths)

        # Discretize data to approximate falling into grid holes
        discreteData = self.discretizeData(smearData, (pitch, pitch, 0))
        
        # Approximate avalanches
        # Diffusion is smaller than the pitch between pixels, so there
        # is zero net diffusion in the amplification region.
        numBelowThresh = int(len(discreteData)*0.05)
        belowID = np.random.choice(discreteData.index, size=numBelowThresh, replace=False)
        avalData = discreteData.drop(belowID).reset_index(drop=True)
        
        # Plot data
        FIMSfig = self.format3DPlot(avalData, title='FIMS')
        
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
        pitch = self.reconInfo['Hole Pitch']
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
        discreteData1 = self.discretizeData(smearData, (pitch, pitch, 0))
        
        # Approximate first set of avalanches
        avalData1 = self.avalancheData(discreteData1, secondDifWidths)
        
        # Discretize data to approximate falling into second GEM holes
        discreteData2 = self.discretizeData(avalData1, (pitch, pitch, 0))
        
        # Approximate second set of avalanches
        avalData2 = self.avalancheData(discreteData2, secondDifWidths)
        
        # Discretize data to approximate pixels readout
        readoutData = self.discretizeData(avalData2, (pixPitch, pixPitch, timeRez))
        
        # Configure data for plotting
        groupData = self._groupData(readoutData)
        plotData = pd.DataFrame({
            'x': [x for x,y in groupData['pixel id']],
            'y': [y for x,y in groupData['pixel id']],
            'z': [min(z) for z in groupData['z']],
            'q': [len(z) for z in groupData['z']]
        })
        
        # Plot data
        beastFig = self.format3DPlot(plotData, title='BEAST', charge=True)
        
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
        pitch = self.reconInfo['Hole Pitch']
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
        discreteData = self.discretizeData(smearData, (pitch, pitch, 0))
        
        # Approximate avalanches
        avalData = self.avalancheData(discreteData, secondDifWidths)
        
        # Discretize data to approximate pixels readout
        padData = self.discretizeData(avalData, (pixPitch, pixPitch, timeRez))

        # Approximate Signal Readout
        #readoutData = self.approximateReadout(padData)
        readoutData = padData.copy()
        
        ## Plot Migdal data ##
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
        pitch = self.reconInfo['Hole Pitch']
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
        discreteData = self.discretizeData(smearData, (pitch, pitch, 0))
        
        # Approximate avalanche
        avalData = self.avalancheData(discreteData, secondDifWidths)

        # Discretize data to approximate pixels readout
        padData = self.discretizeData(avalData, (pitch, pitch, timeRez))

        # Approximate Signal Readout
        readoutData = self.approximateReadout(padData)

        ###########################################################################
        # Get data and combine into single list
        xPlot = [x for x,y in readoutData['pixel id']]
        yPlot = [y for x,y in readoutData['pixel id']]
        zPlot = [z for z,time in readoutData['ToT']]
        densityPlot = [time for z,time in readoutData['ToT']]
        fullZip = list(zip(xPlot, yPlot, zPlot, densityPlot))

        # Unpack pixels with multiple charge bundles
        extendedList = []
        for x,y,z,t in fullZip.copy():
            if len(z) > 1:
                for height, time in list(zip(z,t)):
                    extendedList.append((int(x),int(y),int(height),int(time)))
            else:
                extendedList.append((int(x),int(y),int(z[0]),int(t[0])))

        # Re-seperate data for plotting
        plotData = pd.DataFrame({
            'x': [x for x,y,z,t in extendedList],
            'y': [y for x,y,z,t in extendedList],
            'z': [z for x,y,z,t in extendedList],
            'q': [t for x,y,z,t in extendedList],
        })
        ###########################################################################
        # Plot data
        gridPixFig = self.format3DPlot(plotData, title='GridPix', charge=True)
        
        return gridPixFig

    #********************************************************************************#

