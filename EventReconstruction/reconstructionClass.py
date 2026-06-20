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
        reconstructFIMS
        reconstructBEAST
        reconstructMigdal (TODO: is this actually different from BEAST in terms of function?)
                
        TODO:
        GridPix wrapper
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
        return
    
    #********************************************************************************#

    def _checkInput(self):
        """Checks input to ensure all keys are present."""
        paramKeys = [
            'Gain',
            'Avalanche Sigma',
            'Pitch',
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
            rawData (list, tuples): the x,y,z coordinates of every electron
        """
        treeName = self.reconInfo['Tree Name']
        
        trialData = dataframes[treeName][['x', 'y', 'z']].iloc[0]
        rawData = list(zip(trialData['x']*10000, trialData['y']*10000, trialData['z']))
        
        return rawData
    
    #********************************************************************************#

    def _groupData(self, coordinates):
        """
        Takes the x,y,z coordinates and groups it by the x,y coordinate.
        
        Args:
            coordinates (list): list of x,y,z coordinates of each electron.
        
        returns:
            groupedData (list): z-coordinates with their corresponding 
            pixel location. 
        """
        #Convert to dataArray for faster data processing
        dataArray = pd.DataFrame(coordinates, columns=['x','y','z'])
        groupedArray = list(dataArray.groupby(['x', 'y'])['z'])
        
        #Convert to list for easier data accessing
        groupedData = []
        for elem in groupedArray:
            groupedData.append((elem[0], elem[1].tolist()))

        return groupedData

    #********************************************************************************#
    
    def _sortData(self, groupedData):
        """
        Condenses the data down to only the x,y coordinates and the charge profile.
        
        Args:
            groupedData (list): list of pixel locations with associated electron heights.
            
        Returns:
            screenedData (list): list of x,y coordinates along with total charge at
            a given height along with that height.
        """
        threshold = self.reconInfo['Signal Threshold']
        chargeProfile = []
        charges = []
        # Loop through every pixel
        for elem in groupedData:
            # Loop through every electron at the pixel
            for electron in elem[1]:
                Q = elem[1].count(electron)
                # Avoid duplicates
                if (electron, Q) not in charges:
                    charges.append((electron, Q))
            
            if len(charges) > 0:
                chargeProfile.append((int(elem[0][0]), int(elem[0][1]), sorted(charges)))
            charges = []

        return chargeProfile
    
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
            chargeProfile (list): list of positions with the amount of charge at that
            position.
        returns:
            rangeList (list): list of position points
            netSignal (list): list of signal strengths
        """
        minPos = min(list(zip(*chargeProfile))[0])
        maxPos = max(list(zip(*chargeProfile))[0])
        rangeList = np.arange(minPos, maxPos+800, 5) #TODO: find max position that ensures ToT always calculable

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

        return rangeList, netSignal
    
    #********************************************************************************#

    def _approximateToT(self, times, signal):
        """
        Approximates the TOT of a given group of electrons.
        
        args:
            times (list): list of timestamps for the signal
            signal (list): list signal strengths 
            
        returns:
            ToTList (list): ToTs for each separable charge bundle.
        """
        ToTStart = []
        ToTEnd = []
        threshold = self.reconInfo['Signal Threshold']
        values = range(0,len(signal))
        prev = 0
        
        #TODO: improve with array or tuple instead of two lists
        # Scan through the list of points and keep any where that cross the threshold.
        for index in values:
            current = signal[index]
            position = times[index]
            
            if prev < threshold and current >= threshold:
                ToTStart.append(position)
            if prev > threshold and current <= threshold:
                ToTEnd.append(position)
            
            prev = current
        
        # Take the list of start and end times and calculate the ToTs
        if ToTStart:
            ToTList = [end-start for start,end in list(zip(ToTStart,ToTEnd))]
        else:
            ToTList = None
            ToTStart = None
        
        return ToTStart, ToTList
    
    #********************************************************************************#

    def diffuseData(self, coordinates, diffusionWidths):
        """
        Applies a Gaussian smear to a given data set
        
        Args: 
            coordinates (list, tuple): x,y,z coordinates of each electron prior to
            diffusion.
            diffusionWidths (tuple): standard deviation of the Gaussian smear for each
            coordinate.
        
        Returns:
            diffusedData (list): list of all data points after being diffused
        """
        diffusedData = []
        for x,y,z in coordinates:
            diffusedData.append((
                float(x)+random.gauss(0, diffusionWidths[0]),
                float(y)+random.gauss(0, diffusionWidths[1]),
                float(z)+random.gauss(0, diffusionWidths[2])
            ))
        # TODO: np.random.normal doesn't easily allow for applying different diffusion constants
        # to different axes
        
        # Convert to np.array for efficiency
        #inputArray = np.array(inputList, dtype=float)

        # Add random offsets to each coordinate
        #diffusionAmount = np.random.normal(scale=diffusionWidths, size=inputArray.shape)        
        #inputArray += diffusionAmount

        # Convert back to list of tuples
        #diffusedData = [tuple(row) for row in inputArray]

        
        return diffusedData
        
    #********************************************************************************#

    def discretizeData(self, inputData, binSize):
        """
        Bins a given dataset
        
        Note: assumes units are given in microns
        
        Args:
            inputData (list): list of data points to be binned.
            binSize (tuple): width of each bin.
        
        returns:
            discreteData (list): list of discretized coordinates.
        """
        # Convert to pandas dataframe
        inputArray = pd.DataFrame(inputData, columns=['x','y','z'])
        
        # Bin data
        binDict = {}
        binID = 0
        bound = 15000
        for column in inputArray:
            # Check if data has a bin size
            if binSize[binID] == 0 or binSize[binID] == None:
                discreteData = list(inputArray[column])
            else:
                binEdges = np.arange(-bound, bound, binSize[binID])
                binnedData = pd.cut(inputArray[column], binEdges)
                discreteData = [electron.left+int(binSize[binID]/2) for electron in binnedData]
            binDict[column] = discreteData
            binID += 1

        # Combine data into list of tuples
        discreteData = list(zip(binDict['x'], binDict['y'], binDict['z']))
        
        return discreteData

    #********************************************************************************#

    def approximateGain(self, position=(0,0,0)):
        """
        Takes the x,y coordinates of an electron and approximates an avalanche.
        
        Note: does not apply any diffusion to the new electrons.
        
        args:
            position (tuple): the x,y, and z coordinates of the initial electron
        
        Note: each new electron will have the same coordinates as the initial electron.
        
        returns:
            avalData (list): list of x,y,z coordinates for each new electron
        """
        gain = self.reconInfo['Gain']
        sigma = self.reconInfo['Avalanche Sigma']
        
        avalancheSize = int(random.gauss(gain, sigma))
        avalData = [position]*avalancheSize

        return avalData

    #********************************************************************************#

    def approximateReadout(self, inputData):
        """
        Takes a charge distribution and approximates the readout values for each pixel.

        Args:
            inputData (list): x,y,z coordinates of each charge.
        returns:
            readoutData (list): x,y,z coordinates of the charge bundles as well as the time over threshold.
        """
        # Group data by pixel
        groupedData = self._groupData(inputData)
        
        # Sort and compress data by pixel
        pixelData = self._sortData(groupedData)
        
        # Calculate ToT
        readoutData = []
        for x,y,chargeProfile in pixelData:
            rangeList, netSignal = self._convertToSignal(chargeProfile)
            z, ToT = self._approximateToT(rangeList, netSignal)
            readoutData.append((x,y,z,ToT))
        
        return readoutData
    
    #********************************************************************************#
    ############## Reconstruction Wrapper Functions for Specific Setups ##############
    #********************************************************************************#
    
    def reconstructFIMS(self, includeRaw=False):
        """
        Approximates an event reconstruction using a FIMS readout.
        
        Amplification produced by a thin aluminum mesh that induces amplification 
        below it. Uses a pixel-pad readout with instant reset time, enabling clear 
        distinction of electrons in the vertical direction.
        
        args:
            includeRaw (bool): optionally include raw data in plot.
        
        returns:
            FIMSfig: matplotlib figure
        """
        # Extract relevant data from dictionary and set constant values
        pitch = self.reconInfo['Pitch']
        timeRez = 25
        transDif = 1600
        lonDif = 1000
        
        # Apply Gaussian smear to approximate diffusion
        smearData = self.diffuseData(self.rawData, (transDif, transDif, lonDif))

        # Discretize data to approximate falling into grid holes
        discreteData = self.discretizeData(smearData, (pitch, pitch, 0)).copy()
        
        # Assume one hole per readout pixel, so data does not need to be discretized a
        # second time. TODO: data should probably still be avalanched
        
        ## Plot Data ##
        # Create figure
        FIMSfig = plt.figure()
        FIMS3D = FIMSfig.add_subplot(projection='3d')
        
        # Extract Data
        xPlot = [elem[0] for elem in discreteData]
        yPlot = [elem[1] for elem in discreteData]
            
        FIMS3D.scatter(
            xPlot, 
            yPlot, 
            s=.02,
            c='g',
            label=f'Binned Fims Data [Pitch = {pitch} (\u03BCm)]'
        )
        
        if includeRaw:
            xRawPlot = [elem[0] for elem in rawData]
            yRawPlot = [elem[1] for elem in rawData]
            
            FIMS3D.scatter(
                xRawPlot,
                yRawPlot,
                s=.01,
                c='b',
                label = 'Raw Data'
            )

        plt.xlabel('x pixels')
        plt.ylabel('y pixels')
        plt.legend(markerscale=20)
        plt.title('FIMS Event Reconstruction')
        
        return FIMSfig
        
    #********************************************************************************#
    
    def reconstructBEAST(self):
        """
        Approximates an event reconstruction using a BEAST readout.
        
        Uses GEM amplification structure. Post avalanche electrons are then drifted 
        again before reaching a pixel readout. Readout uses infinite integration time,
        so final readout is purely 2D (x,y).
        
        returns:
            BEASTfig: matplotlib figure
        """
        # Extract relevant data from dictionary
        pitch = self.reconInfo['Pitch']
        timeRez = 25
        transDif = 1600
        lonDif = 1000

        # Apply Gaussian smear to approximate diffusion
        smearData = self.diffuseData(self.rawData, (transDif, transDif, lonDif))

        # Discretize data to approximate falling into grid holes
        discreteData = self.discretizeData(smearData, (pitch, pitch, 0)).copy()
        
        ## Approximate avalanches ##
        avalData = discreteData.copy()

        # Approximate avalanche
        electronID = 0
        for xElec,yElec,zElec in discreteData:
            # Multiply each individual electron to approximate gain
            newElectrons = self.approximateGain((xElec, yElec, zElec)).copy()
            
            # Diffuse each new electron to approximate diffusion during avalanche
            if newElectrons:
                #TODO: find better diffusion metric for diffusion incurred during avalanche
                newDiffused = self.diffuseData(newElectrons, (transDif/100, transDif/100, lonDif/100)).copy()
                avalData.extend(newDiffused)
            newDiffused = []
        
        # Discretize data to approximate pixels readout
        readoutData = self.discretizeData(avalData, (pitch, pitch, timeRez)).copy()
        
        
        ## Plot BEAST data ##
        # Extract Data
        xPlot = [elem[0] for elem in readoutData]
        yPlot = [elem[1] for elem in readoutData]
        totalXWidth = max(readoutData[0]) - min(readoutData[0])
        totalYWidth = max(readoutData[1]) - min(readoutData[1])
        numXBins = int(totalXWidth/timeRez)
        numYBins = int(totalYWidth/timeRez)
        
        # Create figure
        BEASTfig = plt.figure()
        plt.hist2d(xPlot, yPlot, bins=(numXBins, numYBins))
        
        # Add plot elements
        plt.colorbar().set_label('Charge')
        plt.xlabel('x pixels')
        plt.ylabel('y pixels')
        plt.title('BEAST Event Reconstruction')
        
        return BEASTFig

    #********************************************************************************#
    
    def reconstructMigdal(self):
        """
        Approximates an event reconstruction using the Migdal experiment readout.
        
        Uses a THGEM-esque amplification structure. Post avalanche electrons are then
        drifted 2 mm before reaching a pixel readout. Readout uses long integration
        time, so final readout is functionally 2D (x,y).
        
        returns:
            migdalfig: matplotlib figure
        """
        # Extract relevant data from dictionary
        pitch = self.reconInfo['Pitch']
        timeRez = 25
        transDif = 1600
        lonDif = 1000

        # Apply Gaussian smear to approximate diffusion
        smearData = self.diffuseData(self.rawData, (transDif, transDif, lonDif))

        # Discretize data to approximate falling into grid holes
        discreteData = self.discretizeData(smearData, (pitch, pitch, 0)).copy()
        
        ## Approximate avalanches ##
        avalData = discreteData.copy()

        # Approximate avalanche
        electronID = 0
        for xElec,yElec,zElec in discreteData:
            # Multiply each individual electron to approximate gain
            newElectrons = self.approximateGain((xElec, yElec, zElec)).copy()
            
            # Diffuse each new electron to approximate diffusion during avalanche
            if newElectrons:
                #TODO: find better diffusion metric for diffusion incurred during avalanche
                newDiffused = self.diffuseData(newElectrons, (transDif/100, transDif/100, lonDif/100)).copy()
                avalData.extend(newDiffused)
            newDiffused = []
        
        # Discretize data to approximate pixels readout
        padData = reconstruction.discretizeData(avalData, (pitch, pitch, timeRez))

        # Approximate Signal Readout
        readoutData = reconstruction.approximateReadout(padData)
        
        ## Plot Migdal data ##
        # Extract Data
        xPlot = [elem[0] for elem in readoutData]
        yPlot = [elem[1] for elem in readoutData]
        totalXWidth = max(readoutData[0]) - min(readoutData[0])
        totalYWidth = max(readoutData[1]) - min(readoutData[1])
        numXBins = int(totalXWidth/timeRez)
        numYBins = int(totalYWidth/timeRez)
        
        # Create figure
        BEASTfig = plt.figure()
        plt.hist2d(xPlot, yPlot, bins=(numXBins, numYBins))
        
        # Add plot elements
        plt.colorbar().set_label('Charge')
        plt.xlabel('x pixels')
        plt.ylabel('y pixels')
        plt.title('Migdal Experiment Event Reconstruction')
        
        return migdalFig

    #********************************************************************************#

