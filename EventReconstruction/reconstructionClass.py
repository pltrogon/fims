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
    Functions:
        ## Base functions ##
        _checkInput
        _getDataFrames
        _getCoordinates
        diffuseData
        discretizeData
        approximateGain
        groupData (TODO: convert to screenData)
        approximateToT
        
        TODO (remaining methods)
        More?
        
        ## Wrapper functions ##
        reconstructFIMS
        reconstructBEAST (TODO: add dataScreen to remove data below threshold)
        
        TODO
        Migdal wrapper
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
            'Horizontal Diffusion',
            'Vertical Diffusion',
            'Pitch',
            'Time Resolution',
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
        
        Note: assumes total area of 1x1 cm and input units are given in microns
        
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
        bound = 10000
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

    def groupData(self, coordinates, pixelID):
        """
        Sorts the data of a given x-y coordinate into groups based on their z
        distribution.
        
        Args:
            coordinates (list): list of x,y,z coordinates of each electron.
            pixelID (int): ID of the pixel being grouped.
        
        returns:
            groupedData (dict): arranged z-coordinates with their corresponding 
            group ID. 
        """
        timeRez = self.reconInfo['Time Resolution']
        
        # Identify and grab all electrons on a single pixel
        isolatedZ = []
        index = 0
        for elecX in coordinates['x']:
            if elecX == coordinates['x'][pixelID] and coordinates['y'][index] == coordinates['y'][pixelID]:
                isolatedZ.append(coordinates['z'][index])
            index += 1

        # Sort all electrons on pixel into groups based on z-distance
        elecPositionList = []
        elecID = 0
        groupIDList = []
        groupID = 1
        arrangedElectrons = sorted(isolatedZ)
        for electron in arrangedElectrons:
            if electron - arrangedElectrons[elecID-1] > timeRez*2:
                groupID += 1
            elecPositionList.append(electron)
            groupIDList.append(groupID)
            elecID += 1

        groupedData = {
            'z': elecPositionList,
            'group ID': groupIDList
        }

        return groupedData

    #********************************************************************************#

    def approximateToT(self, timeList):
        """
        Approximates the TOT of a given group of electrons.
        
        args:
            timeList (list): z-positions of electrons converted into arrival time.
            
        returns:
            ToTList (list): ToTs for each separable charge bundle.
        """
        # Find every point where the total charge in the group exceeds the threshold
        startCharge = []
        startPoint = []
        threshold = self.reconInfo['Threshold']
        decayRate = self.reconInfo['Signal Decay Rate']
        
        for position in timeList:
            totalCharge = timeList.count(position)
            timeList = [z for z in timeList if z != position]
            if totalCharge > 3:
                startPoint.append(position)
                startCharge.append(totalCharge)
                

        # Determine the crossing points of the ToT
        startTime = startPoint[0]

        # Equations for ToT decay
        def decayEquation(charge, time, startTime, decayRate):
            chargeHeight = charge*math.e**(-(time-startTime)/decayRate)
            return chargeHeight

        def findCrossing(charge, threshold, startTime, decayRate):
            crossingTime = -math.log(threshold/charge)*decayRate + startTime
            return crossingTime
        
        ToTList = []
        activeCharge = startCharge[0]
        activeTime = startTime
        testIndex = 0
        for charge in startCharge:
            # Verify start time is set. If not, set a start time
            if not startTime:
                startTime = startPoint[testIndex]
                activeCharge = charge
                activeTime = startPoint[testIndex]
            
            # Check if decay has reached the threshold
            if decayEquation(activeCharge, startPoint[testIndex], activeTime, decayRate) <= threshold:
                endTime = findCrossing(activeCharge, threshold, activeTime, decayRate)
                ToT = abs(endTime - startTime)
                ToTList.append(ToT)
                startTime = False

                if charge > threshold:
                    activeCharge = charge
                    startTime = startPoint[testIndex]
                    activeTime = startTime
                    
            # Check if decay is reset with later charge
            elif charge > decayEquation(activeCharge, startPoint[testIndex], activeTime, decayRate):
                activeCharge = charge
                activeTime = startPoint[testIndex]
                
            testIndex += 1
        # End ToT loop

        # Find end point of final charge bundle
        endTime = findCrossing(activeCharge, threshold, activeTime, decayRate)
        ToT = abs(endTime - startTime)
        ToTList.append(ToT)
        startTime = False

        return ToTList

    #********************************************************************************#
    ############## Reconstruction Wrapper Functions for Specific Setups ##############
    #********************************************************************************#
    
    def reconstructFIMS(self, includeRaw=False):
        """
        Approximates an event reconstruction using a FIMS readout.
        
        args:
            includeRaw (bool): optionally include raw data in plot.
        
        returns:
            FIMSfig: matplotlib figure
        """
        # Extract relevant data from dictionary
        pitch = self.reconInfo['Pitch']
        timeRez = self.reconInfo['Time Resolution']
        transDif = self.reconInfo['Horizontal Diffusion']
        lonDif = self.reconInfo['Vertical Diffusion']
        
        # Apply Gaussian smear to approximate diffusion
        smearData = self.diffuseData(self.rawData, (transDif, transDif, lonDif))

        # Discretize data to approximate falling into grid holes
        discreteData = self.discretizeData(smearData, (pitch, pitch, 0)).copy()
        
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
        
        returns:
            BEASTfig: matplotlib figure
        """
        # Extract relevant data from dictionary
        pitch = self.reconInfo['Pitch']
        timeRez = self.reconInfo['Time Resolution']
        transDif = self.reconInfo['Horizontal Diffusion']
        lonDif = self.reconInfo['Vertical Diffusion']

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
            if len(newElectrons) > 0:
                #TODO: find better diffusion metric for diffusion incurred during avalanche
                newDiffused = self.diffuseData(newElectrons, (transDif/10, transDif/10, lonDif/10)).copy()
            # Append new electrons to data set
            avalData.extend(newDiffused)

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
        
        return BEASTfig

    #********************************************************************************#

