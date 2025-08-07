########################################
# CLASS DEFINITION FOR SIMULATION DATA #
########################################
import os
import uproot
import pandas as pd
import awkward_pandas
import math
import matplotlib.pyplot as plt
import numpy as np

from polyaClass import myPolya
from functionsFIMS import withinHex

CMTOMICRON = 1e4

class runData:
    """
    Class representing the data acquired from a single simulation.
    The data identifier is the simulation run number.

    The simulation data, originally saved in root format, 
    is read into pandas dataframes upon initialization.
    These dataframe names are listed in 'dataTrees'.

    Attributes:
        runNumber (int): The simulation run number
        dataTrees (list): List of strings containing the names 
                          of all of the dataframes.

        Internal Data Frames:
        *****
        NOTE: 
            These dataframes are intended to be internal to the class, and as such 
            should NOT be accessed directly. Instead utilize the getDataFrame() 
            method to retrieve a copy of the dataframe.
            This is to preserve data integrity.
        *****
            _metaData: Metadata information, including geometry parameters, 
                       simulation limits, git version, etc.
            _fieldLineData: Information for field lines generated at the cathode.
            _gridFieldLineData: Field line information for those generated 
                                above and below the grid.
            _electronData: Information for each individual simulated electron.
            _ionData: Information for each individual simulated ion.
            _avalancheData: Information for each simulated avalanche.
            _electronTrackData: Information for the full tracks of each electron.


    """

#********************************************************************************#
    # Initialize
    def __init__(self, runNumber):
        """
        Initializes a data class containing the data from the specified run number.
        """

        self.runNumber = runNumber

        self.dataTrees = [
            'metaData',
            'fieldLineData',
            'gridFieldLineData',
            'electronData',
            'ionData',
            'avalancheData',
            'electronTrackData'
        ]

        allData = self._readRootTrees()
        if not allData:
            print(f'Warning: No data loaded for run number {runNumber}. \
                    Check file path and contents.')

        for treeName in self.dataTrees:
            setattr(self, f'_{treeName}', allData.get(treeName))

#********************************************************************************#
    #String representation
    def __str__(self):
        """
        String representation of the runData class.
        """
        return f'Run number {self.runNumber}.'

#********************************************************************************#
    #Private function to read from data file
    def _readRootTrees(self):
        """
        Reads all trees from a ROOT file and returns them as a dictionary.
        The keys are the tree names and the values are pandas DataFrames.

        Returns:
            dict: A dictionary where keys are tree names (str) and
                values are pandas DataFrames. Returns an empty dictionary
                if the file cannot be opened or contains no trees.
        """

        dataFilePath = '../Data/'
        dataFile = f'sim.{self.runNumber}.root'
        fullFilePath = os.path.join(dataFilePath, dataFile)
        
        try:
            with uproot.open(fullFilePath) as rootFile:
                dataframes = {}
                for treeKey in rootFile.keys():
                    
                    #Strip the tree number ID and 'Tree' if present
                    strippedName = treeKey.split(';')[0]
                    if strippedName.endswith('Tree'):
                        treeName = strippedName.removesuffix('Tree')
                    else:
                        treeName = strippedName
                    
                    if isinstance(rootFile[treeKey], uproot.behaviors.TTree.TTree):
                        tree = rootFile[treeKey]
                        try:
                            df = tree.arrays(library='pd')
                            dataframes[treeName] = df
                        except Exception as e:
                            print(f"Error reading tree '{treeKey}': {e}")
                return dataframes
        except Exception as e:
            print(f"Error opening or reading ROOT file '{fullFilePath}': {e}")
            return {}
        
    
 #********************************************************************************#
    def _checkName(self, dataSetName):
        """
        Checks that a requested data frame name from the simulation trees is valid.
        
        Returns:
            bool: True if valid name of dataFrame, False otherwise
        """
        if dataSetName not in self.dataTrees:
            print(f"Invalid data set: '{dataSetName}'.")
            return False
        return True
        

 #********************************************************************************#
    def getTreeNames(self):
        """
        Returns a list of all of the available tree names.
        """
        return  self.dataTrees
    

 #********************************************************************************#
    def printMetaData(self):
        """
        Prints all metadata information. Dimensions are in microns.
        """
        for inParam in self.dataTrees['metaData'].columns:
            print(f'{inParam}: {self.getRunParameter(inParam)}')
        return


 #********************************************************************************#
    def printColumns(self, dataSetName):
        """
        Prints all of the column names for a given data set.
        """
        if not self._checkName(dataSetName):
            return
        
        dataFrame = self.getDataFrame(dataSetName)

        if dataFrame is not None:
            print(f'{dataSetName}:')
            print(dataFrame.columns.tolist())

        return


 #********************************************************************************#
    def getDataFrame(self, dataSetName):
        """
        Retrieves a specific DataFrame by its attribute name.

        All dimensions are given in microns.

        Args:
            dataSetName (str): The name of the DataFrame attribute.

        Returns:
            pd.DataFrame: The requested DataFrame if found and loaded, 
                          otherwise None.
        """
        if not self._checkName(dataSetName):
            return None

        #Get a copy of the data - note is saved as cm
        rawData = getattr(self, f'_{dataSetName}', None)
        dataFrame = rawData.copy()

        if dataFrame is None:
            print(f"Missing '{dataSetName}' data.")
            return None

        #Scale to micron
        match dataSetName:
            case 'fieldLineData' | 'gridFieldLineData':
                dataToScale = [
                    'Field Line x', 
                    'Field Line y', 
                    'Field Line z'
                ]

            case 'electronData' | 'ionData':
                dataToScale = [
                    'Initial x', 
                    'Initial y',
                    'Initial z',
                    'Final x', 
                    'Final y',
                    'Final z',
                ]
            
            case 'electronTrackData':
                dataToScale = [
                    'Drift x', 
                    'Drift y', 
                    'Drift z'
                ]

            case _:
                dataToScale = []

        CMTOMICRON = 1e4
        for toScale in dataToScale:
            dataFrame[toScale] *= CMTOMICRON
    
        return dataFrame
            
        
 #********************************************************************************#
    def getRunParameter(self, paramName):
        """
        Retrieves a given parameter from the metadata information.

        Dimensions are returned in microns.

        Args:
            paramName (str): Name of the parameter to be retrieved.

        Returns:
            Any: The value of the given parameter.
                 None if an error occurs.
        """
        dimensionalParams = [
            'Pad Length',
            'Pitch',
            'Grid Standoff',
            'Grid Thickness',
            'Hole Radius',
            'Cathode Height',
            'Thickness SiO2'
        ]
        
        metaData = self.getDataFrame('metaData')
        if metaData is None:
            print("Error: 'metaData' unavailable.")
            return None
        
        if paramName not in metaData.columns:
            print(f"Error: '{paramName}' not in 'metaData'.")
            return None
            
        CMTOMICRON = 1e4
        try:
            if paramName in dimensionalParams:
                return metaData[paramName].iloc[0]*CMTOMICRON
            else:
                return metaData[paramName].iloc[0] 
        
        except IndexError:
            print(f"Error: 'metaData' DataFrame is empty.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred retrieving '{paramName}': {e}")
            return None


 #********************************************************************************#   
    def plotCellGeometry(self):
        """
        Plots a top-down view of the simulation geometry.

        This includes the primary unit cell with the pad and hole in solid lines,
        and the surrounding cells as dotted.
        Additional information such as geometry cell, simulation boundary, and length
        definitions are included.
        
        TODO: The support pillars are also visualized, 
        however these are not yet included in simulations.
        """
        # Extract relevant geometric parameters from metadata.
        pitch = self.getRunParameter('Pitch')
        padLength = self.getRunParameter('Pad Length')
        holeRadius = self.getRunParameter('Hole Radius')

        inRadius = pitch/2.
        outRadius = 2*inRadius/math.sqrt(3)
        
        #Centers of neighbouring cells
        pitchX = pitch*2./math.sqrt(3)
        pitchY = inRadius
        neighbourX = 3./2.*outRadius*np.array([-1, -1, 0, 0, 1, 1])
        neighbourY = inRadius*np.array([1, -1, 2, -2, 1, -1])
        
        #Define the vertices of a hexagon
        hexCornerX = np.array([1., .5, -.5, -1., -.5, .5, 1.])
        hexCornerY = math.sqrt(3.)/2.*np.array([0., 1., 1., 0., -1., -1., 0.])

        # Corners of the pad
        padX = hexCornerX*padLength
        padY = hexCornerY*padLength

        # Corners of the cell
        cellX = hexCornerX*outRadius
        cellY = hexCornerY*outRadius

        # Define circles representing the holes in the grid.
        #    (Must be done as separate patches)
        hole = {}
        #Hole for primary cell
        hole[0] = plt.Circle((0, 0), holeRadius, 
                            facecolor='none', edgecolor='k', lw=1, 
                            label=f'Hole (r = {holeRadius:.0f} um)')
        # Neighbouring cell holes (#808080 = Grey)
        for i in range(len(neighbourX)):
            hole[i+1] = plt.Circle(
                (neighbourX[i], 
                 neighbourY[i]), 
                 holeRadius, 
                 facecolor='none', edgecolor='#808080', ls=':', lw=1)

        # Define circles representing the pillars.
        #    (Must be done as separate patches)
        pillarRadius = 10
        pillar = {}
        for i in range(6):
            pillar[i] = plt.Circle(
                (outRadius*hexCornerX[i], 
                 outRadius*hexCornerY[i]), 
                 pillarRadius,
                 facecolor='none', edgecolor='#808080', ls='--', lw=1)

        pillar[0].set_label(f'Pillar  (r = {pillarRadius:.0f} um)')

        # Make figure and add plots
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(f'Cell Geometry')
        ax1 = fig.add_subplot(111)

        #Add the pad
        ax1.plot(padX, padY, 
                label='Pad', c='m', lw=1)

        #Add the cell boundary
        ax1.plot(cellX, cellY, 
                label='Cell Boundary', c='b', lw=1)
        
        #Add boundaries of neighboring cells and pads
        for i in range(6):
            ax1.plot(neighbourX[i]+cellX, neighbourY[i]+cellY,
                    c='c', ls=':', lw=1)
            ax1.plot(neighbourX[i]+padX, neighbourY[i]+padY,
                    c='m', ls=':', lw=1)
        
        #Add the holes in the grid
        for i in range(len(hole)):
            ax1.add_patch(hole[i])

        #Add pillars
        for i in range(len(pillar)):
            ax1.add_patch(pillar[i])

        #Add markers to center of cells
        ax1.plot(0., 0., marker='x', c='r')
        ax1.plot(neighbourX, neighbourY,
                marker='.', c='r', ls='')

        #Add geometry cell
        geoX = 3./2.*outRadius*np.array([0, 1, 1, 0, 0])
        geoY = inRadius*np.array([1, 1, 0, 0, 1])
        ax1.plot(geoX, geoY,
                c='g', ls=':', lw=1, label='Geometry Boundary')

        #Add simulation boundary
        simX = 3./2.*outRadius*np.array([-1, 1, 1, -1, -1])
        simY = inRadius*np.array([1, 1, -1, -1, 1])
        ax1.plot(simX, simY,
                c='g', ls='--', lw=1, label='Simulation Boundary')

        #Add dimensions
        ax1.plot([0, neighbourX[4]], [0, neighbourY[4]],
                label=f'Pitch ({pitch:.0f} um)', c='r', ls=':', lw=1)
        ax1.plot([0, padLength], [0, 0],
                label=f'Pad Length ({padLength:.0f} um)', c='r', ls='-', lw=1)
        
        # Set other plot elements
        ax1.grid()
        axLim = 1.1*pitch
        ax1.set_xlim(-axLim, axLim)
        ax1.set_ylim(-axLim, axLim)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x (um)')
        ax1.set_ylabel('y (um)')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()   
        plt.show()
        
        return
    
 #********************************************************************************#   
    def _plotAddCellGeometry(self, axis, axes):
        """
        Adds geometry elements to 2D plots, including
        the pad, hole in grid, and outline of the unit cell.

        Args:
            axis (matplotlib subplot): Axes object where geometry is to be added 
            axes (str): String defining the catesian dimensions of 'axes'.
        """
        # Extract relevant geometric parameters from metadata.
        pitch = self.getRunParameter('Pitch')
        padLength = self.getRunParameter('Pad Length')
        gridThickness = self.getRunParameter('Grid Thickness')
        gridStandoff = self.getRunParameter('Grid Standoff')
        cathodeDist = self.getRunParameter('Cathode Height')
        holeRadius = self.getRunParameter('Hole Radius')

        #calculated values
        inRadius = pitch/2.
        outRadius = 2*inRadius/math.sqrt(3)
        halfGrid = gridThickness/2.
        padHeight = - halfGrid - gridStandoff
        cathodeHeight = cathodeDist + halfGrid

        #Define the vertices of a hexagon
        hexCornerX = np.array([1., .5, -.5, -1., -.5, .5, 1.])
        hexCornerY = math.sqrt(3.)/2.*np.array([0., 1., 1., 0., -1., -1., 0.])
        
        # Corners of the pad
        padX = hexCornerX*padLength
        padY = hexCornerY*padLength
        padZ = np.ones(len(padX))*padHeight

        # Corners of the cell
        cellX = hexCornerX*outRadius
        cellY = hexCornerY*outRadius

        # Define the hole in the grid.
        hole = plt.Circle((0, 0), holeRadius,
                        facecolor='none', edgecolor='k', lw=1, label='Hole')
        holeXY = holeRadius*np.array([-1, -1, 1, 1, -1])
        holeZ = halfGrid*np.array([1, 1, -1, -1, 1])

        #Corners of the geometry cell
        geoX = 3./2.*outRadius*np.array([-1, 1, 1, -1, -1])
        geoY = inRadius*np.array([1, 1, -1, -1, 1])

        axLim = 1.01*3./2.*outRadius
        
        match axes:
            case 'xy':
                axis.plot(padX, padY, 
                        label='Pad', c='m', lw=1)
                axis.plot(cellX, cellY, 
                        label='Cell', c='c', lw=1)
                axis.add_patch(hole)
                axis.plot(geoX, geoY,
                        c='g', ls='--', lw=1, label='Simulation Boundary')

                axis.set_xlabel('x (um)')
                axis.set_ylabel('y (um)')
                axis.set_xlim(-axLim, axLim)
                axis.set_ylim(-axLim, axLim)
                axis.set_aspect('equal')

            case 'xz':
                axis.plot(padX, padZ, 
                        label='Pad', c='m', lw=2)
                axis.plot([cellX[3], cellX[0], cellX[0], cellX[3], cellX[3]], 
                        [padHeight, padHeight, cathodeHeight, cathodeHeight, padHeight],
                        label='Cell', c='c', lw=1)
                axis.plot([cellX[2], cellX[1], cellX[1], cellX[2], cellX[2]], 
                        [padHeight, padHeight, cathodeHeight, cathodeHeight, padHeight], 
                        c='c', ls='--', lw=1)
                axis.plot(holeXY, holeZ, 
                        label='Hole', c='k', ls='-')
                axis.plot([geoX[0], geoX[1], geoX[1], geoX[0], geoX[0]], 
                        [padHeight, padHeight, cathodeHeight, cathodeHeight, padHeight], 
                        c='g', ls='--', lw=1, label='Simulation Boundary')
                axis.plot([geoX[0], geoX[1]],
                         [0, 0],
                         c='#808080', ls=':', lw=1, label='Grid')
                
                axis.set_xlabel('x (um)')
                axis.set_ylabel('z (um)')
                axis.set_xlim(-axLim, axLim)
                axis.set_ylim(padHeight, cathodeHeight)

            case 'yz':
                axis.plot(padY, padZ, 
                        label='Pad', c='m', lw=2)
                axis.plot([cellY[4], cellY[1], cellY[1], cellY[4], cellY[4]], 
                        [padHeight, padHeight, cathodeHeight, cathodeHeight, padHeight],
                        label='Cell', c='c', lw=1)
                axis.plot([0, 0], 
                        [padHeight, cathodeHeight],
                        label='Cell', c='c', ls='--', lw=1)
                axis.plot(holeXY, holeZ, 
                        label='Hole', c='k', ls='-')
                axis.plot([geoY[1], geoY[2], geoY[2], geoY[1], geoY[1]], 
                        [padHeight, padHeight, cathodeHeight, cathodeHeight, padHeight], 
                        c='g', ls='--', lw=1, label='Simulation Boundary')
                axis.plot([geoY[0], geoY[2]],
                         [0, 0],
                         c='#808080', ls=':', lw=1, label='Grid')

                axis.set_xlabel('y (um)')
                axis.set_ylabel('z (um)')
                axis.set_xlim(-axLim, axLim)
                axis.set_ylim(padHeight, cathodeHeight)
                

            case _:
                print(f'Invalid form of plot axes: {axes}')

        axis.grid()

        return


#********************************************************************************#   
    def plot2DFieldLines(self, target):
        """
        Generates 2D plots of the simulated field lines. Options include:
            Cathode - Field lines initiated near the cathode.
            AboveGrid - Field lines initiated just above the grid.
            BelowGrid - Field lines initiated just below the grid.

        These plots display the field lines, and include 
        the pad geometry and hole in the grid.
    
        Args:
            target (str): Identifier for what data to plot.
    
        Returns:
            None
        """
        plotOptions = [
            'Cathode',
            'AboveGrid',
            'BelowGrid'
        ]
    
        match target:
            case 'Cathode':
                fieldLineData = self.getDataFrame('fieldLineData')
                
            case 'AboveGrid':
                fieldLineData = self.getDataFrame('gridFieldLineData')
                fieldLineData = fieldLineData[fieldLineData['Grid Line Location']==1]
                
            case 'BelowGrid':
                fieldLineData = self.getDataFrame('gridFieldLineData')
                fieldLineData = fieldLineData[fieldLineData['Grid Line Location']==-1]
    
            case _:
                print(f'Error: Plot options are: {plotOptions}')
                return
    
        groupedData = fieldLineData.groupby('Field Line ID')
    
        if groupedData is None:
            print(f"An error occured plotting '{target}'.")
            return

        # Create figure and subplots for different projections.
        fig2D = plt.figure(figsize=(14, 7))
        fig2D.suptitle(f'Field Lines - {target}')
        ax11 = fig2D.add_subplot(221)
        ax12 = fig2D.add_subplot(223)
        ax13 = fig2D.add_subplot(122)
    
        # iterate through all lines
        for lineID, fieldLine in groupedData:
            ax11.plot(fieldLine['Field Line x'], 
                      fieldLine['Field Line z'], 
                      lw=1)
            ax12.plot(fieldLine['Field Line y'], 
                      fieldLine['Field Line z'], 
                      lw=1)
            ax13.plot(fieldLine['Field Line x'], 
                      fieldLine['Field Line y'], 
                      lw=1)
    
        self._plotAddCellGeometry(ax11, 'xz')
        self._plotAddCellGeometry(ax12, 'yz')
        self._plotAddCellGeometry(ax13, 'xy')
        plt.tight_layout()   
        plt.show()
        
        return


 #********************************************************************************#   
    def _getRawGain(self):
        """
        Returns the mean size of the simulated avalanches.

        This includes any avalanches that hit the simulation limit, and those where
        some electrons have exitted the simulation boundary.
        """
        avalancheData = self.getDataFrame('avalancheData')
        return avalancheData['Total Electrons'].mean()

    
 #********************************************************************************#   
    def _trimAvalanche(self):
        """
        Removes any avalanches that have either:
            Only a single electron (so no avalanching occured), or
            that reached the simulation avalanche limit.

        Note that these situations, either no avalanching or an avalanche
        that was exactly the limit size, can occur. 
        However, it is much more likely that the intial electron attached or drifted
        outside of the simulation bounds before causing an avalanche for the e=1 case. 
        For the avalanche-limit case, it is impossible to tell if this was exacly the
        limit, or if there should be more electrons.

        Returns:
            dataframe: The avalancheData dataframe with the 1 and limitting sizes removed.
        """
        avalancheData = self.getDataFrame('avalancheData')

        trimmedAvalanche = avalancheData[
            ((avalancheData['Total Electrons'] > 1) 
          &  (avalancheData['Reached Limit'] == 0))
        ]
        return trimmedAvalanche

    
 #********************************************************************************#   
    def _histAvalanche(self, trim, binWidth):
        """
        Calculates a histogram of the avalanche electron count data.

        Can optionally trim the dataset to remove avalanches that either have
        a single electron or those that reached the limit.

        Args:
            trim (bool): If True, will remove the edge-case avalanches.
            binWidth (int): The width of the bins to be used for the histogram.

        Returns:
            dict: Dictionary containing the histogram data and parameters:
                - 'binCenters' (ndarray): The center of each histogram bin.
                - 'gain' (float): The mean value of the total electrons.
                - 'counts' (ndarray): The number of data points in each bin.
                - 'countErr' (ndarray): The error in the counts, using Poisson stats.
                - 'prob' (ndarray): The probability density for each bin.
                - 'probErr' (ndarray): The error in the probability density.
                - 'binWidth' (float): The width of the histogram bin.
                - 'trim' (bool): Indicates if the data was trimmed.
        """

        if trim:
            data = self._trimAvalanche()
        else:
            data = self.getDataFrame('avalancheData')

        gain = data['Total Electrons'].mean()
            
        bins = np.arange(
            data['Total Electrons'].min(), 
            data['Total Electrons'].max()+1, 
            binWidth
        )
        binCenters = bins[:-1] + binWidth/2.
        
        counts, _ = np.histogram(
            data['Total Electrons'], 
            bins=bins
        )

        prob = counts/len(data['Total Electrons'])/binWidth

        # Get errors
        countErr = np.where(counts == 0, 1, np.sqrt(counts))
        probErr = countErr/len(data['Total Electrons'])/binWidth

        histData = {
            'binCenters': binCenters,
            'gain': gain,
            'counts': counts,
            'countErr': countErr,
            'prob': prob,
            'probErr': probErr,
            'binWidth': binWidth,
            'trim': trim
        }

        return histData


 #********************************************************************************#   
    def plotAvalancheSize(self, trim=False, binWidth=1):
        """
        Plots a histogram of the simulated avalanche size distribution.

        Shows the probability of an avalanche having a certain size.

        Args:
            trim (bool): If True, the avalanche data is trimmed before plotting.
            binWidth (int): The width of the histogram bins. 
        """
        histData = self._histAvalanche(trim, binWidth)
        
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle(f'Avalanche Size Distribution: Run {self.runNumber}')
        
        ax = fig.add_subplot(111)
        
        ax.bar(
            histData['binCenters'], 
            histData['prob'], 
            width=binWidth,
            label='Simulation'
        )  
        ax.axvline(
            histData['gain'], 
            c='g', ls='--', label=f'Gain = {histData['gain']:.0f}'
        )
        
        ax.set_xlabel('Number of Electrons in Avalanche')
        ax.set_ylabel('Probability of Avalanche Size')
        ax.legend()
        ax.grid()
        plt.tight_layout()   
        plt.show()
        
        return 

 #********************************************************************************#   
    def plotAvalanche2D(self, avalancheID=0, plotName=''):
        """
        Generates 2D plots of a single electron avalanche.

        Includes individual electron tracks and geometry components.
    
        Args:
            avalancheID (int): Index of avalanche within simulation.
    
        """
    
        allData = self.getDataFrame('electronTrackData')
        singleData = allData[allData['Avalanche ID']==avalancheID]
        
        gain = singleData['Electron ID'].max()+1
    
        groupedData = singleData.groupby('Electron ID')
    
        if groupedData is None:
            print(f"An error occured plotting ID='{avalancheID}'.")
            return

        # Create figure and subplots for different projections.
        fig2D = plt.figure(figsize=(14, 7))
        fig2D.suptitle(f'Avalanche: {plotName} (ID #{avalancheID}) Gain = {gain}')
        ax11 = fig2D.add_subplot(221)
        ax12 = fig2D.add_subplot(223)
        ax13 = fig2D.add_subplot(122)
    
        # iterate through all lines
        for electronID, driftLine in groupedData:
            ax11.plot(driftLine['Drift x'], 
                      driftLine['Drift z'], 
                      lw=.5)
            ax12.plot(driftLine['Drift y'], 
                      driftLine['Drift z'], 
                      lw=.5)
            ax13.plot(driftLine['Drift x'], 
                      driftLine['Drift y'], 
                      lw=.5)

        # Plot the initial electron location
        xInit = singleData['Drift x'].iloc[0]
        yInit = singleData['Drift y'].iloc[0]
        zInit = singleData['Drift z'].iloc[0]
        
        ax11.plot(xInit, 
                  zInit, 
                  label='Initial', marker='x', c='r')
        ax12.plot(yInit, 
                  zInit, 
                  label='Initial', marker='x', c='r')
        ax13.plot(xInit, 
                  yInit, 
                  label='Initial', marker='x', c='r')
    
        self._plotAddCellGeometry(ax11, 'xz')
        self._plotAddCellGeometry(ax12, 'yz')
        self._plotAddCellGeometry(ax13, 'xy')
        plt.tight_layout()   
        plt.show()
        
        return
        

 #********************************************************************************#   
    def plotDiffusion(self, target):
        """
        Plots the diffusion of simulated particles in the drift direction (z) and 
        the radial distance within the normal plane (xy).

        Options to plot include: electrons, positive ions, and negative ions.
        Some geometric distances are included.

        Args:
            target (str): The type of particle to plot.
        """
        plotOptions = [
            'electron',
            'posIon',
            'negIon'
        ]
        match target:
            case 'electron':
                particleData = self.getDataFrame('electronData')

            case 'posIon':
                particleData = self.getDataFrame('ionData')
                particleData = particleData[particleData['Ion Charge']==1]
                
            case 'negIon':
                particleData = self.getDataFrame('ionData')
                particleData = particleData[particleData['Ion Charge']==-1]
                
            case _:
                print(f'Error: Plot options are: {plotOptions}')
                return
    
        if particleData is None:
            print(f"An error occured plotting '{target}'.")
            return

        driftX = particleData['Final x'] - particleData['Initial x']
        driftY = particleData['Final y'] - particleData['Initial y']
        driftZ = particleData['Final z'] - particleData['Initial z']

        driftR = np.sqrt(driftX**2 + driftY**2)

        fig = plt.figure(figsize=(12, 4))
        fig.suptitle(f'Total Drift of: {target}s ({self.getRunParameter('Number of Avalanches')} Avalanches)')
        
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.hist(abs(driftZ))
        ax2.hist(driftR)

        ax1.axvline(
            abs(driftZ.iloc[0]), 
            c='r', ls=':', label='Initial Electron'
        )
        ax1.axvline(
            self.getRunParameter('Grid Standoff'), 
            c='g', ls='--', label='Grid Height'
        )

        ax2.axvline(
            driftR.iloc[0], 
            c='r', ls=':', label='Initial Electron'
        )
        ax2.axvline(
            self.getRunParameter('Hole Radius'), 
            c='g', ls='--', label='Hole Radius'
        )
        ax2.axvline(
            self.getRunParameter('Pad Length'), 
            c='m', ls='--', label='Pad Length'
        )
    
        ax1.set_title('Drift in z')
        ax2.set_title('Drift in xy Plane')
    
        ax1.set_xlabel('Drift in z (um)')
        ax2.set_xlabel('Drift in r (um)')

        ax1.legend()
        ax2.legend()
        ax1.grid()
        ax2.grid()
        plt.tight_layout()   
        plt.show()
        
        return
        
 #********************************************************************************#   
    def plotParticleHeatmaps(self, target, numBins=51):
        """
        Plots 2D histograms displaying heatmaps of the intial and final locations 
        of simulated particles.

        Options to plot include: electrons, positive ions, and negative ions.
        Geometry features such as pad and hole are included.

        Args:
            target (str): The type of particle to plot.
            numBins (int): Number of bins for each dimension.
        """
        plotOptions = [
            'electron',
            'posIon',
            'negIon'
        ]
        match target:
            case 'electron':
                particleData = self.getDataFrame('electronData')

            case 'posIon':
                particleData = self.getDataFrame('ionData')
                particleData = particleData[particleData['Ion Charge']==1]
                
            case 'negIon':
                particleData = self.getDataFrame('ionData')
                particleData = particleData[particleData['Ion Charge']==-1]
                
            case _:
                print(f'Error: Plot options are: {plotOptions}')
                return
    
        if particleData is None:
            print(f"An error occured plotting '{target}'.")
            return

        # Create the figure and add subplots
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f'Particle Heatmaps: {target}s')
        
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
    
        # Plot data
        ax1.hist2d(particleData['Initial x'], 
                   particleData['Initial y'], 
                   bins=numBins, cmin=1)
        ax2.hist2d(particleData['Final x'], 
                   particleData['Final y'], 
                   bins=numBins, cmin=1)
        ax3.hist2d(particleData['Initial x'], 
                   particleData['Initial z'], 
                   bins=numBins, cmin=1)
        ax4.hist2d(particleData['Final x'], 
                   particleData['Final z'], 
                   bins=numBins, cmin=1)
    
        #Add geometry Pieces
        self._plotAddCellGeometry(ax1, 'xy')
        self._plotAddCellGeometry(ax2, 'xy')
        self._plotAddCellGeometry(ax3, 'xz')
        self._plotAddCellGeometry(ax4, 'xz')
    
        ax1.set_title('Initial Position')
        ax2.set_title('Final Position')
        ax3.set_title('Initial Position')
        ax4.set_title('Final Position')

        plt.tight_layout()   
        plt.show()
    
        return      

 #********************************************************************************#   
    def _fitAvalancheSize(self, binWidth):
        """
        Fits the trimmed simulated avalanche size distribution 
        to Polya and exponential curves.

        Args:
            binWidth (float): The width of the histogram bins used for the fitting.

        Returns:
            dict: A dictionary containing the fitting results. Includes:
                - 'xVal' (ndarray): The bin centers of the histogram.
                - 'yVal' (ndarray): The probability densities of the histogram.
                - 'dataGain' (float): The calculated mean gain from the data.
                - 'fitPolya' (myPolya object): An object containing the results 
                                               of the Polya distribution fit.
                - 'fitExpo' (myPolya object): An object containing the results 
                                              of the exponential distribution fit.

        """
        histData = self._histAvalanche(trim=True, binWidth=binWidth)

        gain = histData['gain']

        if gain < 5 or gain >= self.getRunParameter('Avalanche Limit'):
            raise ValueError(f'Unable to fit to data. Gain is {gain:.2f}.')

        
        fitDataToPolya = myPolya()
        fitDataToPolya.fitPolya(
            histData['binCenters'],
            histData['prob'],
            histData['gain'],
            histData['probErr'] 
        )
        
        fitDataToExpo = myPolya()
        fitDataToExpo.fitPolya(
            histData['binCenters'],
            histData['prob'],
            histData['gain'],
            histData['probErr'], 
            expo = True
        )
        
        fitResults = {
            'xVal': histData['binCenters'],
            'yVal': histData['prob'],
            'dataGain': histData['gain'],
            'fitPolya': fitDataToPolya,
            'fitExpo': fitDataToExpo,
        }
        
        return fitResults

    
 #********************************************************************************#   
    def plotAvalancheFits(self, binWidth=1):
        """
        Plots a histogram of the simulated avalanche size distribution.
        Includes results of Polya and Exponential fits.

        The raw, trimmed, and fitted gains are all indicated.

        Args:
            binWidth (int): The width of the histogram bins. 
        """
        fitResults = self._fitAvalancheSize(binWidth)

        polyaResults = fitResults['fitPolya'].calcPolya(fitResults['xVal'])
        expoResults = fitResults['fitExpo'].calcPolya(fitResults['xVal'])
        
        fig = plt.figure(figsize=(12, 4))
        fig.suptitle(f'Avalanche Size Distribution: Run {self.runNumber}')
        
        ax = fig.add_subplot(111)

        ax.bar(
            fitResults['xVal'], 
            fitResults['yVal'],
            width=binWidth,
            label='Simulation'
        ) 

        ax.plot(fitResults['xVal'], 
                polyaResults, 
                'm-', lw=2, 
                label=r'Fitted Polya ($\theta$' 
                    + f' = {fitResults['fitPolya'].theta:.3})')
        ax.axvline(x=fitResults['fitPolya'].gain, 
               c='m', ls=':', label=f'Polya Gain = {fitResults['fitPolya'].gain:.1f}e')
        
        ax.plot(fitResults['xVal'], 
                expoResults, 
                'r', lw=2, label=f'Fitted Exponential')
        ax.axvline(x=fitResults['fitExpo'].gain, 
               c='r', ls=':', label=f'Expo Gain = {fitResults['fitExpo'].gain:.1f}e')

        ax.axvline(x=self._getRawGain(), 
               c='g', ls='--', label=f'Raw Gain = {self._getRawGain():.1f}e')
        ax.axvline(x=fitResults['dataGain'], 
               c='g', ls=':', label=f'Trimmed Gain = {fitResults['dataGain']:.1f}e')


        plt.xlabel('Numer of Electrons in Trimmed Avalanche')
        plt.ylabel('Probability of Avalanche Size')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()

        return


 #********************************************************************************#
    def calcBundleRadius(self, zTarget=0):
        """
        Calculates the radius of the outermost electric field line
        at a specified z-coordinate.

        "Outermost Line" - The line with the largest radius at the cathode that
        initiates within the unit cell.
        Does a linear interpolation between available datapoints.

        Args:
            zTarget (float): z-coordinate of desired field bundle radius.

        Returns:
            float: The radius of the outermost field line in 
                the bundle at the specified z coordinate.
        """
        zMax = self.getRunParameter('Cathode Height')
        zMin = -1.*self.getRunParameter('Grid Standoff')
        if not (zMin < zTarget < zMax):
            raise ValueError('Invalid target z.')
        
        allFieldLines = self.getDataFrame('fieldLineData')
        pitch = self.getRunParameter('Pitch')
        unitCellLength = pitch/math.sqrt(3)

        #All lines start at same z near cathode
        initialZ = allFieldLines['Field Line z'].iloc[0]

        #Determine the radius for all field lines
        allFieldLines['Field Line Radius'] = np.sqrt(
            allFieldLines['Field Line x']**2 + 
            allFieldLines['Field Line y']**2
        )

        #Isolate the largest radius at the cathode
        atCathode = allFieldLines[allFieldLines['Field Line z'] == initialZ]
        #Determine what lines initiate within the unit cell
        withinUnitCell = withinHex(
            atCathode['Field Line x'], 
            atCathode['Field Line y'], 
            unitCellLength
            )
        cellLines = atCathode[withinUnitCell]
        #Find line with max radius
        maxRadius = cellLines['Field Line Radius'].max()
        outermostLine = cellLines[cellLines['Field Line Radius'] == maxRadius]
        lineID = outermostLine['Field Line ID'].iloc[0]

        #get entire outermost field line - ensure sorted for interpolation
        targetLine = allFieldLines[allFieldLines['Field Line ID'] == lineID]
        targetLine = targetLine.sort_values(by='Field Line z')

        #Get radius for target z using linear interpolation
        targetRadius = np.interp(
            zTarget, 
            targetLine['Field Line z'],
            targetLine['Field Line Radius']
        )

        return targetRadius





