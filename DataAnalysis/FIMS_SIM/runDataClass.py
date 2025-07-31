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

CMTOMICRON = 1e4

class runData:
    """
    """


    # Initialize
    def __init__(self, runNumber):
        """
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

        allData = self._readRootTrees(runNumber)
        if not allData:
            print(f'Warning: No data loaded for run number {runNumber}. \
                    Check file path and contents.')

        for treeName in self.dataTrees:
            setattr(self, treeName, allData.get(treeName))


    #String representation
    def __str__(self):
        """
        String representation of the runData class.
        """
        return f'Run number {self.runNumber}.'


    #Private function to read from data file
    def _readRootTrees(self, runNumber):
        """
        Reads all trees from a ROOT file and returns them as a dictionary.
        The keys are the tree names and the values are pandas DataFrames.

        Args:
            runNumber (int): The run number of the ROOT file.

        Returns:
            dict: A dictionary where keys are tree names (str) and
                values are pandas DataFrames. Returns an empty dictionary
                if the file cannot be opened or contains no trees.
        """

        dataFilePath = '../../DataGeneration/FIMS_SIM/Data/'
        dataFile = f'sim.{runNumber}.root'
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
        
    
    def _checkName(self, dataSetName):
        """
        """
        if dataSetName not in self.dataTrees:
            print(f"Invalid data set: '{dataSetName}'.")
            return False
        return True
        

    def getTreeNames(self):
        """
        Returns a list of all of the available tree names.
        """
        return  self.dataTrees
    

    def printMetaData(self):
        """
        Prints all metadata information.
        """
        print(self.metaData.iloc[0])
        return
    

    def printColumns(self, dataSetName):
        """
        Prints the column names for a given data set.
        """
        if not self._checkName(dataSetName):
            return
        
        dataFrame = getattr(self, dataSetName, None)

        if dataFrame is None:
            print(f"Missing '{dataSetName}' data.")
        else:
            print(dataFrame.columns.tolist())
        return


    def getDataFrame(self, dataSetName):
            """
            Retrieves a specific DataFrame by its attribute name.

            All dimensions are given in microns.

            Args:
                dataSetName (str): The name of the DataFrame attribute.

            Returns:
                pd.DataFrame: The requested DataFrame if found and loaded, otherwise None.
            """
            if not self._checkName(dataSetName):
                return None

            #Get a copy of the data - note is saved as cm
            rawData = getattr(self, dataSetName, None)
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
            

    def getRunParameter(self, paramName):
        """
        Retrievs a given parameter from the metadata information in micrometers.

        Note: Data is defined in runControl as microns, but Garfield's naitive
              dimension is cm, so avalanche.cc converts and saves as this.

        Args:
            paramName (str): Name of the parameter to be retrieved.

        Returns:
            Any: The value of the given parameter in microns.
                 None if an error occurs.
        """
        if self.metaData is None:
            print("Error: 'metaData' unavailable.")
            return None
        
        if paramName not in self.metaData.columns:
            print(f"Error: '{paramName}' not in 'metaData'.")
            return None
            
        CMTOMICRON = 1e4
        try:
            return self.metaData[paramName].iloc[0]*CMTOMICRON
        
        except IndexError:
            print(f"Error: 'metaData' DataFrame is empty.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred retrieving '{paramName}': {e}")
            return None


    def plotCellGeometry(self):
        """
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
        pillarRadius = 5
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
    
    def _plotAddCellGeometry(self, axis, axes):
        """
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

        axLim = 1.1*pitch
        
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
            figure: The resulting plots if successful, None otherwise.
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
                return None
    
        groupedData = fieldLineData.groupby('Field Line ID')
    
        if groupedData is None:
            print(f"An error occured plotting '{target}'.")
            return None

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
        
        return fig2D


