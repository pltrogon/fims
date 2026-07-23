#################################
# CLASS DEFINITION FOR GEOMETRY #
#################################
from __future__ import annotations
import time
import subprocess
from venv import create

import numpy as np
import os
import math
import gmsh

class geometryClass:
    """
    Class to handle the geometry for the FIMS simulation.
     
    Generates a geometry using Gmsh.
    Solves the electric fields using Elmer.

    Dedicated classes are utilized for these tasks.
        These are: gmshClass and elmerClass.

    The geometry is defined by the following parameters (in microns):
        pitch: The distance between the centers of adjacent holes.
        holeRadius: The radius of the holes in the grid.
        padLength: The length of an outside edge of the pad.
        gridStandoff: The distance from the grid to the cathode.
        cathodeHeight: The distance from the grid to the cathode.
        gridThickness: The thickness of the grid.
        thicknessSiO2: The thickness of the SiO2 layer on the grid.
        pillarRadius: The radius of the pillars supporting the grid.

    The Fields are defined by the following parameters (in V/cm):
        driftField: The electric field in the drift region.
        fieldRatio: The ratio of the electric fields.

    Output files are saved in /Geometry/ and /elmerResults/ directories.
    """

#**********************************************************************#

    def __init__(self, inputParam=None):

        self._param = inputParam
        self._checkParameters()

        self._runGUI = False
        
        self._geoConfig = {
            'unitCell': 'Hexagonal',
            'holeShape': 'circle',
            'padShape': 'hexagon',
            'surrounding': False 
        }
        
        return

#**********************************************************************#

    def _checkParameters(self):
        """
        Checks that the parameters are valid
        for creating the geometry.
        """

        neededParameters = [
            'pitch',
            'holeRadius',
            'padLength',
            'gridStandoff',
            'cathodeHeight',
            'gridThickness',
            'thicknessSiO2',
            'pillarRadius',
            'driftField',
            'fieldRatio'
        ]

        if self._param is None:
            raise ValueError('Error - Invalid parameters.')

        for key in neededParameters:
            if key not in self._param:
                raise ValueError(f"Error - Missing '{key}'")
            if self._param[key] <= 0:
                raise ValueError(f"Error - '{key}' must be positive.")

        # Check geometric constraints
        outRadius = self._param['pitch'] / math.sqrt(3)
        inRadius = self._param['pitch']/2

        if self._param['holeRadius'] >= inRadius:
            raise ValueError('Error - Hole larger than Cell.')
        if self._param['padLength'] >= outRadius:
            raise ValueError('Error - Pad larger than Cell.')
        
        ## Pillars are currently not included in the geometry
        # Check that pillars can fit in the remaining space
        padInRadius = self._param['padLength']*math.sqrt(3)/2
        padSpace = inRadius - padInRadius
        #if self._param['pillarRadius'] >= padSpace:
        #    raise ValueError('Error - Pillar cannot fit.')

        return

#**********************************************************************#

    def _checkGeometryOptions(self, geoConfiguration):
        """
        Checks that the geometry options are valid.
        """
        geometryKeys = [
            'unitCell',
            'surrounding',
            'holeShape',
            'padShape'
        ]
        unitCellOptions = ['Square', 'Hexagonal',]
        holeOptions = ['circle', 'hexagon', 'octagon']
        padOptions = ['square', 'hexagon', 'octagon']
        
        for key in geometryKeys:
            if key not in geoConfiguration:
                raise ValueError(f"Error - Missing '{key}'")
        
        if geoConfiguration['unitCell'] not in unitCellOptions:
            raise ValueError(f'Unit cell must be one of {uniCellOptions}.')
        
        if geoConfiguration['holeShape'] not in holeOptions:
            raise ValueError(f'Hole shape must be one of {holeOptions}.')
        
        if geoConfiguration['padShape'] not in padOptions:
            raise ValueError(f'Pad shape must be one of {padOptions}.')
        
        if not isinstance(geoConfiguration['surrounding'], bool):
            raise ValueError('"surrounding" option must be type "bool".')
        
        return

#**********************************************************************#

    def setGUI(self, runGUI=True):
        """
        Sets whether the Gmsh GUI runs when creating the geometry.
        """
        self._runGUI = runGUI

        return
    
#**********************************************************************#

    def setGeometryConfiguration(self, geoConfig):
        """
        Sets the configuration of the geometry.
        
        args: 
            geoConfig (dict): dictionary with the following optional parameters
                surrounding (bool): whether to include surrounding unit cells.
                holeShape (str): shape of the amplification grid holes.
                padShape (str): shape of the readout pad.
                unitCell (str): shape of the unit cell.
        """
        self._checkGeometryOptions(geoConfig)
        self._geoConfig = geoConfig
        
        return

#**********************************************************************#

    def buildGeometry(self):
        """
        Builds the geometry for the FIMS simulation using Gmsh.
        """

        print('\tBuilding geometry...')
    
        self._checkParameters()
        self._gmshClass = gmshClass(self._param)
        self._gmshClass.generateMesh(
            geoConfig=self._geoConfig,
            runGUI=self._runGUI,
        )

        return

#**********************************************************************#

    def _generateElmerFiles(self, capacitance=False):
        """
        Generates the SIF files for Elmer based on the created geometry.
        """
        runOption = self._geoConfig['unitCell']
        if self._geoConfig['surrounding']:
            runOption += 'Surrounding'
        
        self._elmerClass = elmerClass(
            runOption, capacitance=False
        )

        if capacitance:
            self._elmerClassCapacitance = elmerClass(
                runOption, capacitance=True
            )

        return
    
#**********************************************************************#

    def calculateEFields(self, solveWeighting=True, capacitance=False):

        self._generateElmerFiles(capacitance=capacitance)

        if capacitance:
            self._elmerClassCapacitance.runElmer()
        else:
            self._setVoltages()
            self._elmerClass.runElmer(solveWeighting=solveWeighting)

        return

#**********************************************************************#

    def _setVoltages(self):
        """
        Sets the voltages on the grid and cathode electrodes.
        """

        gridVoltage, cathodeVoltage = self.findPotentials()

        self._elmerClass.resetPotentials()
        self._elmerClass._setPotential('Grid', gridVoltage)
        self._elmerClass._setPotential('Cathode', cathodeVoltage)

        return
        
#**********************************************************************#

    def findPotentials(self):
        """
        Calculates the grid and cathode potentials to achieve the 
        desired electric fields in the drift and amplification regions.

        Note these are negative.

        Returns:
            gridVoltage (float): The voltage to apply to the grid.
            cathodeVoltage (float): The voltage to apply to the cathode.
        """

        MICRONTOCM = 1e-4
        driftField = self._param['driftField']
        fieldRatio = self._param['fieldRatio']

        amplificationField = driftField*fieldRatio

        halfGrid = self._param['gridThickness']/2

        driftGap = self._param['cathodeHeight']-halfGrid
        amplificationGap = self._param['gridStandoff']-halfGrid

        gridVoltage = -1*amplificationField*amplificationGap*MICRONTOCM
        cathodeVoltage = -1*driftField*driftGap*MICRONTOCM + gridVoltage

        return gridVoltage, cathodeVoltage

#**********************************************************************#
#**********************************************************************#
#**********************************************************************#

class gmshClass:
    """
    Class to handle the geometry creation using Gmsh.
    """

#**********************************************************************#
    def __init__(self, inputParams=None):
        """
        Initializes the gmshClass instance with the given parameters.

        Args:
            inputParams (dict): A dictionary containing the parameters 
                                defining the geometry.
        """
    
        self._occ = gmsh.model.occ
        self._param = inputParams
        
        self._unitCell = 'Hexagonal'
        self._surrounding = False
        self._holeShape = 'circle'
        self._padShape = 'hexagon'

        return

#**********************************************************************#
    
    def _buildGrid(self, hexCell=True):
        """
        Constructs the volume for the amplification grid.
        
        args:
            holeShape (str): shape of the hole to be made
            hexCell (bool): Determines whether the holes have a square or 
        hexagonal arrangement.
        
        returns:
            gridVolume: object representing the volume of the amplification grid.
        """
        # TODO: can this be combined with buildSurroundingGrid?
        # Get relevant geometry parameters
        gridThickness = self._param['gridThickness']
        holeRadius = self._param['holeRadius']
        pitch = self._param['pitch']
        holeShape = self._holeShape
        
        # Create hole cut tool based on given hole shape.
        match holeShape:
            case 'circle':
                centerGridHole = self._occ.addCylinder(
                    0, 0, -gridThickness/2,
                    0, 0, gridThickness,
                    holeRadius
                )
            case 'hexagon':
                centerGridHole = self._createHexagon(
                    holeRadius, -gridThickness/2, gridThickness
                )
            case 'octagon':
                centerGridHole = self._createOctagon(
                    holeRadius, -gridThickness/2, gridThickness
                )
        
        # Determine if the unit cell is hexagonal or not.
        if hexCell:
            # Create a grid without holes
            gridBox = self._occ.addBox(
                0, 0, -gridThickness/2,
                pitch*math.sqrt(3)/2, pitch/2, gridThickness
            )
            
            # Add corner hole
            cornerGridHole = self._occ.copy([(3, centerGridHole)])
            self._occ.translate(
                cornerGridHole, 
                pitch*math.sqrt(3)/2, 
                pitch/2,
                0
            )
            
            # Use the hole cut tools to cut a hole into the grid volume.
            gridVolume, _ = self._occ.cut(
                [(3, gridBox)],
                [(3, centerGridHole), cornerGridHole[0]]
            )

        else:
            # Create grid without holes
            gridBox = self._occ.addBox(
                0, 0, -gridThickness/2,
                pitch/2, pitch/2, gridThickness
            )
            
            # Use the hole cut tool to cut a hole into the grid volume.
            gridVolume, _ = self._occ.cut(
                [(3, gridBox)],
                [(3, centerGridHole)]
            )

        return gridVolume

#**********************************************************************#

    def _buildDielectric(self, hexCell=True):
        """
        Builds the volume associated with the dielectric.
        
        args:
            holeShape (str): shape of the holes for the readout pads.
            hexCell (bool): determines if the arangement is hexagonal or square.
        
        returns:
            dielectricVolume: object for the volume of the dielectric.
        """
        # TODO: can this be combined with buildSurroundingDielectric?
        
        # Get relevant geometry parameters
        pitch = self._param['pitch']
        padLength = self._param['padLength']
        gridStandoff = self._param['gridStandoff']
        thicknessSiO2 = self._param['thicknessSiO2']
        holeShape = self._padShape
        padSurfaces = []
        
        # Determine hole and pad shape
        match holeShape:
            case 'square':
                # Add central pad hole
                centerPadHole = self._occ.addBox(
                    0, 0, -gridStandoff,
                    padLength/2, padLength/2, thicknessSiO2
                )
                # Add central readout pad
                centerPadSurface = self._occ.addRectangle(
                    0, 0, -gridStandoff,
                    padLength/2, padLength/2
                )
                centerPadSurface = (2, centerPadSurface)

            case 'hexagon':
                # Add central pad hole
                centerPadHole = self._createHexagon(
                    padLength, -gridStandoff, thicknessSiO2
                )
                
                # Add central readout pad
                centerPadHex = self._createHexagon(
                    padLength, -gridStandoff
                )
                fullReadoutPlane = self._occ.addBox(
                    0, 0, -gridStandoff,
                    pitch*math.sqrt(3)/2, pitch/2, 1.0
                )
                centerPadSurface, _ = self._occ.intersect(
                    [(2, centerPadHex)],
                    [(3, fullReadoutPlane)],
                    removeObject=True, removeTool=True
                )
                centerPadSurface = centerPadSurface[0]
                
            case 'octagon':
                # Add central pad hole
                centerPadHole = self._createOctagon(
                    padLength, -gridStandoff, thicknessSiO2
                )
                
                # Add central readout pad
                centerPadHex = self._createOctagon(
                    padLength, -gridStandoff
                )
                fullReadoutPlane = self._occ.addBox(
                    0, 0, -gridStandoff,
                    pitch*math.sqrt(3)/2, pitch/2, 1.0
                )
                centerPadSurface, _ = self._occ.intersect(
                    [(2, centerPadHex)],
                    [(3, fullReadoutPlane)],
                    removeObject=True, removeTool=True
                )
                centerPadSurface = centerPadSurface[0]
        
        padSurfaces.append(centerPadSurface)
        
        # Determine if the unit cell is hexagonal or not.
        if hexCell:
            # Create a dielectric without holes
            dielectricBox = self._occ.addBox(
                0, 0, -gridStandoff, 
                pitch*math.sqrt(3)/2, pitch/2, thicknessSiO2
            )
            
            # Add corner pad hole
            cornerPadHole = self._occ.copy([(3, centerPadHole)])
            self._occ.translate(cornerPadHole, pitch*math.sqrt(3)/2, pitch/2, 0)
            
            # Cut holes in dielectric
            dielectricVolume, _ = self._occ.cut(
                [(3, dielectricBox)],
                [(3, centerPadHole), cornerPadHole[0]]
            )
            
            # Add corner readout pad
            cornerPadSurface = self._occ.copy([centerPadSurface])
            self._occ.rotate(
                cornerPadSurface,
                0, 0, 0,
                0, 0, 1,
                math.pi
            )
            self._occ.translate(cornerPadSurface, pitch*math.sqrt(3)/2, pitch/2, 0)
            padSurfaces.append(cornerPadSurface[0])
        
        else:
            # Create a dielectric without holes
            dielectricBox = self._occ.addBox(
                0, 0, -gridStandoff, 
                pitch/2, pitch/2, thicknessSiO2
            )
            
            #Cut holes in dielectric
            dielectricVolume, _ = self._occ.cut(
                [(3, dielectricBox)],
                [(3, centerPadHole)]
            )
        
        return dielectricVolume, padSurfaces

#**********************************************************************#
    
    def _buildSurroundingGrid(self, hexCell=True):
        """
        Constructs the volume for the amplification grid with surrounding Cells.
        
        args:
            holeShape (str): shape of the hole to be made
            hexCell (bool): Determines whether the holes have a square or 
        hexagonal arrangement.
        
        returns:
            gridVolume: object representing the volume of the amplification grid.
        """
        # Get relevant geometry parameters
        gridThickness = self._param['gridThickness']
        holeRadius = self._param['holeRadius']
        pitch = self._param['pitch']
        
        # Create hole cut tool based on given hole shape.
        match self._holeShape:
            case 'circle':
                centerGridHole = self._occ.addCylinder(
                    0, 0, -gridThickness/2,
                    0, 0, gridThickness,
                    holeRadius
                )
            case 'hexagon':
                centerGridHole = self._createHexagon(
                    holeRadius, -gridThickness/2, gridThickness
                )
            case 'octagon':
                centerGridHole = self._createOctagon(
                    holeRadius, -gridThickness/2, gridThickness
                )
        
        # Determine if the unit cell is hexagonal or not.
        if hexCell:
            # Create a grid without holes
            gridBox = self._occ.addBox(
                -pitch*math.sqrt(3)/2, -pitch, -gridThickness/2,
                pitch*math.sqrt(3), 2*pitch, gridThickness
            )
            
            # Locate the center points for all surrounding holes
            xLength = pitch*math.sqrt(3)/2
            yLength = pitch
            neighborCenters = [
                (0, yLength), #Top
                (0, -yLength), #Bottom
                (xLength, yLength/2), #Top-Right
                (xLength, -yLength/2), #Bottom-Right
                (-xLength, yLength/2), #Top-Left
                (-xLength, -yLength/2) #Bottom-Left
            ]
            
        else:
            # Create grid without holes
            gridBox = self._occ.addBox(
                -pitch, -pitch, -gridThickness/2,
                2*pitch, 2*pitch, gridThickness
            )
            
            # Locate the center points of all surrounding holes
            neighborCenters = [
                (0, pitch), # Top
                (pitch, pitch), # Top-Right
                (pitch, 0), # Right
                (pitch, -pitch), # Bottom-Right
                (0, -pitch), # Bottom
                (-pitch, -pitch), #Bottom-Left
                (-pitch, 0), # Left
                (-pitch, pitch) # Top-Left
            ]

        # Create the surrounding holes
        gridHoleTools = [(3, centerGridHole)]
        for x, y in neighborCenters:
            newHole = self._occ.copy([(3, centerGridHole)])
            self._occ.translate(newHole, x, y, 0)
            gridHoleTools.extend(newHole)
        
        gridVolume, _ = self._occ.cut(
            [(3, gridBox)],
            gridHoleTools
        )
        
        return gridVolume

#**********************************************************************#

    def _buildSurroundingDielectric(self, hexCell=True):
        """
        Builds the volume associated with the dielectric, including surrounding cells.
        
        args:
            holeShape (str): shape of the holes for the readout pads.
            hexCell (bool): determines if the arangement is hexagonal or square.
        
        returns:
            dielectricVolume: object for the volume of the dielectric.
        """
        # Get relevant geometry parameters
        pitch = self._param['pitch']
        padLength = self._param['padLength']
        gridStandoff = self._param['gridStandoff']
        thicknessSiO2 = self._param['thicknessSiO2']
        padSurfaces = []
        
        # Determine hole and pad shape
        match self._padShape:
            case 'square':
                # Add central pad hole
                centerPadHole = self._occ.addBox(
                    -padLength/2, -padLength/2, -gridStandoff,
                    padLength, padLength, thicknessSiO2
                )
                # Add central readout pad
                centerPadSurface = self._occ.addRectangle(
                    -padLength/2, -padLength/2, -gridStandoff,
                    padLength, padLength
                )
                centerPadSurface = (2, centerPadSurface)

            case 'hexagon':
                # Add central pad hole
                centerPadHole = self._createHexagon(
                    padLength, -gridStandoff, thicknessSiO2
                )
                
                # Add central readout pad
                centerPadHex = self._createHexagon(
                    padLength, -gridStandoff
                )
                fullReadoutPlane = self._occ.addBox(
                    -pitch, -pitch, -gridStandoff,
                    2*pitch, 2*pitch, 1.0
                )
                centerPadSurface, _ = self._occ.intersect(
                    [(2, centerPadHex)],
                    [(3, fullReadoutPlane)],
                    removeObject=True, removeTool=True
                )
                centerPadSurface = centerPadSurface[0]
                
            case 'octagon':
                # Add central pad hole
                centerPadHole = self._createOctagon(
                    padLength, -gridStandoff, thicknessSiO2
                )
                
                # Add central readout pad
                centerPadHex = self._createOctagon(
                    padLength, -gridStandoff
                )
                fullReadoutPlane = self._occ.addBox(
                    -pitch, -pitch, -gridStandoff,
                    2*pitch, 2*pitch, 1.0
                )
                centerPadSurface, _ = self._occ.intersect(
                    [(2, centerPadHex)],
                    [(3, fullReadoutPlane)],
                    removeObject=True, removeTool=True
                )
                centerPadSurface = centerPadSurface[0]

        # Determine if the unit cell is hexagonal or not.
        if hexCell:
            xLength = pitch*math.sqrt(3)/2
            yLength = pitch
            
            # Locate the centers of all surrounding holes
            neighborCenters = [
                (0, yLength), #Top
                (0, -yLength), #Bottom
                (xLength, yLength/2), #Top-Right
                (xLength, -yLength/2), #Bottom-Right
                (-xLength, yLength/2), #Top-Left
                (-xLength, -yLength/2) #Bottom-Left
            ]
        
        else:
            xLength = pitch
            yLength = pitch
            # Locate the centers of all surrounding holes
            neighborCenters = [
                (0, pitch), # Top
                (pitch, pitch), # Top-Right
                (pitch, 0), # Right
                (pitch, -pitch), # Bottom-Right
                (0, -pitch), # Bottom
                (-pitch, -pitch), #Bottom-Left
                (-pitch, 0), # Left
                (-pitch, pitch) # Top-Left
            ]
        
        # Create a dielectric without holes
        dielectricBox = self._occ.addBox(
            -xLength, -yLength, -gridStandoff, 
            2*xLength, 2*yLength, thicknessSiO2
        )
        # Create readout surface
        padCutBox = self._occ.addBox(
            -xLength, -yLength, -gridStandoff,
            2*xLength, 2*yLength, 1.0
        )

        # Create pad hole tools for each surrounding pad
        padHoleTools = [(3, centerPadHole)]
        for x, y in neighborCenters:
            newHole = self._occ.copy([(3, centerPadHole)])
            self._occ.translate(newHole, x, y, 0)
            padHoleTools.extend(newHole)
                
        # Cut holes in dielectric
        dielectricVolume, _ = self._occ.cut(
            [(3, dielectricBox)],
            padHoleTools,
        )
        
        # Add surrounding readout pads
        padSurfaces.append(centerPadSurface)
        for x, y in neighborCenters:
            newPad = self._occ.copy([centerPadSurface])
            self._occ.translate(newPad, x, y, 0)
            padSurface, _ = self._occ.intersect(
                newPad,
                [(3, padCutBox)],
                removeObject=True, removeTool=False
            )
            padSurfaces.append(padSurface[0])
        
        return dielectricVolume, padSurfaces

#**********************************************************************#

    def _buildSquareCell(self):
        """
        Builds the geometry for a single, square unit cell.

        Note: Pillars are currently not included in the geometry.

        Returns:
            A dictionary containing the following parts of the unit cell:
                Gas: The gas volume in the unit cell.
                Dielectric: The dielectric volume in the unit cell.
                Grid: The grid volume in the unit cell.
                CenterPad: The center pad surface in the unit cell.
                Cathode: The cathode surface in the unit cell.
        """
        pitch = self._param['pitch']
        gridStandoff = self._param['gridStandoff']
        cathodeHeight = self._param['cathodeHeight']
        
        ## Dielectric
        dielectricVolume, padSurfaces = self._buildDielectric(hexCell=False)
        
        ## Grid
        gridVolume = self._buildGrid(hexCell=False)
        
        ## Gas
        gasHeight = cathodeHeight + gridStandoff
        gasBox = self._occ.addBox(
            0, 0, -gridStandoff,
            pitch/2, pitch/2, gasHeight
        )
        gasVolume, _ = self._occ.cut(
            [(3, gasBox)], 
            [(3, dielectricVolume[0][1]), (3, gridVolume[0][1])], 
            removeObject=True, removeTool=False
        )

        ## Cathode
        cathodeSurface = self._occ.addRectangle(
            0, 0, cathodeHeight,
            pitch/2, pitch/2
        )

        cellParts = {
            'Gas': (3, gasVolume[0][1]),
            'Dielectric': (3, dielectricVolume[0][1]),
            'Grid': (3, gridVolume[0][1]),
            'CenterPad': padSurfaces[0],
            'Cathode': (2, cathodeSurface)
        }

        return cellParts
    
#**********************************************************************#

    def _buildHexagonalCell(self):
        """
        Builds the geometry for a single unit cell of the FIMS geometry.

        Note: Pillars are currently not included in the geometry.

        Returns:
            A dictionary containing the following parts of the unit cell:
                Gas: The gas volume in the unit cell.
                Dielectric: The dielectric volume in the unit cell.
                Grid: The grid volume in the unit cell.
                CenterPad: The center pad surface in the unit cell.
                CornerPad: The corner pad surface in the unit cell.
                Cathode: The cathode surface in the unit cell.
        """
        pitch = self._param['pitch']
        gridStandoff = self._param['gridStandoff']
        cathodeHeight = self._param['cathodeHeight']
        
        xLength = pitch*math.sqrt(3)/2
        yLength = pitch/2
        outRadius = pitch/math.sqrt(3)

        ## Dielectric
        dielectricVolume, padSurfaces = self._buildDielectric(hexCell=True)
        
        ## Grid
        gridVolume = self._buildGrid(hexCell=True)
        
        ## Gas
        gasHeight = cathodeHeight + gridStandoff
        gasBox = self._occ.addBox(
            0, 0, -gridStandoff,
            xLength, yLength, gasHeight
        )
        gasVolume, _ = self._occ.cut(
            [(3, gasBox)], 
            [(3, dielectricVolume[0][1]), (3, gridVolume[0][1])], 
            removeObject=True, removeTool=False
        )

        ## Cathode
        cathodeSurface = self._occ.addRectangle(
            0, 0, cathodeHeight,
            xLength, yLength
        )

        cellParts = {
            'Gas': (3, gasVolume[0][1]),
            'Dielectric': (3, dielectricVolume[0][1]),
            'Grid': (3, gridVolume[0][1]),
            'CenterPad': padSurfaces[0],
            'CornerPad': padSurfaces[1],
            'Cathode': (2, cathodeSurface)
        }

        return cellParts

#**********************************************************************#
    def _buildSquareCellSurrounding(self):
        """
        """
        pitch = self._param['pitch']
        gridStandoff = self._param['gridStandoff']
        cathodeHeight = self._param['cathodeHeight']

        ## Dielectric
        dielectricVolume, padSurfaces = self._buildSurroundingDielectric(hexCell=False)
    
        ## Grid
        gridVolume = self._buildSurroundingGrid(hexCell=False)
        
        ## Gas
        gasHeight = cathodeHeight + gridStandoff
        gasBox = self._occ.addBox(
            -pitch, -pitch, -gridStandoff,
            2*pitch, 2*pitch, gasHeight
        )
        gasVolume, _ = self._occ.cut(
            [(3, gasBox)], 
            [(3, dielectricVolume[0][1]), (3, gridVolume[0][1])], 
            removeObject=True, removeTool=False
        )

        ## Cathode
        cathodeSurface = self._occ.addRectangle(
            -pitch, -pitch, cathodeHeight,
            2*pitch, 2*pitch
        )

        cellParts = {
            'Gas': gasVolume[0], 
            'Dielectric': dielectricVolume[0],
            'Grid': gridVolume[0],
            'CenterPad': padSurfaces[0],
            'TopPad': padSurfaces[1],
            'RightTopPad': padSurfaces[2],
            'RightPad': padSurfaces[3],
            'RightBottomPad': padSurfaces[4],
            'BottomPad': padSurfaces[5],
            'LeftBottomPad': padSurfaces[6],
            'LeftPad': padSurfaces[7],
            'LeftTopPad': padSurfaces[8],
            'Cathode': (2, cathodeSurface)
        }

        return cellParts

#**********************************************************************#
    def _buildHexagonalCellSurrounding(self):
        """
        """
        pitch = self._param['pitch']
        gridStandoff = self._param['gridStandoff']
        cathodeHeight = self._param['cathodeHeight']
        xLength = pitch*math.sqrt(3)/2
        yLength = pitch

        ## Dielectric
        dielectricVolume, padSurfaces = self._buildSurroundingDielectric(hexCell=True)
        
        ## Grid
        gridVolume = self._buildSurroundingGrid(hexCell=True)

        ## Gas
        gasHeight = cathodeHeight + gridStandoff
        gasBox = self._occ.addBox(
            -1*xLength, -1*yLength, -gridStandoff,
            2*xLength, 2*yLength, gasHeight
        )
        gasVolume, _ = self._occ.cut(
            [(3, gasBox)], 
            [(3, dielectricVolume[0][1]), (3, gridVolume[0][1])], 
            removeObject=True, removeTool=False
        )

        ## Cathode
        cathodeSurface = self._occ.addRectangle(
            -xLength, -yLength, cathodeHeight,
            2*xLength, 2*yLength
        )

        cellParts = {
            'Gas': gasVolume[0], 
            'Dielectric': dielectricVolume[0],
            'Grid': gridVolume[0],
            'CenterPad': padSurfaces[0],
            'TopPad': padSurfaces[1],
            'BottomPad': padSurfaces[2],
            'RightTopPad': padSurfaces[3],
            'RightBottomPad': padSurfaces[4],
            'LeftTopPad': padSurfaces[5],
            'LeftBottomPad': padSurfaces[6],
            'Cathode': (2, cathodeSurface)
        }

        return cellParts

#**********************************************************************#

    def _makeSquareCell(self):
        """
        Constructs a complete square unit cell structure.

        returns:
            entityMap: map of geometry objects and their values
        """

        allObjects = []

        # TODO: can these be combined?
        if self._surrounding:
            inCell = self._buildSquareCellSurrounding()
        else:
            inCell = self._buildSquareCell()

        allObjects.extend(inCell.values())

        _, entityMap = self._occ.fragment(allObjects, [])
        self._occ.synchronize()

        return entityMap
    
#**********************************************************************#

    def _makeHexagonalCell(self):
        """
        Constructs a complete square unit cell structure.

        returns:
            entityMap: map of geometry objects and their values
        """

        allObjects = []
        # TODO: can these be combined?
        if self._surrounding:
            inCell = self._buildHexagonalCellSurrounding()
        else:
            inCell = self._buildHexagonalCell()

        allObjects.extend(inCell.values())

        _, entityMap = self._occ.fragment(allObjects, [])
        self._occ.synchronize()

        return entityMap
    
#**********************************************************************#
    def _createHexagon(self, outRadius, z, zDist=None):
        """
        Makes a hexagon in the xy-plane with the center at the origin.
        Extrudes it in the z-direction if zDist is provided.

        Args:
            outRadius: The distance from the hexagon center to each vertex.
            z: The z-coordinate of the hexagon.
            zDist: The distance to extrude the hexagon in the z-direction. 
                   If None, the hexagon will remain a 2D surface.
        """
        
        points = []
        for i in range(6):
            angle = math.radians(i*60)
            x = outRadius*math.cos(angle)
            y = outRadius*math.sin(angle)

            inPoint = self._occ.addPoint(x, y, z)
            points.append(inPoint)

        lines = []
        for i in range(6):
            inLine = self._occ.addLine(points[i], points[(i+1)%6])
            lines.append(inLine)

        loop = self._occ.addCurveLoop(lines)
        surface = self._occ.addPlaneSurface([loop])
        if zDist is not None:
            hexagon = self._occ.extrude(
                [(2, surface)],
                0, 0, zDist
            )
            return hexagon[1][1]
        else:
            return surface
        
#**********************************************************************#

    def _createOctagon(self, outRadius, z, zDist=None):
        """
        Makes an octagon in the xy-plane with the center at the origin.
        Extrudes it in the z-direction if zDist is provided.

        Args:
            outRadius: The distance from the octagon center to each vertex.
            z: The z-coordinate of the octagon.
            zDist: The distance to extrude the octagon in the z-direction. 
                   If None, the octagon will remain a 2D surface.
        """
        
        points = []
        for i in range(8):
            angle = math.radians(i*45+22.5) # Rotate by 15 degrees to align flat sides with axes
            x = outRadius*math.cos(angle)
            y = outRadius*math.sin(angle)

            inPoint = self._occ.addPoint(x, y, z)
            points.append(inPoint)

        lines = []
        for i in range(8):
            inLine = self._occ.addLine(points[i], points[(i+1)%8])
            lines.append(inLine)

        loop = self._occ.addCurveLoop(lines)
        surface = self._occ.addPlaneSurface([loop])
        if zDist is not None:
            octagon = self._occ.extrude(
                [(2, surface)],
                0, 0, zDist
            )
            return octagon[1][1]
        else:
            return surface
            
#**********************************************************************#

    def _assignPhysicalGroups(self, entityMap):
        """
        Assigns physical groups to the geometry entities based 
        on their type and location.
        """

        allHexPads = [
            'CentralPad', 'TopPad', 'BottomPad', 
            'RightTopPad', 'RightBottomPad', 
            'LeftTopPad', 'LeftBottomPad'
        ]
        altHexPads = ['CentralPad', 'CornerPad']

        allSquarePads = [
            'CentralPad', 'TopPad', 'RightTopPad',
            'RightPad', 'RightBottomPad', 'BottomPad', 
            'LeftBottomPad', 'LeftPad', 'LeftTopPad'
        ]
        
        
        allVolumes = ['Gas', 'Dielectric', 'Grid']
        otherSurfaces = ['Cathode']

        configuration = {
            'Square': {
                'keys': allVolumes + ['CentralPad'] + otherSurfaces,
                'pads': ['CentralPad']
            },
            'SquareSurrounding': {
                'keys': allVolumes + allSquarePads + otherSurfaces,
                'pads': allSquarePads
            },
            'Hexagonal': {
                'keys': allVolumes + altHexPads + otherSurfaces,
                'pads': altHexPads
            },
            'HexagonalSurrounding': {
                'keys': allVolumes + allHexPads + otherSurfaces,
                'pads': allHexPads
            },
        }

        runOption = self._unitCell
        if self._surrounding:
            runOption += 'Surrounding'
            
        config = configuration[runOption]

        isHex = 'Hexagonal' in runOption

        partKey = config['keys']
        padNames = config['pads']

        globalGroup = {name: [] for name in ['Gas', 'Dielectric', 'Grid', 'Cathode']}
        padTags = []
        allGridSurfaces = []

        for i, fragments in enumerate(entityMap):
            idx = i % len(partKey) if isHex else i
            if idx >= len(partKey): continue
            
            partType = partKey[idx]
            tags = [f[1] for f in fragments if f[1] > 0]
            if not tags: continue

            if 'Pad' in partType:
                padTags.append(tags)
            else:
                globalGroup[partType].extend(tags)

                # Special boundary handling for the Grid
                if partType == 'Grid':
                    validVol = [(3, t) for t in tags if gmsh.model.occ.getEntities(3).count((3, t)) > 0]
                    if validVol:
                        boundary = gmsh.model.getBoundary(validVol, oriented=False)
                        allGridSurfaces.extend([b[1] for b in boundary])


        # --- Physical Group Assignments ---
        
        # Volumes
        for name in ['Grid', 'Dielectric', 'Gas']:
            if globalGroup[name]:
                gmsh.model.addPhysicalGroup(3, globalGroup[name], name=name)

        # Surfaces (Dim 2)
        if globalGroup['Cathode']:
            gmsh.model.addPhysicalGroup(2, globalGroup['Cathode'], name='Cathode')
        if allGridSurfaces:
            gmsh.model.addPhysicalGroup(2, list(set(allGridSurfaces)), name='Grid')

        # Individual Pads
        for tags, name in zip(padTags, padNames):
            gmsh.model.addPhysicalGroup(2, tags, name=name)

        return

#**********************************************************************#

    def _makeRefinementLines(self, runOption):
        """
        Makes lines with finer FEM values within the geometry.
        
        returns:
            refinementLines (list): list of refinement lines in Gmsh API
        """
        # Cell dimensions
        pitch = self._param['pitch']
        gridThickness = self._param['gridThickness']
        driftLength = self._param['cathodeHeight'] - gridThickness/2.
        sqrt3 = math.sqrt(3)
        
        # List of coordinates for each refinement line point in a specified geometry
        refinementOptions = {
            'Square': [
                (0, 0, driftLength),
                (pitch/2, 0, driftLength), 
                (pitch/2, pitch/2, driftLength),
                (0, pitch/2, driftLength)
            ],
            
            'SquareSurrounding': [
                (-pitch/2, -pitch/2, driftLength),
                (pitch/2, -pitch/2, driftLength), 
                (pitch/2, pitch/2, driftLength),
                (-pitch/2, pitch/2, driftLength)
            ],
            
            'Hexagonal': [
                (0, 0, driftLength),
                (pitch/sqrt3, 0, driftLength), 
                (pitch/sqrt3/2, pitch/2, driftLength),
                (0, pitch/2, driftLength)
            ],
            
            'HexagonalSurrounding': [
                (pitch/sqrt3, 0, driftLength), 
                (pitch/sqrt3/2, pitch/2, driftLength),
                (-pitch/sqrt3/2, pitch/2, driftLength),
                (-pitch/sqrt3, 0, driftLength),
                (-pitch/sqrt3/2, -pitch/2, driftLength),
                (pitch/sqrt3/2, -pitch/2, driftLength)
            ]
        }
        
        refinement = refinementOptions[runOption]
        refinementLines = []
        
        firstPoint = self._occ.addPoint(*refinement[0])
        curPoint = firstPoint
        for x, y, z in refinement[1:]:
            newPoint = self._occ.addPoint(x, y, z)
            newLine = self._occ.addLine(curPoint, newPoint)
            refinementLines.append(newLine)
            
            curPoint = newPoint
            
        finalLine = self._occ.addLine(curPoint, firstPoint)
        refinementLines.append(finalLine)

        return refinementLines

#**********************************************************************#

    def _setMeshSizes(self):
        """Sets the mesh sizes for the geometry based on the run option."""
        sqrt3 = math.sqrt(3)

        # Cell dimensions
        pitch = self._param['pitch']
        holeRadius = self._param['holeRadius']
        gridThickness = self._param['gridThickness']
        thicknessSiO2 = self._param['thicknessSiO2']
        padLength = self._param['padLength']
        cathodeHeight = self._param['cathodeHeight']
        gridStandoff = self._param['gridStandoff']

        # Derived cell lengths
        driftLength = cathodeHeight - gridThickness/2.
        SiO2Height = thicknessSiO2 - gridStandoff + gridThickness/2.
        xLength = pitch*sqrt3/2
        yLength = pitch/2
        outRadius = pitch/sqrt3
        
        #=========================#
        #=== DEFINE MESH SIZES ===#
        #=========================#
        fineMesh = gridThickness*(3./4.)
        gridMesh = gridThickness/4.
        refineMesh = gridThickness*(3./2.)
        backgroundMesh = pitch/4.
        #=========================#
        
        # FEM region scales
        smallRadius = min(holeRadius, padLength)
        largeRadius = max(holeRadius, padLength)
        htransitionWidth = pitch/10.
        vtransitionWidth = driftLength/10.
        refineRadius = (pitch-holeRadius)/2.
        
        # Assign the correct boundary limits to the FEM
        meshSettings = {
            'Square': {
                'x': (0, pitch/2), 
                'y': (0, pitch/2)
            },
            'SquareSurrounding': {
                'x': (-pitch, pitch), 
                'y': (-pitch, pitch)
            },
            'Hexagonal': {
                'x': (0, xLength), 
                'y': (0, yLength)
            },
            'HexagonalSurrounding': {
                'x': (-xLength, xLength), 
                'y': (-yLength, yLength)
            }
        }
        runOption = self._unitCell
        if self._surrounding:
            runOption += 'Surrounding'
        bounds = meshSettings[runOption]
        
        # Create a line from the center of the pad to above the center hole
        pipeBottom = self._occ.addPoint(
            0, 0, -self._param['gridStandoff'], 
        )
        pipeTop = self._occ.addPoint( 
            0, 0, driftLength/10.
        ) 
        amplificationLine = self._occ.addLine(pipeBottom, pipeTop)
        
        # Create lines for refinement around the top edge of the unit cell
        refinementLines = self._makeRefinementLines(runOption)
        
        self._occ.synchronize()

        # Find distance from center line
        gmsh.model.mesh.field.add('Distance', 1)
        gmsh.model.mesh.field.setNumbers(1, 'EdgesList', [amplificationLine])
        
        # Find distance from refinement lines
        gmsh.model.mesh.field.add('Distance', 2)
        gmsh.model.mesh.field.setNumbers(2, 'EdgesList', refinementLines)
        
        # Define fine mesh within smallRadius
        gmsh.model.mesh.field.add('Threshold', 3)
        gmsh.model.mesh.field.setNumber(3, 'InField', 1)
        gmsh.model.mesh.field.setNumber(3, 'SizeMin', fineMesh)
        gmsh.model.mesh.field.setNumber(3, 'SizeMax', backgroundMesh)
        gmsh.model.mesh.field.setNumber(3, 'DistMin', smallRadius)
        gmsh.model.mesh.field.setNumber(3, 'DistMax', smallRadius + htransitionWidth)

        # Define ultra fine mesh inside the entire grid
        gmsh.model.mesh.field.add('Box', 4)
        gmsh.model.mesh.field.setNumber(4, 'VIn', gridMesh)
        gmsh.model.mesh.field.setNumber(4, 'VOut', backgroundMesh)
        gmsh.model.mesh.field.setNumber(4, 'XMin', bounds['x'][0])
        gmsh.model.mesh.field.setNumber(4, 'XMax', bounds['x'][1])
        gmsh.model.mesh.field.setNumber(4, 'YMin', bounds['y'][0])
        gmsh.model.mesh.field.setNumber(4, 'YMax', bounds['y'][1])
        gmsh.model.mesh.field.setNumber(4, 'ZMin', -gridThickness)
        gmsh.model.mesh.field.setNumber(4, 'ZMax', gridThickness)
        gmsh.model.mesh.field.setNumber(4, 'Thickness', gridThickness)
        
        # Define fine mesh in the vicinity around the entire grid
        gmsh.model.mesh.field.add('Box', 5)
        gmsh.model.mesh.field.setNumber(5, 'VIn', refineMesh)
        gmsh.model.mesh.field.setNumber(5, 'VOut', backgroundMesh)
        gmsh.model.mesh.field.setNumber(5, 'XMin', bounds['x'][0])
        gmsh.model.mesh.field.setNumber(5, 'XMax', bounds['x'][1])
        gmsh.model.mesh.field.setNumber(5, 'YMin', bounds['y'][0])
        gmsh.model.mesh.field.setNumber(5, 'YMax', bounds['y'][1])
        gmsh.model.mesh.field.setNumber(5, 'ZMin', -gridThickness*2.)
        gmsh.model.mesh.field.setNumber(5, 'ZMax', gridThickness*2.)
        gmsh.model.mesh.field.setNumber(5, 'Thickness', vtransitionWidth)
        
        # Define fine mesh around the pad
        gmsh.model.mesh.field.add('Box', 6)
        gmsh.model.mesh.field.setNumber(6, 'VIn', refineMesh)
        gmsh.model.mesh.field.setNumber(6, 'VOut', backgroundMesh)
        gmsh.model.mesh.field.setNumber(6, 'XMin', bounds['x'][0])
        gmsh.model.mesh.field.setNumber(6, 'XMax', bounds['x'][1])
        gmsh.model.mesh.field.setNumber(6, 'YMin', bounds['y'][0])
        gmsh.model.mesh.field.setNumber(6, 'YMax', bounds['y'][1])
        gmsh.model.mesh.field.setNumber(6, 'ZMin', SiO2Height - thicknessSiO2/2.)
        gmsh.model.mesh.field.setNumber(6, 'ZMax', SiO2Height + thicknessSiO2/2.)
        gmsh.model.mesh.field.setNumber(6, 'Thickness', vtransitionWidth/2.)
        
        # Define coarse mesh near edge/corner refinement lines
        gmsh.model.mesh.field.add('Threshold', 7)
        gmsh.model.mesh.field.setNumber(7, 'InField', 2)
        gmsh.model.mesh.field.setNumber(7, 'SizeMin', refineMesh)
        gmsh.model.mesh.field.setNumber(7, 'SizeMax', backgroundMesh)
        gmsh.model.mesh.field.setNumber(7, 'DistMin', refineRadius)
        gmsh.model.mesh.field.setNumber(7, 'DistMax', refineRadius + vtransitionWidth)
        
        # Use the smallest mesh size
        gmsh.model.mesh.field.add('Min', 8)
        gmsh.model.mesh.field.setNumbers(8, 'FieldsList', [3,4,5,6,7])
        gmsh.model.mesh.field.setAsBackgroundMesh(8)
        
        # Final settings
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0) # FEM already defined in volume
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0) # FEM not set at points
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 1)
        gmsh.option.setNumber('Mesh.MeshSizeMax', backgroundMesh)
        gmsh.option.setNumber('Mesh.Algorithm3D', 10) # Runs faster than default

        return
    
#**********************************************************************#

    def generateMesh(self, geoConfig, runGUI=False):
        """
        Generates the mesh for the given run option using Gmsh.

        Args:
            geoConfig (dict): The run options for the geometry.
            runGUI (bool): Whether to launch the Gmsh GUI.
        """
        self._unitCell = geoConfig['unitCell']
        self._surrounding = geoConfig['surrounding']
        self._holeShape = geoConfig['holeShape']
        self._padShape = geoConfig['padShape']
        
        filePath = 'Geometry'
        if self._surrounding:
            filename = os.path.join(filePath, f'{self._unitCell}Surrounding.msh')
        else:
            filename = os.path.join(filePath, f'{self._unitCell}.msh')

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()

        gmsh.option.setNumber('Mesh.ElementOrder', 2)
        gmsh.option.setNumber('Mesh.HighOrderOptimize', 1)
        gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary', 1)
        gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 1)

        gmsh.model.add(filename)
        # TODO: restore singleHexagon? 
        # Or, perhaps extending the bounds in the existing methods is sufficient?
        match self._unitCell:
            case 'Square':
                allCellsMap = self._makeSquareCell()
            
            case 'Hexagonal':
                allCellsMap = self._makeHexagonalCell()
             
        self._assignPhysicalGroups(allCellsMap)
        self._setMeshSizes()

        print('\tCreating mesh...')
        gmsh.model.mesh.generate(3)
        gmsh.write(filename)

        logPath = os.path.join(os.getcwd(), 'log/logGmsh.txt')
        logMessages = gmsh.logger.get()
        with open(logPath, 'w') as f:
            for msg in logMessages:
                f.write(f"{msg}\n")
            
        if runGUI:
            gmsh.fltk.run()

        gmsh.finalize()

        return


#**********************************************************************#
#**********************************************************************#
#**********************************************************************#
    
class elmerClass:
    """
    Class for generating Elmer SIF files and running Elmer simulations.

    Can be used for any mesh generated by gmshClass.

    Can optionally calculate the capacitance matrix for the geometry.
    """

    def __init__(self, runOption, capacitance=False):
        """
        Initializes the elmerClass instance.
        
        Args:
            runOption (str): The run option for the simulation.
            capacitance (bool): Whether to calculate the capacitance matrix.
        """

        self._runOption = runOption
        self._setElectrodeMap()
            
        self._capacitance = capacitance

        self._meshFilename = f'{runOption}.msh'

        self._elmerName = runOption
        if self._capacitance:
            self._elmerName += 'Capacitance'
        
        #Currently not generating pillars
        self._numMaterials = 3
        
        self._elmerBaseInfo()
        self._elmerSimulationInfo()
        self._selectSolver()
        self._addMaterials()
        self._assignBoundaryConditions()

        self._writeAllSIF()

        return
    
#**********************************************************************#

    def _setElectrodeMap(self):
        """
        Sets the mapping from physical group numbers to electrode 
        names based on the run option.
        """        
        self._electrodeMap = {
            1: 'Cathode', 2: 'Grid', 3: 'CentralPad'
        }

        match self._runOption:
            case 'Square':
                pass
            
            case 'SquareSurrounding':
                self._electrodeMap.update({
                    4: 'TopPad', 5: 'RightTopPad', 6: 'RightPad',
                    7: 'RightBottomPad', 8: 'BottomPad', 9: 'LeftBottomPad',
                    10: 'LeftPad', 11: 'LeftTopPad'
                })

            case 'Hexagonal':
                self._electrodeMap.update({4: 'CornerPad'})

            case 'HexagonalSurrounding':
                self._electrodeMap.update({
                    4: 'TopPad', 5: 'BottomPad',
                    6: 'RightTopPad', 7: 'RightBottomPad',
                    8: 'LeftTopPad', 9: 'LeftBottomPad'
                })

            case _:
                raise ValueError('Invalid run option.')
            
        return

#**********************************************************************#

    def _elmerBaseInfo(self):
        """
        Initializes the base information for the Elmer SIF file.
        """
        
        self._intro = (
            '! This file was generated by the FIMS code.\n'
            '! Do NOT edit manually.\n\n'
        )

        self._header = (
            'Header\n'
            '  CHECK KEYWORDS Warn\n'
            '  Mesh DB "elmerResults" "."\n'
            '  Include Path ""\n'
            '  Results Directory "elmerResults"\n'
            'End\n\n'
        )

        self._constants = {
            'Constants': {
                'Permittivity of Vacuum': '8.85418781e-12',
                'Permeability of Vacuum': '1.25663706e-6',
                'Boltzmann Constant': '1.380649e-23',
                'Unit Charge': '1.6021766e-19'
            }
        }

        self._equation = {
            'Equation 1': {
                'Name': '"EField"',
                'Electric Field': 'Computed',
                'Active Solvers(1)': '1'
            }
        }

        return
    
#**********************************************************************#

    def _elmerSimulationInfo(self): 

        self._simulation = {
            'Simulation': {
                'Max Output Level': '5',
                'Coordinate System': 'Cartesian',
                'Coordinate Mapping(3)': '1 2 3',
                'Simulation Type': 'Steady state',
                'Steady State Max Iterations': '1',
                'Output Intervals(1)': '1',
                'Coordinate Scaling': '1e-6',
                'Solver Input File': f'{self._runOption}.sif',
                '! Post File': f'{self._elmerName}.ep, {self._elmerName}.vtu',
                'Output file': f'"elmerResults/{self._elmerName}.result"'
            }
        }

        self._weighting = {}
        for i, electrode in self._electrodeMap.items():
            #Don't need weighting for cathode or grid
            if electrode == 'Cathode' or electrode == 'Grid':
                continue

            self._weighting[i] = {
            'Simulation': {
                'Max Output Level': '5',
                'Coordinate System': 'Cartesian',
                'Coordinate Mapping(3)': '1 2 3',
                'Simulation Type': 'Steady state',
                'Steady State Max Iterations': '1',
                'Output Intervals(1)': '1',
                'Coordinate Scaling': '1e-6',
                'Solver Input File': f'{self._elmerName}{electrode}Weighting.sif',
                'Output file': f'"elmerResults/{self._elmerName}{electrode}Weighting.result"'
            }
        }
            
        return

#**********************************************************************#

    def _selectSolver(self):
        """
        Selects a solver for the Elmer simulation based on whether 
        capacitance calculation is needed.
        """

        self._solver = {
            'Solver 1': {
                'Equation': 'Electrostatics',
                'Variable': 'Potential',
                'Calculate Electric Field': 'True',
                'Procedure': '"StatElecSolve" "StatElecSolver"',
                'Exec Solver': 'Always',
                'Stabilize': 'True',
                'Optimize Bandwidth': 'True',
                'Steady State Convergence Tolerance': '1.0e-5',
                'Linear System Solver': 'Iterative',
                'Linear System Iterative Method': 'BiCGStab',
                'Linear System Max Iterations': '500',
                'Linear System Convergence Tolerance': '1.0e-14',
                'BiCGstabl polynomial degree': '2',
                'Linear System Preconditioning': 'ILU0',
                'Linear System ILUT Tolerance': '1.0e-3',
                'Linear System Abort Not Converged': 'False',
                'Linear System Residual Output': '10',
                'Linear System Precondition Recompute': '1',
                'Output Format': 'Vtu'
            }
        }
        
        if self._capacitance:
            self._solver['Solver 1'].update({
                'Calculate Capacitance Matrix': 'True',
                'Capacitance Matrix Filename': '"elmerResults/CapacitanceMatrix.dat"'
            })

        return
    
#**********************************************************************#

    def _addMaterials(self):
        """Adds materials to the simulation."""

        allMaterials = [
            {
                'Name': '"Air (room temperature)"', 
                'Relative Permittivity': '1.0'
            },
            {
                'Name': '"Aluminum (generic)"', 
                'Relative Permittivity': '1e10' #TODO - 1e6 instead (easier for solver)?
            },
            {
                'Name': '"SiO2"', 
                'Relative Permittivity': '3.9'
            },
            {
                'Name': '"Pillars"',
                'Relative Permittivity': '3.0'
            },
        ]

        allBodies = [
            {
                'Target Bodies(1)': 3, 
                'Name': '"Gas"', 
                'Equation': 1, 
                'Material': 1
            },
            {
                'Target Bodies(1)': 1, 
                'Name': '"Amplification Grid"', 
                'Equation': 1, 
                'Material': 2
            },
            {
                'Target Bodies(1)': 2, 
                'Name': '"SiO2"', 
                'Equation': 1, 
                'Material': 3
            },
            {
                'Target Bodies(1)': 4, 
                'Name': '"Pillars"', 
                'Equation': 1, 
                'Material': 4
            },
        ]


        self._materials = {}
        self._bodies = {}

        for i in range(self._numMaterials):
            self._materials[f'Material {i+1}'] = allMaterials[i]
            self._bodies[f'Body {i+1}'] = allBodies[i]

        self._makeDielectricsFile()

        return  
    
#**********************************************************************#

    def _makeDielectricsFile(self):
        """Writes the dielectric properties to a file."""

        dielectricValues = ['1e10', '3.9', '1.0', '3.0']

        try:
            with open('Geometry/dielectrics.dat', 'w') as f:
                f.write(self._numMaterials.__str__() + '\n')

                for i in range(self._numMaterials):
                    f.write(f'{i+1} {dielectricValues[i]}\n')

        except Exception as e:
            print(f"Error writing dielectrics.dat: {e}")
    
        return

#**********************************************************************#

    def _assignBoundaryConditions(self):
        """
        Assigns boundary conditions to the surfaces in the geometry 
        based on the number of pads.
        
        Adapts based on whether capacitance matrix is needed.
        """

        self._boundaries = {}

        numPads = len(self._electrodeMap) - 2 #Subtract cathode and grid

        for i in range(1, numPads+3):
            name = self._electrodeMap[i]
            content = {
                'Target Boundaries(1)': i,
                'Name': f'"{name}"',
            }

            if self._capacitance:
                content['Capacitance Body'] = i
            else:
                content['Potential'] = '0.0'

            self._boundaries[f'Boundary Condition {i}'] = content

        dielectricID = numPads+3
        boundaryID = numPads+4

        self._boundaries[f'Boundary Condition {dielectricID}'] = {
            'Target Boundaries(1)': dielectricID,
            'Name': '"DielectricSurfaceCharge"'
        }
        if not self._capacitance:
            self._boundaries[f'Boundary Condition {dielectricID}'].update({
                'Charge Density': 'Variable Coordinate 1, Coordinate 2, Coordinate 3',
                'File': '"chargeBuildup.dat"'
            })

        self._boundaries[f'Boundary Condition {boundaryID}'] = {
            'Target Boundaries(1)': boundaryID,
            'Name': '"MirrorBoundaries"',
            'Electric Flux': '0.0'
        }

        return

#**********************************************************************#

    def _writeAllSIF(self):
        """
        Writes all SIF files for the Elmer simulation.
        
        Including the main file and the weighting files.
        """
        self._writeSIF()
        self._writeSIFWeighting()

        return

#**********************************************************************#

    def _writeSIF(self):
        """
        Writes the physics SIF file for the Elmer simulation.

        Note that grid, cathode, and pad potentials are 0.0 by default.
        """

        with open(f'Geometry/{self._runOption}.sif', 'w') as f:

            f.write(self._intro)
            f.write(self._header)

            sections = [
                self._simulation, self._constants,
                self._solver, self._equation, self._materials,
                self._bodies, self._boundaries
            ]

            for section in sections:
                for title, content in section.items():
                    f.write(f'{title}\n')

                    for key, value in content.items():

                        #Handle charge density file differently
                        if 'Boundary' in title and key == 'File':
                            f.write(f'  {key} {value}\n')
                        else:
                            f.write(f'  {key} = {value}\n')
                            
                    f.write('End\n\n')
        return

#**********************************************************************#

    def _writeSIFWeighting(self):
        """Writes the SIF weighing files for the Elmer simulation."""
        for i, electrode in self._electrodeMap.items():
            if electrode == 'Cathode' or electrode == 'Grid':
                continue
            inElectrode = False

            with open(f'Geometry/{self._elmerName}{electrode}Weighting.sif', 'w') as f:

                f.write(self._intro)
                f.write(self._header)

                sections = [
                    self._weighting[i], self._constants,
                    self._solver, self._equation, self._materials,
                    self._bodies, self._boundaries
                ]

                for section in sections:
                    for title, content in section.items():
                        f.write(f'{title}\n')

                        for key, value in content.items():

                            #Handle charge density file differently
                            if 'Boundary' in title:
                                if key == 'File':
                                    f.write(f'  {key} {value}\n')

                                elif key == 'Name' and f'"{electrode}"' in value:
                                    f.write(f'  {key} = {value}\n')
                                    inElectrode = True

                                elif inElectrode and key == 'Potential':
                                    f.write(f'  {key} = 1.0\n')
                                    inElectrode = False

                                else:
                                    f.write(f'  {key} = {value}\n')
                            else:
                                f.write(f'  {key} = {value}\n')
                                
                        f.write('End\n\n')

        return

#**********************************************************************#

    def runElmer(self, solveWeighting=True):
        """
        Runs the Elmer simulation using the generated SIF file and mesh.
        """
        self._executeElmer('ElmerGrid')
        self._executeElmer('ElmerSolver')

        if not self._capacitance and solveWeighting:
            self._executeElmer('ElmerWeighting')

        return

#**********************************************************************#

    def _executeElmer(self, processName):
        """
        Executes a given Elmer process.

        Options are:
        - 'ElmerGrid': Generates the Elmer mesh from the Gmsh mesh.
        - 'ElmerSolver': Runs the main Elmer simulation.
        - 'ElmerWeighting': Runs the weighting potential simulations for each pad.

        Args:
            processName: The name of the Elmer process to execute.
        """

        originalCWD = os.getcwd()
        os.chdir('./Geometry')

        os.makedirs('elmerResults', exist_ok=True)

        padList = [e for e in self._electrodeMap.values() if e not in {'Cathode', 'Grid'}]

        elmerCommands = {
            'ElmerGrid': [
                ['ElmerGrid', '14', '2', self._meshFilename, '-names', '-out', 'elmerResults', '-autoclean']
            ],
            'ElmerSolver': [
                ['ElmerSolver', f'{self._runOption}.sif']
            ],
            'ElmerWeighting': [
                [f'ElmerSolver', f'{self._elmerName}{e}Weighting.sif'] for e in padList
            ]
        }

        try:
            print(f'\tExecuting {processName}...')
            logFile = f'log/log{processName}.txt'
            
            with open(os.path.join(originalCWD, logFile), 'w+') as elmerOutput:
                startTime = time.monotonic()
                for cmd in elmerCommands[processName]:
                    subprocess.run(
                        cmd,
                        stdout=elmerOutput,
                        check=True
                    )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\n{processName} run time: {endTime - startTime} s')
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Elmer failed with exit code {e.returncode}.')
        
        finally:
            os.chdir(originalCWD)

        return
    
#**********************************************************************#

    def _setPotential(self, electrode='', potential=0.0):
        """
        Sets the potential for a given electrode in the SIF file.

        Args:
            electrode: The name of the electrode to set the potential.
            potential: The potential value to set (in volts).
        """

        if electrode not in [
            'Cathode', 'Grid', 
            'CentralPad', 'CornerPad', 
            'TopPad', 'BottomPad',
            'RightTopPad', 'RightBottomPad', 
            'LeftTopPad', 'LeftBottomPad'
        ]:
            raise ValueError('Invalid electrode name.')

        with open(f'Geometry/{self._runOption}.sif', 'r') as f:
            sifContent = f.read()

        newContent = ''
        lines = sifContent.splitlines()
        inBoundary = False
        inElectrode = False

        for line in lines:
            strippedLine = line.strip()
            if strippedLine.startswith('Boundary Condition'):
                inBoundary = True
                newContent += line + '\n'
                continue
            
            if inBoundary and strippedLine.startswith('Name') and f'"{electrode}"' in strippedLine:
                newContent += line + '\n'
                inElectrode = True
                continue

            if inElectrode and strippedLine.startswith('Potential'):
                newContent += f'  Potential = {potential}\n'
                inElectrode = False
                inBoundary = False
                continue
            
            newContent += line + '\n'

        with open(f'Geometry/{self._runOption}.sif', 'w') as f:
            f.write(newContent)

        return
    
#**********************************************************************#

    def resetPotentials(self):
        """Resets the potentials for all electrodes in the SIF file."""

        with open(f'Geometry/{self._runOption}.sif', 'r') as f:
            sifContent = f.read()

        newContent = ''
        lines = sifContent.splitlines()
        inBoundary = False

        for line in lines:
            strippedLine = line.strip()
            if strippedLine.startswith('Boundary Condition'):
                inBoundary = True
                newContent += line + '\n'
                continue
            
            if inBoundary and strippedLine.startswith('Potential'):
                newContent += f'  Potential = 0.0\n'
                inBoundary = False
                continue
            
            newContent += line + '\n'

        with open(f'Geometry/{self._runOption}.sif', 'w') as f:
            f.write(newContent)

        return






