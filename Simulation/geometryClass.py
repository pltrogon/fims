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

from configs import GeometryConfiguration, UnitCell, HoleShape, PadShape, ScaleOption

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

    def __init__(self, inputParam=None, geoConfig=None):
        """
        Initializes a geometryClass instance with parameters and geometry options.
        """
        self._param = inputParam
        self._geoConfig = geoConfig
        self._runGUI = False
        
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
            'fieldRatio',
        ]

        if self._param is None:
            raise ValueError('Error - Invalid parameters.')

        for key in neededParameters:
            if key not in self._param:
                raise ValueError(f"Error - Missing '{key}'")
            if self._param[key] <= 0:
                raise ValueError(f"Error - '{key}' must be positive.")

        # Find bounds of the unit cell
        inRadius = self._param['pitch'] / 2

        hexCell = self._geoConfiguration.unitCell == UnitCell.HEXAGON

        # Find scaling of the hole size
        match self._geoConfiguration.holeShape:
            case HoleShape.HEXAGON:
                holeScale = math.sqrt(3) / 2 if hexCell else 1.0

            case HoleShape.OCTAGON:
                holeScale = (
                    1.00863 if hexCell else math.cos(math.radians(22.5))
                )

            case HoleShape.TRIVIALPURSUIT:
                holeScale = 2.2

            case HoleShape.NESTEGGS:
                holeScale = 3.55

            case _:
                holeScale = 1.0

        # Find scaling of the pad size
        match self._geoConfiguration.padShape:
            case PadShape.SQUARE:
                padScale = (
                    1 / (2 * math.cos(math.radians(60))) if hexCell else 0.5
                )

            case PadShape.HEXAGON:
                padScale = math.sqrt(3) / 2 if hexCell else 1.0

            case PadShape.OCTAGON:
                padScale = (
                    1.00863 if hexCell else math.cos(math.radians(22.5))
                )

            case _:
                padScale = 1.0

        # Grid hole must be smaller than the pitch
        if self._param['holeRadius'] * holeScale >= inRadius:
            raise ValueError('Error - Hole size too large relative to cell.')

        if self._param['padLength'] * padScale >= inRadius:
            raise ValueError('Error - Pad larger than cell.')

        return

#**********************************************************************#

    def setGUI(self, runGUI=True):
        """
        Sets whether the Gmsh GUI runs when creating the geometry.
        """
        self._runGUI = runGUI

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
        runOption = (
            self._geoConfig.unitCell.value 
            + self._geoConfig.scale.value
        )
        
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
        """
        Executes field solvers based on target simulation mode.
        """
        self._checkParameters()
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
        gridVoltage, cathodeVoltage = self._findPotentials()

        self._elmerClass.resetPotentials()
        self._elmerClass._setPotential('Grid', gridVoltage)
        self._elmerClass._setPotential('Cathode', cathodeVoltage)

        return
        
#**********************************************************************#

    def _findPotentials(self):
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

        driftGap = self._param['cathodeHeight'] - halfGrid
        amplificationGap = self._param['gridStandoff'] - halfGrid

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
        
        self._unitCell = 'hexagon'
        self._scale = 'corner'
        self._holeShape = 'circle'
        self._padShape = 'hexagon'

        return

#**********************************************************************#
    
    def _buildGrid(self):
        """
        Constructs the volume for the amplification grid with surrounding Cells.
        
        returns:
            gridVolume: object representing the volume of the amplification grid.
        """
        # Get relevant geometry parameters
        gridThickness = self._param['gridThickness']
        holeRadius = self._param['holeRadius']
        pitch = self._param['pitch']
        yLength = pitch
        
        # Determine if the unit cell is hexagonal or not.
        if self._unitCell == 'hexagon':
            # Create single cell grid without holes
            xLength = pitch*math.sqrt(3)
            gridBox = self._createHexagon(
                xLength/3, -gridThickness/2, gridThickness
            )
            
        else:
            xLength = pitch
            # Create single cell grid without holes
            gridBox = self._occ.addBox(
                -xLength/2, -yLength/2, -gridThickness/2,
                xLength, yLength, gridThickness
            )
        
        # Create hole cut tool based on given hole shape.
        match self._holeShape:
            case 'circle':
                centerGridHole = self._occ.addCylinder(
                    0, 0, -gridThickness/2,
                    0, 0, gridThickness,
                    holeRadius
                )
                gridHoleTools = [(3, centerGridHole)]
                
            case 'hexagon':
                centerGridHole = self._createHexagon(
                    holeRadius, -gridThickness/2, gridThickness
                )
                gridHoleTools = [(3, centerGridHole)]
                
            case 'octagon':
                centerGridHole = self._createOctagon(
                    holeRadius, -gridThickness/2, gridThickness
                )
                gridHoleTools = [(3, centerGridHole)]
                
            case 'triangle':
                centerGridHole = self._createTriangle(
                    holeRadius, -gridThickness/2, gridThickness,
                )
                gridHoleTools = [(3, centerGridHole)]
                
            case 'nesteggs':
                # Create all holes
                hole1 = self._occ.addCylinder(
                    0, 0, -gridThickness/2,
                    0, 0, gridThickness,
                    holeRadius
                )
                hole2 = self._occ.copy([(3, hole1)])
                hole3 = self._occ.copy([(3, hole1)])
                hole4 = self._occ.copy([(3, hole1)])
                hole5 = self._occ.copy([(3, hole1)])
                hole6 = self._occ.copy([(3, hole1)])
                
                # Translate holes off center
                xCenter = pitch/math.sqrt(3)/4
                yCenter = pitch/4
                self._occ.translate([(3, hole1)], -xCenter*2, 0, 0)
                self._occ.translate(hole2, xCenter*2, 0, 0)
                self._occ.translate(hole3, -xCenter, -yCenter, 0)
                self._occ.translate(hole4, -xCenter, yCenter, 0)
                self._occ.translate(hole5, xCenter, -yCenter, 0)
                self._occ.translate(hole6, xCenter, yCenter, 0)
                
                # Append to list for duplication to surrounding cells
                gridHoleTools = [(3, hole1), hole2[0], hole3[0], hole4[0], hole5[0], hole6[0]]
            
            case 'trivialpursuit':
                # Create all holes
                hole1 = self._createTriangle(
                    holeRadius,
                    -gridThickness/2,
                    gridThickness,
                    angle=30
                )
                hole2 = self._createTriangle(
                    holeRadius,
                    -gridThickness/2,
                    gridThickness,
                    angle=210
                )
                hole3 = self._createTriangle(
                    holeRadius,
                    -gridThickness/2,
                    gridThickness,
                    angle=30
                )
                hole4 = self._createTriangle(
                    holeRadius,
                    -gridThickness/2,
                    gridThickness,
                    angle=210
                )
                hole5 = self._createTriangle(
                    holeRadius,
                    -gridThickness/2,
                    gridThickness,
                    angle=30
                )
                hole6 = self._createTriangle(
                    holeRadius,
                    -gridThickness/2,
                    gridThickness,
                    angle=210
                )
                
                # Translate holes off center
                xCenter = pitch*math.sqrt(3)/8
                yCenter = pitch/4
                self._occ.translate([(3, hole1)], xCenter, yCenter/2, 0)
                self._occ.translate([(3, hole2)], 0, yCenter, 0)
                self._occ.translate([(3, hole3)], -xCenter, yCenter/2, 0)
                self._occ.translate([(3, hole4)], -xCenter, -yCenter/2, 0)
                self._occ.translate([(3, hole5)], 0, -yCenter, 0)
                self._occ.translate([(3, hole6)], xCenter, -yCenter/2, 0)
                
                # Append to list for duplication to surrounding cells
                gridHoleTools = [(3, hole1), (3, hole2), (3, hole3), (3, hole4), (3, hole5), (3, hole6)]
        
        # Adjust grid size to match scale
        match self._scale:
            case 'single':
                adjustedGrid = [(3, gridBox)]
            
            case 'corner':
                # Remove unit cell box and replace it with just the corner.
                self._occ.remove([(3, gridBox)], recursive=True)
                cornerGrid = self._occ.addBox(
                    0, 0, -gridThickness/2,
                    xLength/2, yLength/2, gridThickness
                )
                adjustedGrid = [(3, cornerGrid)]

            case 'surrounding':
                adjustedGrid = self._copyToSurrounding((3, gridBox))
                surroundingHoleList = []
                for hole in gridHoleTools:
                    surroundingHoleList.append(self._copyToSurrounding(hole, fuse=False))
                gridHoleTools += surroundingHoleList
            
            case 'half':
                self._occ.remove([(3, gridBox)], recursive=True)
                halfBox = self._occ.addBox(
                    -xLength/2, -yLength, -gridThickness/2,
                    xLength, 2*yLength, gridThickness
                )
                adjustedGrid = [(3, halfBox)]
                surroundingHoleList = []
                for hole in gridHoleTools:
                    surroundingHoleList.append(self._copyToSurrounding(hole, fuse=False))
                gridHoleTools += surroundingHoleList
        
        # Duplicate into surrounding cells
        gridVolume, _ = self._occ.cut(
            adjustedGrid,
            gridHoleTools
        )
        
        return gridVolume

#**********************************************************************#

    def _buildDielectric(self):
        """
        Builds the volume associated with the dielectric, including surrounding cells.
        
        returns:
            dielectricVolume: object for the volume of the dielectric.
        """
        # Get relevant geometry parameters
        padThickness = self._param['padThickness']
        pitch = self._param['pitch']
        padLength = self._param['padLength']
        thicknessSiO2 = self._param['thicknessSiO2'] + padThickness
        padBase = -self._param['gridStandoff'] - padThickness - self._param['gridThickness']/2
        yLength = pitch
        
        # Determine if the unit cell is hexagonal or not.
        if self._unitCell == 'hexagon':
            xLength = pitch*math.sqrt(3)
            # Create a dielectric without holes
            dielectricBox = self._createHexagon(
                xLength/3, padBase, thicknessSiO2
            )
        
        else:
            xLength = pitch
            # Create a dielectric without holes
            dielectricBox = self._occ.addBox(
                -xLength/2, -yLength/2, padBase, 
                xLength, yLength, thicknessSiO2
            )
            
        match self._scale:
            case 'single':
                adjustedDielectric = [(3, dielectricBox)]
            
            case 'corner':
                # Remove unit cell box and replace it with just the corner.
                self._occ.remove([(3, dielectricBox)], recursive=True)
                cornerDielectric = self._occ.addBox(
                    0, 0, padBase,
                    xLength/2, yLength/2, thicknessSiO2
                )
                adjustedDielectric = [(3, cornerDielectric)]
                
            case 'surrounding':
                # Create surrounding dielectric
                adjustedDielectric = self._copyToSurrounding((3, dielectricBox))
            
            case 'half':
                self._occ.remove([(3, dielectricBox)], recursive=True)
                halfBox = self._occ.addBox(
                    -xLength/2, -yLength, padBase,
                    xLength, 2*yLength, thicknessSiO2
                )
                adjustedDielectric = [(3, halfBox)]
                
        # Determine hole and pad shape
        match self._padShape:
            case 'square':
                # Add central pad hole
                centerPadHole = self._occ.addBox(
                    -padLength/2, -padLength/2, padBase,
                    padLength, padLength, thicknessSiO2
                )
                # Add central readout pad object
                centerPad = self._occ.addBox(
                    -padLength/2, -padLength/2, padBase,
                    padLength, padLength, padThickness
                )
                
            case 'hexagon':
                # Add central pad hole
                centerPadHole = self._createHexagon(padLength, padBase, thicknessSiO2)
                
                # Add central readout pad object
                centerPad = self._createHexagon(padLength, padBase, padThickness)
                
            case 'octagon':
                # Add central pad hole
                centerPadHole = self._createOctagon(padLength, padBase, thicknessSiO2)
                
                # Add central readout pad object
                centerPad = self._createOctagon(padLength, padBase, padThickness)
        
        # Create pads and tools for dielectric holes
        padHoleTools = self._copyToSurrounding((3, centerPadHole), fuse=False)
        padList = self._copyToSurrounding((3, centerPad), fuse=False)
        padVolumes = []
        for pad in padList:
            padVolume, _ = self._occ.intersect(
                [pad],
                adjustedDielectric,
                removeObject=True, removeTool=False
            )
            if len(padVolume) > 0:
                padVolumes.append(padVolume[0])
        
        # Cut holes in dielectric
        dielectricVolume, _ = self._occ.cut(
            adjustedDielectric,
            padHoleTools,
        )
        
        return dielectricVolume, padVolumes

#**********************************************************************#

    def _buildGas(self, gridVolume, dielectricVolume):
        """
        Builds the gas volume and the cathode surface for a geometry.
        
        args:
            gridVolume: the volume object for the grid.
            dielectricVolume: the volume object for the dielectric.
            
        returns:
            gasVolume: the volume object for the gas.
            cathodeSurface: the surface object for the cathode.
        """
        pitch = self._param['pitch']
        gridStandoff = self._param['gridStandoff'] + self._param['gridThickness']/2.
        cathodeHeight = self._param['cathodeHeight'] + self._param['gridThickness']/2.
        gasHeight = cathodeHeight + gridStandoff
        yLength = pitch
        
        # Check unit cell shape
        if self._unitCell == 'hexagon':
            xLength = pitch*math.sqrt(3)
            # Create a volume object for the gas
            gasBox = self._createHexagon(
                xLength/3, -gridStandoff, gasHeight
            )
            
            # Create the cathode surface
            cathodeSurface = self._createHexagon(
                xLength/3, cathodeHeight
            )
            
        else:
            xLength = pitch
            # Create a volume object for the gas
            gasBox = self._occ.addBox(
                -xLength/2, -yLength/2, -gridStandoff,
                xLength, yLength, gasHeight
            )
            
            # Create the cathode surface
            cathodeSurface = self._occ.addRectangle(
                -xLength/2, -yLength/2, cathodeHeight,
                xLength, yLength
            )
        
        match self._scale:
            case 'single':
                adjustedGas = [(3, gasBox)]
                adjustedCathode = [(2, cathodeSurface)]

            case 'corner':
                # Create corner gas
                self._occ.remove([(3, gasBox)], recursive=True)
                cornerGas = self._occ.addBox(
                    0, 0, -gridStandoff,
                    xLength/2, yLength/2, gasHeight
                )
                adjustedGas = [(3, cornerGas)]
                
                # Create corner cathode
                self._occ.remove([(2, cathodeSurface)], recursive=True)
                cornerCathode = self._occ.addRectangle(
                    0, 0, cathodeHeight,
                    xLength/2, yLength/2
                )
                adjustedCathode = [(2, cornerCathode)]

            case 'surrounding':
                # Create surrounding gas
                adjustedGas = self._copyToSurrounding((3, gasBox))
                adjustedCathode = self._copyToSurrounding((2, cathodeSurface))
            
            case 'half':
                # Create half gas
                self._occ.remove([(3, gasBox)], recursive=True)
                halfBox = self._occ.addBox(
                    -xLength/2, -yLength, -gridStandoff,
                    xLength, 2*yLength, gasHeight
                )
                adjustedGas = [(3, halfBox)]
                
                # Create half cathode
                self._occ.remove([(2, cathodeSurface)], recursive=True)
                halfCathode = self._occ.addRectangle(
                    -xLength/2, -yLength, cathodeHeight,
                    xLength, 2*yLength
                )
                adjustedCathode = [(2, halfCathode)]
                                
        # Remove non-gas volumes from the gas box
        gasVolume, _ = self._occ.cut(
            adjustedGas, 
            [(3, dielectricVolume[0][1]), (3, gridVolume[0][1])], 
            removeObject=True, removeTool=False
        )
    
        return gasVolume, adjustedCathode

#**********************************************************************#
    
    def _copyToSurrounding(self, geometry, fuse=True):
        """
        Takes a geometry element and copies it to the surrounding cells.
        
        Note: removes original geometry element and replaces it with the
        fused, full geometry.
        
        args:
            geometry: geometry element to be duplicated.
            fuse (bool): choose to fuse elements together or not.
        Note: input format should be (#, name)
        
        returns:
            fullGeometry: the geometry elements post duplication.
        Note: given as a list of elements if fuse = False.
        """
        if self._unitCell == 'hexagon':
            xLength = self._param['pitch']*math.sqrt(3)
            yLength = self._param['pitch']
            neighborCenters = [
                (xLength/2, yLength/2), #Top-Right
                (xLength/2, -yLength/2), #Bottom-Right
                (0, -yLength), #Bottom
                (-xLength/2, -yLength/2), #Bottom-Left
                (-xLength/2, yLength/2), #Top-Left
                (0, yLength) #Top
            ]
        else:
            xLength = self._param['pitch']
            yLength = self._param['pitch']
            neighborCenters = [
                (0, yLength), # Top
                (xLength, yLength), # Top-Right
                (xLength, 0), # Right
                (xLength, -yLength), # Bottom-Right
                (0, -yLength), # Bottom
                (-xLength, -yLength), #Bottom-Left
                (-xLength, 0), # Left
                (-xLength, yLength) # Top-Left
            ]
        
        # Create and fuse surrounding geometry
        fullGeometry = []
        for x, y in neighborCenters:
            newGeo = self._occ.copy([geometry])
            self._occ.translate(newGeo, x, y, 0)                
            if fuse:
                if fullGeometry:
                    fullGeometry, _ = self._occ.fuse(
                        fullGeometry,
                        newGeo,
                        removeObject=True, removeTool=True
                    )
                else:
                    fullGeometry = newGeo
            
            else:
                fullGeometry.extend(newGeo)
        
        # Deal with original geometry element
        if fuse:    
            # Fuse surrounding with the original
            fullGeometry, _ = self._occ.fuse(
                fullGeometry,
                [geometry],
                removeObject=True, removeTool=True
            )
        else:
            # Append original to list
            fullGeometry.append(geometry)
        return fullGeometry

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
        # Dielectric
        dielectricVolume, padVolumes = self._buildDielectric()
            
        # Grid
        gridVolume = self._buildGrid()
        
        # Gas
        gasVolume, cathodeSurface = self._buildGas(gridVolume, dielectricVolume)
        
        cellParts = {
            'Gas': (3, gasVolume[0][1]),
            'Dielectric': (3, dielectricVolume[0][1]),
            'Grid': (3, gridVolume[0][1]),
            'CenterPad': padVolumes[0]
        }
        
        if self._scale == 'surrounding':
            cellParts.update({
                'TopPad': padVolumes[1],
                'RightTopPad': padVolumes[2],
                'RightPad': padVolumes[3],
                'RightBottomPad': padVolumes[4],
                'BottomPad': padVolumes[5],
                'LeftBottomPad': padVolumes[6],
                'LeftPad': padVolumes[7],
                'LeftTopPad': padVolumes[8]
            })
        
        cellParts['Cathode'] = cathodeSurface[0] # Cathode must be listed last for Elmer FEM unpacking
        
        return cellParts
    
#**********************************************************************#

    def _buildHexagonalCell(self):
        """
        Builds the geometry for a single, hexagonal unit cell.

        Note: Pillars are currently not included in the geometry.

        Returns:
            A dictionary containing the following parts of the unit cell:
                Gas: The gas volume in the unit cell.
                Dielectric: The dielectric volume in the unit cell.
                Grid: The grid volume in the unit cell.
                CenterPad: The center pad surface in the unit cell.
                RightTopPad: The corner pad surface in the unit cell.
                Cathode: The cathode surface in the unit cell.
        """
        # Dielectric
        dielectricVolume, padVolumes = self._buildDielectric()
        
        # Grid
        gridVolume = self._buildGrid()
        
        # Gas
        gasVolume, cathodeSurface = self._buildGas(gridVolume, dielectricVolume)
      
        cellParts = {
            'Gas': (3, gasVolume[0][1]),
            'Dielectric': (3, dielectricVolume[0][1]),
            'Grid': (3, gridVolume[0][1]),
            'CenterPad': padVolumes[0]
        }
        
        match self._scale:
            case 'surrounding':
                cellParts.update({
                    'RightTopPad': padVolumes[1],
                    'RightBottomPad': padVolumes[2],
                    'BottomPad': padVolumes[3],
                    'LeftBottomPad': padVolumes[4],
                    'LeftTopPad': padVolumes[5],
                    'TopPad': padVolumes[6]
                })
            
            case 'single':
                pass
            
            case 'corner':
                cellParts['RightTopPad'] = padVolumes[1]

        cellParts['Cathode'] = cathodeSurface[0] # Cathode must be listed last for Elmer FEM unpacking
        
        return cellParts

#**********************************************************************#

    def _makeCell(self):
        """
        Constructs a complete unit cell structure.

        returns:
            entityMap: map of geometry objects and their values
        """
        allObjects = []
        match self._unitCell:
            case 'square':
                inCell = self._buildSquareCell()
            
            case 'hexagon':
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

    def _createTriangle(self, outRadius, z, zDist=None, angle=30):
        """
        Makes an equilateral triangle in the xy-plane with the center at the origin.
        Extrudes it in the z-direction if zDist is provided.

        Args:
            sideLength (float): The side length of the triangle.
            z (float): The z-coordinate of the triangle.
            zDist (float): The distance to extrude the triangle in the z-direction. 
        If None, the triangle will remain a 2D surface.
            angle (float): Rotation angle of the triangle. Defaults to point in the positive z-direction.
        
        returns:
            surface: geometry object
        """
        points = []
        for i in range(3):
            netAngle = math.radians(i*120 - angle)
            x = outRadius*math.cos(netAngle)
            y = outRadius*math.sin(netAngle)
            
            inPoint = self._occ.addPoint(x, y, z)
            points.append(inPoint)

        lines = []
        for i in range(3):
            inLine = self._occ.addLine(points[i], points[(i+1)%3])
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
            'CentralPad', 'RightTopPad', 'RightBottomPad', 
            'BottomPad', 'LeftBottomPad', 
            'LeftTopPad', 'TopPad'
        ]
        altHexPads = ['CentralPad', 'RightTopPad']

        allSquarePads = [
            'CentralPad', 'TopPad', 'RightTopPad',
            'RightPad', 'RightBottomPad', 'BottomPad', 
            'LeftBottomPad', 'LeftPad', 'LeftTopPad'
        ]
        
        allVolumes = ['Gas', 'Dielectric', 'Grid']
        otherSurfaces = ['Cathode']

        configuration = {
            'squarecorner': {
                'keys': allVolumes + ['CentralPad'] + otherSurfaces,
                'pads': ['CentralPad']
            },
            
            'squaresingle': {
                'keys': allVolumes + ['CentralPad'] + otherSurfaces,
                'pads': ['CentralPad']
            },
            
            'squarehalf': {
                'keys': allVolumes + allSquarePads + otherSurfaces,
                'pads': allSquarePads
            },
            
            'squaresurrounding': {
                'keys': allVolumes + allSquarePads + otherSurfaces,
                'pads': allSquarePads
            },
            
            'hexagoncorner': {
                'keys': allVolumes + altHexPads + otherSurfaces,
                'pads': altHexPads
            },
            
            'hexagonsingle': {
                'keys': allVolumes + ['centralPad'] + otherSurfaces,
                'pads': ['centralPad']
            },
            
            'hexagonhalf': {
                'keys': allVolumes + allHexPads + otherSurfaces,
                'pads': allHexPads
            },
            
            'hexagonsurrounding': {
                'keys': allVolumes + allHexPads + otherSurfaces,
                'pads': allHexPads
            }
        }

        runOption = self._unitCell + self._scale
            
        config = configuration[runOption]

        isHex = 'hexagon' in runOption

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
        
        # Individual pads
        for tags, name in zip(padTags, padNames):
            gmsh.model.addPhysicalGroup(3, tags, name=name)

        # Surfaces
        if globalGroup['Cathode']:
            gmsh.model.addPhysicalGroup(2, globalGroup['Cathode'], name='Cathode')
        if allGridSurfaces:
            gmsh.model.addPhysicalGroup(2, list(set(allGridSurfaces)), name='Grid')

        # Individual pads
        for tags, name in zip(padTags, padNames):
            allPadSurfaces = []
            validVol = [(3, t) for t in tags if gmsh.model.occ.getEntities(3).count((3, t)) > 0]
            if validVol:
                boundary = gmsh.model.getBoundary(validVol, oriented=False)
                allPadSurfaces.extend([b[1] for b in boundary])
            gmsh.model.addPhysicalGroup(2, list(set(allPadSurfaces)), name=name)
            
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
        driftLength = self._param['cathodeHeight'] + gridThickness/2.
        sqrt3 = math.sqrt(3)
        
        # List of coordinates for each refinement line point in a specified geometry
        refinementOptions = {
            'squarecorner': [
                (0, 0, driftLength),
                (pitch/2, 0, driftLength), 
                (pitch/2, pitch/2, driftLength),
                (0, pitch/2, driftLength)
            ],
            
            'squaresingle': [
                (-pitch/4, -pitch/4, driftLength),
                (pitch/4, -pitch/4, driftLength), 
                (pitch/4, pitch/4, driftLength),
                (-pitch/4, pitch/4, driftLength)
            ],
            
            'squarehalf': [
                (-pitch/2, -pitch/2, driftLength),
                (pitch/2, -pitch/2, driftLength), 
                (pitch/2, pitch/2, driftLength),
                (-pitch/2, pitch/2, driftLength)
            ],
            
            'squaresurrounding': [
                (-pitch/2, -pitch/2, driftLength),
                (pitch/2, -pitch/2, driftLength), 
                (pitch/2, pitch/2, driftLength),
                (-pitch/2, pitch/2, driftLength)
            ],
            
            'hexagoncorner': [
                (0, 0, driftLength),
                (pitch/sqrt3, 0, driftLength), 
                (pitch/sqrt3/2, pitch/2, driftLength),
                (0, pitch/2, driftLength)
            ],
            
            'hexagonsingle': [
                (pitch/sqrt3/2, 0, driftLength), 
                (pitch/sqrt3/4, pitch/4, driftLength),
                (-pitch/sqrt3/4, pitch/4, driftLength),
                (-pitch/sqrt3/2, 0, driftLength),
                (-pitch/sqrt3/4, -pitch/4, driftLength),
                (pitch/sqrt3/4, -pitch/4, driftLength)
            ],
            
            'hexagonhalf': [
                (pitch/sqrt3, 0, driftLength), 
                (pitch/sqrt3/2, pitch/2, driftLength),
                (-pitch/sqrt3/2, pitch/2, driftLength),
                (-pitch/sqrt3, 0, driftLength),
                (-pitch/sqrt3/2, -pitch/2, driftLength),
                (pitch/sqrt3/2, -pitch/2, driftLength)
            ],
            
            'hexagonsurrounding': [
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
        # Cell dimensions
        pitch = self._param['pitch']
        holeRadius = self._param['holeRadius']
        gridThickness = self._param['gridThickness']
        thicknessSiO2 = self._param['thicknessSiO2']
        padLength = self._param['padLength']
        cathodeHeight = self._param['cathodeHeight']
        gridStandoff = self._param['gridStandoff']
        
        # Derived cell lengths
        driftLength = cathodeHeight + gridThickness/2.
        amplificationGap = gridStandoff + gridThickness/2.
        SiO2Height = thicknessSiO2 - gridStandoff - gridThickness/2.
        xLength = pitch*math.sqrt(3)
        yLength = pitch
        outRadius = xLength/3
        
        # FEM region scales
        smallRadius = min(holeRadius, padLength)
        largeRadius = max(holeRadius, padLength)
        htransitionWidth = pitch/10.
        vtransitionWidth = driftLength/10.
        refineRadius = (pitch-holeRadius)/2.
        
        # Assign the correct boundary limits to the FEM
        meshSettings = {
            'squarecorner': {
                'x': (0, pitch/2), 
                'y': (0, pitch/2)
            },
            
            'squaresingle': {
                'x': (-pitch/2, pitch/2), 
                'y': (-pitch/2, pitch/2)
            },
            
            'squarehalf': {
                'x': (-pitch, pitch), 
                'y': (-pitch, pitch)
            },
            
            'squaresurrounding': {
                'x': (-pitch, pitch), 
                'y': (-pitch, pitch)
            },
            
            'hexagoncorner': {
                'x': (0, xLength/2), 
                'y': (0, yLength/2)
            },
            
            'hexagonsingle': {
                'x': (-outRadius, outRadius), 
                'y': (-yLength/2, yLength/2)
            },
            
            'hexagonhalf': {
                'x': (-xLength/2, xLength/2), 
                'y': (-yLength/2, yLength/2)
            },
            
            'hexagonsurrounding': {
                'x': (-xLength/2, xLength/2), 
                'y': (-yLength/2, yLength/2)
            }
        }
        runOption = self._unitCell + self._scale
        bounds = meshSettings[runOption]
        
        # Create a line from the center of the pad to above the center hole
        pipeBottom = self._occ.addPoint(
            0, 0, -amplificationGap, 
        )
        pipeTop = self._occ.addPoint( 
            0, 0, driftLength/10.
        ) 
        amplificationLine = self._occ.addLine(pipeBottom, pipeTop)
        
        # Create lines for refinement around the top edge of the unit cell
        refinementLines = self._makeRefinementLines(runOption)
        
        self._occ.synchronize()
        
        #=========================#
        #=== DEFINE MESH SIZES ===#
        #=========================#
        fineMesh = 2#gridThickness*(3./4.)
        gridMesh = 2#gridThickness/4.
        refineMesh = 3#gridThickness*(3./2.)
        backgroundMesh = pitch/4.
        #=========================#
        
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
        
        # Define coarse mesh near edge/corner refinement lines
        gmsh.model.mesh.field.add('Threshold', 4)
        gmsh.model.mesh.field.setNumber(4, 'InField', 2)
        gmsh.model.mesh.field.setNumber(4, 'SizeMin', refineMesh)
        gmsh.model.mesh.field.setNumber(4, 'SizeMax', backgroundMesh)
        gmsh.model.mesh.field.setNumber(4, 'DistMin', refineRadius)
        gmsh.model.mesh.field.setNumber(4, 'DistMax', refineRadius + vtransitionWidth)
        
        # Define ultra fine mesh around the entire grid
        gmsh.model.mesh.field.add('Box', 5)
        gmsh.model.mesh.field.setNumber(5, 'VIn', gridMesh)
        gmsh.model.mesh.field.setNumber(5, 'VOut', backgroundMesh)
        gmsh.model.mesh.field.setNumber(5, 'XMin', bounds['x'][0])
        gmsh.model.mesh.field.setNumber(5, 'XMax', bounds['x'][1])
        gmsh.model.mesh.field.setNumber(5, 'YMin', bounds['y'][0])
        gmsh.model.mesh.field.setNumber(5, 'YMax', bounds['y'][1])
        gmsh.model.mesh.field.setNumber(5, 'ZMin', -gridThickness/2.)
        gmsh.model.mesh.field.setNumber(5, 'ZMax', gridThickness/2.)
        gmsh.model.mesh.field.setNumber(5, 'Thickness', vtransitionWidth/4.)
        
        # Define coarse mesh as a transition region around the grid
        gmsh.model.mesh.field.add('Box', 6)
        gmsh.model.mesh.field.setNumber(6, 'VIn', refineMesh)
        gmsh.model.mesh.field.setNumber(6, 'VOut', backgroundMesh)
        gmsh.model.mesh.field.setNumber(6, 'XMin', bounds['x'][0])
        gmsh.model.mesh.field.setNumber(6, 'XMax', bounds['x'][1])
        gmsh.model.mesh.field.setNumber(6, 'YMin', bounds['y'][0])
        gmsh.model.mesh.field.setNumber(6, 'YMax', bounds['y'][1])
        gmsh.model.mesh.field.setNumber(6, 'ZMin', -gridThickness*2.)
        gmsh.model.mesh.field.setNumber(6, 'ZMax', gridThickness*2.)
        gmsh.model.mesh.field.setNumber(6, 'Thickness', vtransitionWidth)
        
        # Define fine mesh around the pad
        gmsh.model.mesh.field.add('Box', 7)
        gmsh.model.mesh.field.setNumber(7, 'VIn', refineMesh)
        gmsh.model.mesh.field.setNumber(7, 'VOut', backgroundMesh)
        gmsh.model.mesh.field.setNumber(7, 'XMin', bounds['x'][0])
        gmsh.model.mesh.field.setNumber(7, 'XMax', bounds['x'][1])
        gmsh.model.mesh.field.setNumber(7, 'YMin', bounds['y'][0])
        gmsh.model.mesh.field.setNumber(7, 'YMax', bounds['y'][1])
        gmsh.model.mesh.field.setNumber(7, 'ZMin', SiO2Height - thicknessSiO2/2.)
        gmsh.model.mesh.field.setNumber(7, 'ZMax', SiO2Height + thicknessSiO2/2.)
        gmsh.model.mesh.field.setNumber(7, 'Thickness', vtransitionWidth/2.)
        
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
        self._scale = geoConfig['scale']
        self._holeShape = geoConfig['holeShape']
        self._padShape = geoConfig['padShape']
        
        filePath = 'Geometry'
        filename = os.path.join(filePath, f'{self._unitCell}{self._scale}.msh')

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()

        gmsh.option.setNumber('Mesh.ElementOrder', 2)
        gmsh.option.setNumber('Mesh.HighOrderOptimize', 1)
        gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary', 1)
        gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 1)

        gmsh.model.add(filename)
        
        # Create geometry
        allCellsMap = self._makeCell()     
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
        
        # TODO: Currently not generating pillars
        self._genPillars = False
        
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
            case 'squarecorner':
                pass
            
            case 'squaresingle':
                pass
            case 'squarehalf':
                self._electrodeMap.update({
                    4: 'TopPad', 5: 'RightTopPad', 6: 'RightPad',
                    7: 'RightBottomPad', 8: 'BottomPad', 9: 'LeftBottomPad',
                    10: 'LeftPad', 11: 'LeftTopPad'
                })
                
            case 'squaresurrounding':
                self._electrodeMap.update({
                    4: 'TopPad', 5: 'RightTopPad', 6: 'RightPad',
                    7: 'RightBottomPad', 8: 'BottomPad', 9: 'LeftBottomPad',
                    10: 'LeftPad', 11: 'LeftTopPad'
                })

            case 'hexagoncorner':
                self._electrodeMap.update({4: 'RightTopPad'})

            case 'hexagonsingle':
                pass
            
            case 'hexagonhalf':
                self._electrodeMap.update({
                    4: 'RightTopPad', 5: 'RightBottomPad',
                    6: 'BottomPad', 7: 'LeftBottomPad',
                    8: 'LeftTopPad', 9: 'TopPad'
                })

            case 'hexagonsurrounding':
                self._electrodeMap.update({
                    4: 'RightTopPad', 5: 'RightBottomPad',
                    6: 'BottomPad', 7: 'LeftBottomPad',
                    8: 'LeftTopPad', 9: 'TopPad'
                })

            case _:
                raise ValueError('Invalid run option.')
        
        # Create list of pad names for volume allotment
        self._padList = [self._electrodeMap[key] for key in self._electrodeMap if key not in [1, 2]] 
        
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
                'Relative Permittivity': '1e10'
            },
            {
                'Name': '"SiO2"', 
                'Relative Permittivity': '3.9'
            }
        ]
        if self._genPillars:
            allMaterials.append({
                'Name': '"Pillars"',
                'Relative Permittivity': '3.0'
            })
        
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
            }
        ]
        
        volNum = 4
        for padName in self._padList:
            allBodies.append({
                'Target Bodies(1)': volNum,
                'Name': f'"{padName}"',
                'Equation': 1,
                'Material': 2
            })
            volNum += 1
        
        if self._genPillars:
            allBodies.append({
                'Target Bodies(1)': len(allBodies)+1, 
                'Name': '"Pillars"', 
                'Equation': 1, 
                'Material': 4
            })
        
        self._materials = {}
        self._bodies = {}

        for i in range(len(allMaterials)):
            self._materials[f'Material {i+1}'] = allMaterials[i]
        for i in range(len(allBodies)):
            self._bodies[f'Body {i+1}'] = allBodies[i]

        self._makeDielectricsFile()

        return  
    
#**********************************************************************#

    def _makeDielectricsFile(self):
        """Writes the dielectric properties to a file."""

        dielectricValues = ['1e10', '3.9', '1.0']
        
        # Include Pad volumes
        for pad in self._padList:
            dielectricValues.append('1e10')
        
        # Optionally include pillars
        if self._genPillars:
            dielectricValues.append('3.0')
        
        
        try:
            with open('Geometry/dielectrics.dat', 'w') as f:
                f.write(len(dielectricValues).__str__() + '\n')

                for i in range(len(dielectricValues)):
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
            'CentralPad', 'RightTopPad', 
            'TopPad', 'BottomPad',
            'RightBottomPad', 'LeftTopPad',
            'LeftBottomPad'
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






