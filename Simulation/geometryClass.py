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

        self._runOption = None

        self._runGUI = False

        self._unitCell = 'FIMS'
        self._surroundingCells = False

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
    def setGUI(self, runGUI=True):
        """
        Sets whether the Gmsh GUI runs when creating the geometry.
        """
        self._runGUI = runGUI

        return
    
#**********************************************************************#
    def setUnitCell(self, unitCell='FIMS'):
        """
        Sets the unit cell type.

        Options are:
            - FIMS: A single unit cell of the FIMS geometry.
            - FIMSHexagonal: A single hexagonal unit cell of the FIMS geometry.
            - GridPix: A single unit cell of the GridPix geometry.

        Note: FIMS and GridPix require mirroring. FIMSHexagonal is standalone.
        """

        cellOptions = ['FIMS', 'FIMSHexagonal', 'GridPix']

        if unitCell not in cellOptions:
            raise ValueError(f'Error - Invalid unit cell type. Must be one of {cellOptions}.')
        
        self._unitCell = unitCell

        return


#**********************************************************************#
    def setSurroundingCells(self, surrounding=False):
        """
        Sets whether to include the surrounding cells in the geometry.

        In hexagonal geometry, this will be the entirety of each cell.
        For FIMS, this will only be half of each cell.
        """

        self._surroundingCells = surrounding

        return

#**********************************************************************#
    def buildGeometry(self):
        """
        Builds the geometry for the FIMS simulation using Gmsh.
        """

        print('\tBuilding geometry...')
    
        self._checkParameters()
        self._gmshClass = gmshClass(self._param)

        self._runOption = self._unitCell
        if self._surroundingCells:
            self._runOption += 'Surrounding'

        self._gmshClass.generateMesh(
            runOption=self._runOption,
            runGUI=self._runGUI
        )

        return
    
#**********************************************************************#
    def _generateElmerFiles(self, capacitance=False):
        """
        Generates the SIF files for Elmer based on the created geometry.
        """

        self._elmerClass = elmerClass(
            self._runOption, capacitance=False
        )

        if capacitance:
            self._elmerClassCapacitance = elmerClass(
                self._runOption, capacitance=True
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

        return
    
#**********************************************************************#
    def _buildFIMS(self):
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
        holeRadius = self._param['holeRadius']
        padLength = self._param['padLength']
        gridStandoff = self._param['gridStandoff']
        cathodeHeight = self._param['cathodeHeight']
        gridThickness = self._param['gridThickness']
        thicknessSiO2 = self._param['thicknessSiO2']
        pillarRadius = self._param['pillarRadius']

        xLength = pitch*math.sqrt(3)/2
        yLength = pitch/2

        outRadius = pitch/math.sqrt(3)

        ## Dielectric
        dielectricBox = self._occ.addBox(
            0, 0, -gridStandoff, 
            xLength, yLength, thicknessSiO2
        )
        # Define holes for the pads
        centerPadHole = self._createHexagon(
            padLength, -gridStandoff, thicknessSiO2
        )
        cornerPadHole = self._occ.copy([(3, centerPadHole)])
        self._occ.translate(cornerPadHole, xLength, yLength, 0)
        #Cut holes in dielectric
        dielectricVolume, _ = self._occ.cut(
            [(3, dielectricBox)],
            [(3, centerPadHole), cornerPadHole[0]]
        )

        ## Grid
        gridBox = self._occ.addBox(
            0, 0, -gridThickness/2,
            xLength, yLength, gridThickness
        )
        centerGridHole = self._occ.addCylinder(
            0, 0, -gridThickness/2,
            0, 0, gridThickness,
            holeRadius
        )
        cornerGridHole = self._occ.copy([(3, centerGridHole)])
        self._occ.translate(cornerGridHole, xLength, yLength, 0)
        gridVolume, _ = self._occ.cut(
            [(3, gridBox)],
            [(3, centerGridHole), cornerGridHole[0]]
        )

        
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

        ## Pads
        centerPadFull = self._createHexagon(
            padLength, -gridStandoff
        )
        cornerPadFull = self._occ.copy([(2, centerPadFull)])
        self._occ.translate(cornerPadFull, xLength, yLength, 0)

        padCutBox = self._occ.addBox(
            0, 0, -gridStandoff,
            xLength, yLength, 1.0
        )

        centerPadSurface, _ = self._occ.intersect(
            [(2, centerPadFull)],
            [(3, padCutBox)],
            removeObject=True, removeTool=False
        )
        cornerPadSurface, _ = self._occ.intersect(
            [(2, cornerPadFull[0][1])],
            [(3, padCutBox)],
            removeObject=True, removeTool=True
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
            'CenterPad': centerPadSurface[0],
            'CornerPad': cornerPadSurface[0],
            'Cathode': (2, cathodeSurface)
        }

        return cellParts
    
#**********************************************************************#
    def _buildFIMSSurrounding(self):
        """
        """

        pitch = self._param['pitch']
        holeRadius = self._param['holeRadius']
        padLength = self._param['padLength']
        gridStandoff = self._param['gridStandoff']
        cathodeHeight = self._param['cathodeHeight']
        gridThickness = self._param['gridThickness']
        thicknessSiO2 = self._param['thicknessSiO2']
        pillarRadius = self._param['pillarRadius']

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

        ## Dielectric
        dielectricBox = self._occ.addBox(
            -1*xLength, -1*yLength, -gridStandoff, 
            2*xLength, 2*yLength, thicknessSiO2
        )

        # Define holes for the pads
        centerPadHole = self._createHexagon(
            padLength, -gridStandoff, thicknessSiO2
        )
        padHoleTools = [(3, centerPadHole)]
        for x, y in neighborCenters:
            newHole = self._occ.copy([(3, centerPadHole)])
            self._occ.translate(newHole, x, y, 0)
            padHoleTools.extend(newHole)
        
        # Cut holes in dielectric
        dielectricVolume, _ = self._occ.cut(
            [(3, dielectricBox)],
            padHoleTools
        )

        ## Grid
        gridBox = self._occ.addBox(
            -1*xLength, -1*yLength, -gridThickness/2, 
            2*xLength, 2*yLength, gridThickness
        )
        # Define holes
        centerGridHole = self._occ.addCylinder(
            0, 0, -gridThickness/2,
            0, 0, gridThickness,
            holeRadius
        )
        gridHoleTools = [(3, centerGridHole)]
        for x, y in neighborCenters:
            newHole = self._occ.copy([(3, centerGridHole)])
            self._occ.translate(newHole, x, y, 0)
            gridHoleTools.extend(newHole)

        # Cut holes in grid
        gridVolume, _ = self._occ.cut(
            [(3, gridBox)],
            gridHoleTools
        )

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

        ## Pads
        centerPadFull = self._createHexagon(
            padLength, -gridStandoff
        )

        padList = [(2, centerPadFull)]
        for x, y in neighborCenters:

            newPad = self._occ.copy([(2, centerPadFull)])
            self._occ.translate(newPad, x, y, 0)
            padList.extend(newPad)

        padCutBox = self._occ.addBox(
            -xLength, -yLength, -gridStandoff,
            2*xLength, 2*yLength, 1.0
        )

        padSurfaces = []
        for pad in padList:
            padSurface, _ = self._occ.intersect(
                [pad],
                [(3, padCutBox)],
                removeObject=True, removeTool=False
            )
            padSurfaces.append(padSurface[0])
        

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

    def _makeFIMS(self, surrounding=False):
        """
        TODO
        """

        allCellData = []
        allObjects = []

        if surrounding:
            inCell = self._buildFIMSSurrounding()
        else:
            inCell = self._buildFIMS()

        allCellData.append(inCell)
        allObjects.extend(inCell.values())

        _, entityMap = self._occ.fragment(allObjects, [])
        self._occ.synchronize()

        return allCellData, entityMap

#**********************************************************************#

    def _buildHexagonalFIMSCell(self):
        """
        build the geometry for a single hexagonal unit cell 
        of the FIMS geometry.
        
        Returns:
            A dictionary containing the following parts of the unit cell:
                Gas: The gas volume in the unit cell.
                Dielectric: The dielectric volume in the unit cell.
                Grid: The grid volume in the unit cell.
                Pad: The pad surface in the unit cell.
                Cathode: The cathode surface in the unit cell.
        """
        pitch = self._param['pitch']
        holeRadius = self._param['holeRadius']
        padLength = self._param['padLength']
        gridStandoff = self._param['gridStandoff']
        cathodeHeight = self._param['cathodeHeight']
        gridThickness = self._param['gridThickness']
        thicknessSiO2 = self._param['thicknessSiO2']

        outRadius = pitch/math.sqrt(3)

        ## Dielectric
        hexagonBase = self._createHexagon(
            outRadius, -gridStandoff, thicknessSiO2
        )
        padHole = self._createHexagon(
            padLength, -gridStandoff, thicknessSiO2
        )
        dielectricVolume, _ = self._occ.cut(
            [(3, hexagonBase)],
            [(3, padHole)]
        )

        ## Grid
        gridContainer = self._createHexagon(
            outRadius, -gridThickness/2, gridThickness
        )
        gridHole = self._occ.addCylinder(
            0, 0, -gridThickness/2,
            0, 0, gridThickness,
            holeRadius        
        )
        gridVolume, _ = self._occ.cut(
            [(3, gridContainer)],
            [(3, gridHole)]
        )

        ## Gas
        gasHeight = cathodeHeight + gridStandoff
        gasContainer = self._createHexagon(
            outRadius, -gridStandoff, gasHeight
        )
        gasVolume, _ = self._occ.cut(
            [(3, gasContainer)], 
            [(3, dielectricVolume[0][1]), (3, gridVolume[0][1])], 
            removeObject=True, removeTool=False
        )
        
        padSurface = self._createHexagon(
            padLength, -gridStandoff
        )
        cathodeSurface = self._createHexagon(
            outRadius, cathodeHeight
        )

        cellParts = {
            'Gas': (3, gasVolume[0][1]),
            'Dielectric': (3, dielectricVolume[0][1]),
            'Grid': (3, gridVolume[0][1]),
            'Pad': (2, padSurface),
            'Cathode': (2, cathodeSurface)
        }

        return cellParts
    
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
    def _makeFIMSHexagonal(self, surrounding=False):
        """
        builds a hexagonal array of unit cells.

        Central cell is always created.
        The 6 adjacent cells are created if surrounding is True.

        Args:
            surrounding (bool): Option to include the surrounding cells.

        Returns:
            A tuple containing:
                allCellData - A list of dictionaries, 
                            where each dictionary contains
                            the parts of a unit cell.
                entityMap - A mapping from the original entities 
                            to the new entities after fragmentation.
        """

        pitch = self._param['pitch']
        xShift = pitch*math.sqrt(3)/2.0
        yShift = pitch/2.0

        cellCenters = [(0.0, 0.0)]
        if surrounding:
            cellCenters = [
                (0.0, 0.0), #Center
                (0.0, pitch), #Top
                (0.0, -pitch), #Bottom
                (xShift, yShift), #Top-Right
                (xShift, -yShift), #Bottom-Right
                (-xShift, yShift), #Top-Left
                (-xShift, -yShift) #Bottom-Left
            ]

        allCellData = []
        allObjects = []

        for dx, dy in cellCenters:

            inCell = self._buildHexagonalFIMSCell()

            for inPart in inCell.values():
                self._occ.translate([inPart], dx, dy, 0)

            allCellData.append(inCell)
            allObjects.extend(inCell.values())

        _, entityMap = self._occ.fragment(allObjects, [])
        self._occ.synchronize()

        return allCellData, entityMap
    

#**********************************************************************#
    def _buildGridPix(self):
        """
        Builds the geometry for a single unit cell of the gridPix geometry.

        Returns:
            A dictionary containing the following parts of the unit cell:
                Gas: The gas volume in the unit cell.
                Dielectric: The dielectric volume in the unit cell.
                Grid: The grid volume in the unit cell.
                CenterPad: The center pad surface in the unit cell.
                Cathode: The cathode surface in the unit cell.
        """

        pitch = self._param['pitch']
        holeRadius = self._param['holeRadius']
        padLength = self._param['padLength']
        gridStandoff = self._param['gridStandoff']
        cathodeHeight = self._param['cathodeHeight']
        gridThickness = self._param['gridThickness']
        thicknessSiO2 = self._param['thicknessSiO2']
        pillarRadius = self._param['pillarRadius']

        xLength = pitch/2
        yLength = pitch/2

        ## Dielectric
        dielectricBox = self._occ.addBox(
            0, 0, -gridStandoff, 
            xLength, yLength, thicknessSiO2
        )
        # Define holes for the pads
        centerPadHole = self._createOctagon(
            padLength, -gridStandoff, thicknessSiO2
        )
        #Cut holes in dielectric
        dielectricVolume, _ = self._occ.cut(
            [(3, dielectricBox)],
            [(3, centerPadHole)]
        )

        ## Grid
        gridBox = self._occ.addBox(
            0, 0, -gridThickness/2,
            xLength, yLength, gridThickness
        )
        centerGridHole = self._occ.addCylinder(
            0, 0, -gridThickness/2,
            0, 0, gridThickness,
            holeRadius
        )
        gridVolume, _ = self._occ.cut(
            [(3, gridBox)],
            [(3, centerGridHole)]
        )

        
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

        ## Pads
        centerPadFull = self._createOctagon(
            padLength, -gridStandoff
        )

        padCutBox = self._occ.addBox(
            0, 0, -gridStandoff,
            xLength, yLength, 1.0
        )

        centerPadSurface, _ = self._occ.intersect(
            [(2, centerPadFull)],
            [(3, padCutBox)],
            removeObject=True, removeTool=True
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
            'Pad': centerPadSurface[0],
            'Cathode': (2, cathodeSurface)
        }

        return cellParts
    
#**********************************************************************#

    def _makeGridPix(self):
        """
        TODO
        """

        allCellData = []
        allObjects = []


        inCell = self._buildGridPix()

        allCellData.append(inCell)
        allObjects.extend(inCell.values())

        _, entityMap = self._occ.fragment(allObjects, [])
        self._occ.synchronize()

        return allCellData, entityMap
    
#**********************************************************************#

    def _assignPhysicalGroups(self, entityMap, mode):
        """
        TODO
        """

        allPads = [
            'CentralPad', 'TopPad', 'BottomPad', 
            'RightTopPad', 'RightBottomPad', 
            'LeftTopPad', 'LeftBottomPad'
        ]
        altPads = ['CentralPad', 'CornerPad']

        allVolumes = ['Gas', 'Dielectric', 'Grid']
        otherSurfaces = ['Cathode']

        configuration = {
            'FIMS': {
                'keys': allVolumes + altPads + otherSurfaces,
                'pads': altPads
            },
            'FIMSSurrounding': {
                'keys': allVolumes + allPads + otherSurfaces,
                'pads': allPads
            },
            'FIMSHexagonal': {
                'keys': allVolumes + ['Pad'] + otherSurfaces,
                'pads': ['CentralPad']
            },
            'FIMSHexagonalSurrounding': {
                'keys': allVolumes + ['Pad'] + otherSurfaces,
                'pads': allPads
            },
            'GridPix': {
                'keys': allVolumes + ['Pad'] + otherSurfaces,
                'pads': ['CentralPad']
            }
        }


        config = configuration[mode]

        isHex = 'Hexagonal' in mode

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

    def _setMeshSizes(self, runOption):
        """
        TODO
        """

        # Cell dimensions
        pitch = self._param['pitch']
        holeRadius = self._param['holeRadius']

        xLength = pitch*math.sqrt(3)/2
        yLength = pitch/2
        outRadius = self._param['pitch'] / math.sqrt(3)

        meshSettings = {
            'FIMS': {
                'x': (0, xLength), 
                'y': (0, yLength)
            },
            'FIMSSurrounding': {
                'x': (-xLength, xLength), 
                'y': (-yLength, yLength)
            },
            'FIMSHexagonal': {
                'x': (-outRadius, outRadius), 
                'y': (-yLength, yLength)
            },
            'FIMSHexagonalSurrounding': {
                'x': (-outRadius, outRadius), 
                'y': (-yLength, yLength)
            },
            'GridPix': {
                'x': (0, yLength), 
                'y': (0, yLength)
            }
        }

        bounds = meshSettings[runOption]

        fineMesh = self._param['gridThickness']*2 #TODO - 1 here?
        courseMesh = fineMesh*5
        transitionWidth = pitch

        # Define a high-res cylinder between central hole and pad
        largerHole = max(
            self._param['holeRadius'],
            self._param['padLength']
        )
        pipeRadius = largerHole + self._param['gridThickness']
        
        pipeBottom = self._occ.addPoint(
            0, 0, -self._param['gridStandoff']
        )
        pipeTop = self._occ.addPoint(
            0, 0, self._param['holeRadius']/2
        ) 
        amplificationLine = self._occ.addLine(pipeBottom, pipeTop)
        self._occ.synchronize()

        # Find distance from line
        gmsh.model.mesh.field.add('Distance', 1)
        gmsh.model.mesh.field.setNumbers(1, 'EdgesList', [amplificationLine])
        
        # Define mesh size based on distance from line
        gmsh.model.mesh.field.add('Threshold', 2)
        gmsh.model.mesh.field.setNumber(2, 'InField', 1)
        gmsh.model.mesh.field.setNumber(2, 'SizeMin', fineMesh)
        gmsh.model.mesh.field.setNumber(2, 'SizeMax', courseMesh)
        gmsh.model.mesh.field.setNumber(2, 'DistMin', pipeRadius)
        gmsh.model.mesh.field.setNumber(2, 'DistMax', pipeRadius + transitionWidth)

        # Keep fine-ish mesh around the entire grid
        gmsh.model.mesh.field.add('Box', 3)
        gmsh.model.mesh.field.setNumber(3, 'VIn', 2*fineMesh)
        gmsh.model.mesh.field.setNumber(3, 'VOut', courseMesh)
        gmsh.model.mesh.field.setNumber(3, 'XMin', bounds['x'][0])
        gmsh.model.mesh.field.setNumber(3, 'XMax', bounds['x'][1])
        gmsh.model.mesh.field.setNumber(3, 'YMin', bounds['y'][0])
        gmsh.model.mesh.field.setNumber(3, 'YMax', bounds['y'][1])
        gmsh.model.mesh.field.setNumber(3, 'ZMin', -holeRadius/2)
        gmsh.model.mesh.field.setNumber(3, 'ZMax', holeRadius/2)
        gmsh.model.mesh.field.setNumber(3, 'Thickness', transitionWidth)

        # Use the smallest mesh size
        gmsh.model.mesh.field.add('Min', 4)
        gmsh.model.mesh.field.setNumbers(4, 'FieldsList', [2, 3])
        gmsh.model.mesh.field.setAsBackgroundMesh(4)

        # Final settings
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 12)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)

        return
    
#**********************************************************************#

    def _checkRunOption(self, runOption):
        """
        Checks that the run option is valid.
        """

        runOptions = [
            'FIMS',
            'FIMSSurrounding',
            'FIMSHexagonal',
            'FIMSHexagonalSurrounding',
            'GridPix'
        ]

        if runOption not in runOptions:
            raise ValueError(f'Option must be one of {runOptions}.')
        
        return

#**********************************************************************#

    def generateMesh(self, runOption, runGUI=False):
        """
        TODO
        """

        self._checkRunOption(runOption)

        filePath = 'Geometry'
        filename = os.path.join(filePath, f'{runOption}.msh')

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()

        gmsh.option.setNumber('Mesh.ElementOrder', 2)
        gmsh.option.setNumber('Mesh.HighOrderOptimize', 1)
        gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary', 1)
        gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 1)

        gmsh.model.add(filename)

        match runOption:
            case 'FIMS':
                _, allCellsMap = self._makeFIMS()

            case 'FIMSSurrounding':
                _, allCellsMap = self._makeFIMS(surrounding=True)
            
            case 'FIMSHexagonal':
                _, allCellsMap = self._makeFIMSHexagonal()

            case 'FIMSHexagonalSurrounding':
                _, allCellsMap = self._makeFIMSHexagonal(surrounding=True)

            case 'GridPix':
                _, allCellsMap = self._makeGridPix()

        self._assignPhysicalGroups(allCellsMap, runOption)
        self._setMeshSizes(runOption)

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
            case 'FIMS':
                self._electrodeMap.update({4: 'CornerPad'})

            case 'FIMSSurrounding':
                self._electrodeMap.update({
                    4: 'TopPad', 5: 'BottomPad',
                    6: 'RightTopPad', 7: 'RightBottomPad',
                    8: 'LeftTopPad', 9: 'LeftBottomPad'
                })

            case 'FIMSHexagonal' | 'GridPix':
                pass

            case 'FIMSHexagonalSurrounding':
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
            #Dont need weighting for cathode or grid
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
                'Linear System Convergence Tolerance': '1.0e-10',
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
        """
        Adds materials to the simulation.
        """

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
        """
        Writes the dielectric properties to a file.
        """

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
        """
        Writes the SIF weighing files for the Elmer simulation.
        """
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
        TODO
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

        """
        Resets the potentials for all electrodes in the SIF file.
        """

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






