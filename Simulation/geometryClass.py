#################################
# CLASS DEFINITION FOR GEOMETRY #
#################################
from __future__ import annotations
import time
import subprocess

import numpy as np
import os
import math
import gmsh

class geometryClass:


#**********************************************************************#
    def __init__(self, inputParam=None):

        self._param = inputParam
        self._checkParameters()

        return

#**********************************************************************#
    def _checkParameters(self):
        """
        Checks that the parameters in self._param are valid
        for creating the geometry.
        """

        neededParameters = [
            'pitch',
            'holeRadius',
            'padLength',
            'gridStandoff',
            'cathodeHeight',
            'gridThickness',
            'dielectricThickness',
            'pillarRadius',
            'driftField',
            'fieldRatio'
        ]

        if self._param is None:
            raise ValueError('Error - Invalid parameters.')

        for inParam in self._param:
            if inParam not in neededParameters:
                raise ValueError(f"Error - Missing '{inParam}'")
            if self._param[inParam] <= 0:
                raise ValueError(f"Error - '{inParam}' must be positive.")
            

        # Check geometric constraints
        if self._param['holeRadius'] >= self._param['pitch'] / math.sqrt(3):
            raise ValueError('Error - Hole larger than hexagon radius.')
        if self._param['padLength'] >= self._param['pitch'] / math.sqrt(3):
            raise ValueError('Error - Pad larger than hexagon radius.')
        if self._param['pillarRadius'] >= (self._param['pitch']/2 - self._param['padLength']*math.sqrt(3)/2):
            raise ValueError('Error - Pillar cannot fit.')
        return

        

#**********************************************************************#
    def createGeometry(self, runGUI=False, hexagonal=False, neighborCells=False):
        """TODO"""
    
        self._checkParameters()
        self._gmshClass = gmshClass(self._param)

        self._hexagonal = hexagonal
        self._neighborCells = neighborCells

        if self._hexagonal:
            self._fileName = self._gmshClass.createHexagonalGeometry(
                neighborCells=self._neighborCells, 
                runGUI=runGUI
            )
        else:
            self._fileName = self._gmshClass.createFIMSGeometry(
                runGUI=runGUI
            )

        return
    
#**********************************************************************#
    def _generateElmerFiles(self):
        """TODO"""

        if self._hexagonal:
            self._numPads = 7 if self._neighborCells else 1
        else:
            self._numPads = 2

        self._elmerClass = elmerClass(self.fileName, numPads=self._numPads, capacitance=False)
        self._elmerClass.writeAllSIF()

        if self._numPads == 7:
            self._elmerClassCapacitance = elmerClass(self.fileName, numPads=self._numPads, capacitance=True)
            self._elmerClassCapacitance.writeAllSIF()
        return
    
#**********************************************************************#

    def solveEFields(self, capacitance=False):

        self._generateElmerFiles()

        if capacitance:
            self._elmerClassCapacitance.runElmerSolver()
        else:
            self._setVoltages()
            self._elmerClass._runElmerSolver()

        return
    
    #**********************************************************************#

    def solveWeightingFields(self):

        
        self._elmerClass.runElmerWeighting()

        return

#**********************************************************************#

    def _setVoltages(self):

        gridVoltage, cathodeVoltage = self.findPotentials()

        self._elmerClass.resetPotentials()
        self._elmerClass._setPotential('Grid', gridVoltage)
        self._elmerClass._setPotential('Cathode', cathodeVoltage)

        return
        
#**********************************************************************#

    def findPotentials(self):
        """TODO"""

        MICRONTOCM = 1e-4
        driftField = self._param['driftField']
        fieldRatio = self._param['fieldRatio']

        amplificationField = driftField*fieldRatio

        driftGap = self._param['cathodeHeight']+self._param['gridThickness']/2
        amplificationGap = self._param['gridStandoff']-self._param['gridThickness']/2

        gridVoltage = amplificationField*amplificationGap*MICRONTOCM
        cathodeVoltage = driftField*driftGap*MICRONTOCM + gridVoltage

        return gridVoltage, cathodeVoltage






class gmshClass:
    """
    TODO
    """

#**********************************************************************#
    def __init__(self, inputParams=None):
        """
        Initializes the gmshClass instance with the given parameters.
        """
    
        self._occ = gmsh.model.occ
        self._param = inputParams

        return
#**********************************************************************#
    def _createUnitCell(self):
        """
        TODO
        """

        pitch = self._param['pitch']
        holeRadius = self._param['holeRadius']
        padLength = self._param['padLength']
        gridStandoff = self._param['gridStandoff']
        cathodeHeight = self._param['cathodeHeight']
        gridThickness = self._param['gridThickness']
        dielectricThickness = self._param['dielectricThickness']
        pillarRadius = self._param['pillarRadius']

        xLength = pitch*math.sqrt(3)/2
        yLength = pitch/2

        outRadius = pitch/math.sqrt(3)

        ## Dielectric
        dielectricBox = self._occ.addBox(
            0, 0, -gridStandoff, 
            xLength, yLength, dielectricThickness
        )
        # Define holes for the pads
        centerPadHole = self._makeHexagon(
            padLength, -gridStandoff, dielectricThickness
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

        '''
        ##Pillars
        xPillarFull = self._occ.addCylinder(
            outRadius, 0, -gridStandoff+dielectricThickness,
            0, 0, gridStandoff-dielectricThickness-gridThickness/2,
            pillarRadius
        )
        yPillarFull = self._occ.addCylinder(
            xLength - outRadius, yLength, -gridStandoff+dielectricThickness,
            0, 0, gridStandoff-dielectricThickness-gridThickness/2,
            pillarRadius
        )

        pillarCutBox = self._occ.addBox(
            0, 0, -gridStandoff, 
            xLength, yLength, gridStandoff
        )
        xPillar, _ = self._occ.intersect(
            [(3, xPillarFull)], [(3, pillarCutBox)], 
            removeObject=True, removeTool=False
        )
        yPillar, _ = self._occ.intersect(
            [(3, yPillarFull)], [(3, pillarCutBox)], 
            removeObject=True, removeTool=True
        )
        '''
        
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
        centerPadFull = self._makeHexagon(
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

    def _makeUnitCell(self):

        allCellData = []
        allObjects = []

        inCell = self._createUnitCell()

        allCellData.append(inCell)
        allObjects.extend(inCell.values())

        _, entityMap = self._occ.fragment(allObjects, [])
        self._occ.synchronize()

        return allCellData, entityMap

#**********************************************************************#

    def _createHexagonalUnitCell(self):
        """
        Create the geometry for a single hexagonal unit cell 
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
        dielectricThickness = self._param['dielectricThickness']

        outRadius = pitch/math.sqrt(3)

        ## Dielectric
        hexagonBase = self._makeHexagon(
            outRadius, -gridStandoff, dielectricThickness
        )
        padHole = self._makeHexagon(
            padLength, -gridStandoff, dielectricThickness
        )
        dielectricVolume, _ = self._occ.cut(
            [(3, hexagonBase)],
            [(3, padHole)]
        )

        ## Grid
        gridContainer = self._makeHexagon(
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
        gasContainer = self._makeHexagon(
            outRadius, -gridStandoff, gasHeight
        )
        gasVolume, _ = self._occ.cut(
            [(3, gasContainer)], 
            [(3, dielectricVolume[0][1]), (3, gridVolume[0][1])], 
            removeObject=True, removeTool=False
        )
        
        padSurface = self._makeHexagon(
            padLength, -gridStandoff
        )
        cathodeSurface = self._makeHexagon(
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
    def _makeHexagon(self, outRadius, z, zDist=None):
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
    def _makeHexagonalCells(self, neighborCells=False):
        """
        Creates a hexagonal array of unit cells.

        Central cell is always created.
        If neighborCells is True, the 6 adjacent cells are also created.

        Args:
            neighborCells: If True, also creates the 6 adjacent cells.

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
        if neighborCells:
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

            inCell = self._createHexagonalUnitCell()

            for inPart in inCell.values():
                self._occ.translate([inPart], dx, dy, 0)

            allCellData.append(inCell)
            allObjects.extend(inCell.values())

        _, entityMap = self._occ.fragment(allObjects, [])
        self._occ.synchronize()

        return allCellData, entityMap

#**********************************************************************#

    def _assignPhysicalGroups(self, entityMap):
        """
        """

        padNames = [
            'CentralPad', 'CornerPad'
        ]

        globalGroup = {
            'Gas': [], 
            'Dielectric': [], 
            'Grid': [], 
            'Cathode': []
        }
        allGridSurfaces = []
        padTags = []

        for i, fragments in enumerate(entityMap):
            partKey = [
                'Gas', 'Dielectric', 'Grid', 
                'CenterPad', 'CornerPad', 'Cathode'
            ]
            partType = partKey[i]
            
            # 1. Filter out invalid tags (tags <= 0)
            newTags = [f[1] for f in fragments if f[1] > 0]
            if not newTags:
                continue

            if 'Pad' in partType:
                padTags.append(newTags)
            else:
                globalGroup[partType].extend(newTags)
                
                if partType == 'Grid':
                    validGrid = []
                    for t in newTags:
                        if gmsh.model.occ.getEntities(3).count((3, t)) > 0:
                            validGrid.append((3, t))
                    
                    if validGrid:
                        gridBoundary = gmsh.model.getBoundary(validGrid, oriented=False)
                        allGridSurfaces.extend([b[1] for b in gridBoundary])

        # Assign Volumes
        for name in ['Grid', 'Dielectric', 'Gas']:
            if globalGroup[name]:
                gmsh.model.addPhysicalGroup(
                    3, 
                    globalGroup[name], 
                    name=name
                )

        # Assign Surfaces
        if globalGroup['Cathode']:
            gmsh.model.addPhysicalGroup(
                2, 
                globalGroup['Cathode'], 
                name='Cathode'
            )
        if allGridSurfaces:
            gmsh.model.addPhysicalGroup(
                2, 
                list(set(allGridSurfaces)),
                name='Grid'
            )
        # Assign Pads individually
        for i in range(min(len(padTags), len(padNames))):
            gmsh.model.addPhysicalGroup(
                2, 
                padTags[i], 
                name=padNames[i]
            )

        return
    
#**********************************************************************#
    def _assignPhysicalGroupsHexagon(self, entityMap):
        """
        Assigns physical groups to the entities in the geometry.

        Assumes that the entities in entityMap are ordered as follows:
            0: Gas
            1: Dielectric
            2: Grid
            3: Pad
            4: Cathode

        Args:
            entityMap: A mapping from the original entities to the 
                    new entities after fragmentation.
        """

        padNames = [
            'CentralPad', 'TopPad', 'BottomPad',
            'RightTopPad', 'RightBottomPad',
            'LeftTopPad', 'LeftBottomPad'
        ]

        globalGroup = {
            'Gas': [], 
            'Dielectric': [], 
            'Grid': [], 
            'Cathode': []
        }
        allGridSurfaces = []
        padTags = []
        numEntityPerCell = 5

        for i, fragments in enumerate(entityMap):
            partKey = ['Gas', 'Dielectric', 'Grid', 'Pad', 'Cathode']
            partType = partKey[i % numEntityPerCell]
            
            # 1. Filter out invalid tags (tags <= 0)
            newTags = [f[1] for f in fragments if f[1] > 0]
            if not newTags:
                continue

            if partType == 'Pad':
                padTags.append(newTags)
            else:
                globalGroup[partType].extend(newTags)
                
                if partType == 'Grid':
                    validGrid = []
                    for t in newTags:
                        if gmsh.model.occ.getEntities(3).count((3, t)) > 0:
                            validGrid.append((3, t))
                    
                    if validGrid:
                        gridBoundary = gmsh.model.getBoundary(validGrid, oriented=False)
                        allGridSurfaces.extend([b[1] for b in gridBoundary])

        # Assign Volumes
        for name in ['Grid', 'Dielectric', 'Gas']:
            if globalGroup[name]:
                gmsh.model.addPhysicalGroup(
                    3, 
                    globalGroup[name], 
                    name=name
                )

        # Assign Surfaces
        if globalGroup['Cathode']:
            gmsh.model.addPhysicalGroup(
                2, 
                globalGroup['Cathode'], 
                name='Cathode'
            )
        if allGridSurfaces:
            gmsh.model.addPhysicalGroup(
                2, 
                list(set(allGridSurfaces)),
                name='Grid'
            )
        # Assign Pads individually
        for i in range(min(len(padTags), len(padNames))):
            gmsh.model.addPhysicalGroup(
                2, 
                padTags[i], 
                name=padNames[i]
            )

        return
    
#**********************************************************************#

    def _setMeshSizes(self):
        #TODO - make this more robust and flexible for different geometries
        fineMesh = self._param['gridThickness']*2 #Should be 1?
        courseMesh = 5*fineMesh #Could be 10?

        transitionWidth = self._param['holeRadius']

        pitch = self._param['pitch']
        holeRadius = self._param['holeRadius']

        largerHole = max(self._param['holeRadius'], self._param['padLength'])
        pipeRadius = largerHole + self._param['gridThickness']

        xLength = pitch*math.sqrt(3)/2 + 5.0
        yLength = pitch/2 + 5.0
        
        # Center line below grid
        padCenter = self._occ.addPoint(0, 0, -self._param['gridStandoff'])
        holeCenter = self._occ.addPoint(0, 0, self._param['holeRadius']/2) 
        amplificationLine = self._occ.addLine(padCenter, holeCenter)
        self._occ.synchronize()

        # --- FIELD 1: Below the Grid ---
        gmsh.model.mesh.field.add('Distance', 1)
        gmsh.model.mesh.field.setNumbers(1, 'EdgesList', [amplificationLine])
        
        gmsh.model.mesh.field.add('Threshold', 2)
        gmsh.model.mesh.field.setNumber(2, 'InField', 1)
        gmsh.model.mesh.field.setNumber(2, 'SizeMin', fineMesh)
        gmsh.model.mesh.field.setNumber(2, 'SizeMax', courseMesh)
        gmsh.model.mesh.field.setNumber(2, 'DistMin', pipeRadius)
        gmsh.model.mesh.field.setNumber(2, 'DistMax', pipeRadius + transitionWidth)

        # --- FIELD 3: Above the Grid ---
        gmsh.model.mesh.field.add('Box', 3)
        gmsh.model.mesh.field.setNumber(3, 'VIn', 2*fineMesh)
        gmsh.model.mesh.field.setNumber(3, 'VOut', courseMesh)
        gmsh.model.mesh.field.setNumber(3, 'XMin', 0)
        gmsh.model.mesh.field.setNumber(3, 'XMax', xLength)
        gmsh.model.mesh.field.setNumber(3, 'YMin', 0)
        gmsh.model.mesh.field.setNumber(3, 'YMax', yLength)
        gmsh.model.mesh.field.setNumber(3, 'ZMin', -holeRadius/2)
        gmsh.model.mesh.field.setNumber(3, 'ZMax', holeRadius)
        gmsh.model.mesh.field.setNumber(3, 'Thickness', transitionWidth)

        # --- FIELD 4: Combine ---
        gmsh.model.mesh.field.add('Min', 4)
        gmsh.model.mesh.field.setNumbers(4, 'FieldsList', [2, 3])
        
        gmsh.model.mesh.field.setAsBackgroundMesh(4)

        # Final settings
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 12)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)

        return
    
#**********************************************************************#

    def _setMeshSizesHexagon(self):
        fineMesh = self._param['gridThickness']
        courseMesh = 10*fineMesh
        transitionWidth = 2*self._param['holeRadius']

        largerHole = max(self._param['holeRadius'], self._param['padLength'])
        pipeRadius = largerHole + self._param['gridThickness']
        aboveGrid = self._param['gridThickness'] + self._param['holeRadius']
        hexLimit = self._param['pitch'] / math.sqrt(3) + 5.0

        # Center line below grid
        padCenter = self._occ.addPoint(0, 0, -self._param['gridStandoff'])
        holeCenter = self._occ.addPoint(0, 0, self._param['holeRadius']/2) 
        amplificationLine = self._occ.addLine(padCenter, holeCenter)
        self._occ.synchronize()

        # --- FIELD 1: Below the Grid ---
        gmsh.model.mesh.field.add('Distance', 1)
        gmsh.model.mesh.field.setNumbers(1, 'EdgesList', [amplificationLine])
        
        gmsh.model.mesh.field.add('Threshold', 2)
        gmsh.model.mesh.field.setNumber(2, 'InField', 1)
        gmsh.model.mesh.field.setNumber(2, 'SizeMin', fineMesh)
        gmsh.model.mesh.field.setNumber(2, 'SizeMax', courseMesh)
        gmsh.model.mesh.field.setNumber(2, 'DistMin', pipeRadius)
        gmsh.model.mesh.field.setNumber(2, 'DistMax', pipeRadius + transitionWidth)

        # --- FIELD 3: Above the Grid ---
        gmsh.model.mesh.field.add('Box', 3)
        gmsh.model.mesh.field.setNumber(3, 'VIn', 2*fineMesh)
        gmsh.model.mesh.field.setNumber(3, 'VOut', courseMesh)
        gmsh.model.mesh.field.setNumber(3, 'XMin', -hexLimit)
        gmsh.model.mesh.field.setNumber(3, 'XMax', hexLimit)
        gmsh.model.mesh.field.setNumber(3, 'YMin', -hexLimit)
        gmsh.model.mesh.field.setNumber(3, 'YMax', hexLimit)
        gmsh.model.mesh.field.setNumber(3, 'ZMin', -self._param['gridThickness'])
        gmsh.model.mesh.field.setNumber(3, 'ZMax', aboveGrid)
        gmsh.model.mesh.field.setNumber(3, 'Thickness', transitionWidth)

        # --- FIELD 4: Combine ---
        gmsh.model.mesh.field.add('Min', 4)
        gmsh.model.mesh.field.setNumbers(4, 'FieldsList', [2, 3])
        
        gmsh.model.mesh.field.setAsBackgroundMesh(4)

        # Final settings
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 12)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)

        return

#**********************************************************************#

    def createHexagonalGeometry(self, neighborCells=False, runGUI=False):
        """
        Generates a FEM of the hexagonal cell geometry using Gmsh.

        Args:
            neighborCells: If True, also creates the 6 adjacent cells.
            runGUI: Run the GUI to visualize the geometry and mesh.

        Returns:
            The base filename of the generated mesh.
        """

        filePath = 'Geometry'
        fileBase = 'FIMSHexagonalCell'
        filename = os.path.join(filePath, f'{fileBase}.msh')

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()

        gmsh.option.setNumber('Mesh.ElementOrder', 2)
        gmsh.option.setNumber('Mesh.HighOrderOptimize', 1)
        gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary', 1)
        gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 1)
        #gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)

        gmsh.model.add(filename)

        _, allCellsMap = self._makeHexagonalCells(neighborCells)
        self._assignPhysicalGroupsHexagon(allCellsMap)
        self._setMeshSizesHexagon()

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

        return fileBase
    

#**********************************************************************#

    def createFIMSGeometry(self, runGUI=False):
        """
        TODO
        """

        filePath = 'Geometry'
        fileBase = 'FIMS'
        filename = os.path.join(filePath, f'{fileBase}.msh')

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()

        gmsh.option.setNumber('Mesh.ElementOrder', 2)
        gmsh.option.setNumber('Mesh.HighOrderOptimize', 1)
        gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary', 1)
        gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 1)
        #gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)

        gmsh.model.add(filename)

        _, allCellsMap = self._makeUnitCell()
        self._assignPhysicalGroups(allCellsMap)
        self._setMeshSizes()

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

        return fileBase

    
            

#**********************************************************************#
#**********************************************************************#
#**********************************************************************#

class elmerClass:
    """
    Class for generating Elmer SIF files and running Elmer simulations.

    Can be used for any mesh generated by gmshClass.

    Can optionally calculate the capacitance matrix for the geometry.
    """

    def __init__(self, name, numPads, capacitance=False):
        """
        Initializes the elmerClass instance.
        
        Args:
            name (str): The base name for the mesh and SIF files.
            numPads (int): The number of pads in the geometry (1, 2 or 7).
            capacitance (bool): Calculates the capacitance matrix if true.
        """

        self.capacitance = capacitance

        self.meshFilename = f'{name}.msh'
        if self.capacitance:
            self.name = f'{name}Capacitance'
        else:
            self.name = name
        self.sifFilename = f'{self.name}.sif'

        self.numPads = numPads

        self._setElectrodeMap()
        
        self.numMaterials = 3
        #Currently not generating pillars
        #self.numMaterials = 4 if self.numPads == 2 else 3
        
        self._elmerBaseInfo()
        self._elmerSimulationInfo()
        self._selectSolver()
        self._addMaterials()
        self._assignBoundaryConditions()

        return
    
#**********************************************************************#

    def _setElectrodeMap(self):

        self.electrodeMap = {
            1: 'Cathode', 2: 'Grid', 3: 'CentralPad'
        }

        if self.numPads == 1:
            pass
        if self.numPads == 2:
            self.electrodeMap[4] = 'CornerPad'
        elif self.numPads == 7:
            self.electrodeMap.update({
                4: 'TopPad', 5: 'BottomPad',
                6: 'RightTopPad', 7: 'RightBottomPad',
                8: 'LeftTopPad', 9: 'LeftBottomPad'
            })
        else:
            raise ValueError('Number of pads must be 1, 2 or 7.')

#**********************************************************************#

    def _elmerBaseInfo(self):
        """
        Initializes the base information for the Elmer SIF file.
        """
        
        self.intro = (
            '! This file was generated by the FIMS code.\n'
            '! Do NOT edit manually.\n\n'
        )

        self.header = (
            'Header\n'
            '  CHECK KEYWORDS Warn\n'
            '  Mesh DB "elmerResults" "."\n'
            '  Include Path ""\n'
            '  Results Directory "elmerResults"\n'
            'End\n\n'
        )

        self.constants = {
            'Constants': {
                'Permittivity of Vacuum': '8.85418781e-12',
                'Permeability of Vacuum': '1.25663706e-6',
                'Boltzmann Constant': '1.380649e-23',
                'Unit Charge': '1.6021766e-19'
            }
        }

        self.equation = {
            'Equation 1': {
                'Name': '"EField"',
                'Electric Field': 'Computed',
                'Active Solvers(1)': '1'
            }
        }

        return
    
#**********************************************************************#

    def _elmerSimulationInfo(self): 

        self.simulation = {
            'Simulation': {
                'Max Output Level': '5',
                'Coordinate System': 'Cartesian',
                'Coordinate Mapping(3)': '1 2 3',
                'Simulation Type': 'Steady state',
                'Steady State Max Iterations': '1',
                'Output Intervals(1)': '1',
                'Coordinate Scaling': '1e-6',
                'Solver Input File': self.sifFilename,
                '! Post File': f'{self.name}.ep, {self.name}.vtu',
                'Output file': f'"elmerResults/{self.name}.result"'
            }
        }

        for i in self.electrodeMap:
            #Dont need weighting for cathode or grid
            if i == 'Cathode' or i == 'Grid':
                continue

            self.weighting[i] = {
            'Simulation': {
                'Max Output Level': '5',
                'Coordinate System': 'Cartesian',
                'Coordinate Mapping(3)': '1 2 3',
                'Simulation Type': 'Steady state',
                'Steady State Max Iterations': '1',
                'Output Intervals(1)': '1',
                'Coordinate Scaling': '1e-6',
                'Solver Input File': f'{self.name}{i}.sif',
                'Output file': f'"elmerResults/{self.name}+{i}.result"'
            }
        }
            
        return

#**********************************************************************#

    def _selectSolver(self):
        """
        Selects a solver for the Elmer simulation based on whether 
        capacitance calculation is needed.
        """

        self.solver = {
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
        
        if self.capacitance:
            self.solver['Solver 1'].update({
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
            {'Name': '"Air (room temperature)"', 'Relative Permittivity': '1.0'},
            {'Name': '"Aluminum (generic)"', 'Relative Permittivity': '1e10'},
            {'Name': '"SiO2"', 'Relative Permittivity': '3.9'},
            {'Name': '"Pillars"', 'Relative Permittivity': '3.0'},
        ]

        allBodies = [
            {'Target Bodies(1)': 3, 'Name': '"Gas"', 'Equation': 1, 'Material': 1},
            {'Target Bodies(1)': 1, 'Name': '"Amplification Grid"', 'Equation': 1, 'Material': 2},
            {'Target Bodies(1)': 2, 'Name': '"SiO2"', 'Equation': 1, 'Material': 3},
            {'Target Bodies(1)': 4, 'Name': '"Pillars"', 'Equation': 1, 'Material': 4},
        ]


        self.materials = {}
        self.bodies = {}

        for i in range(self.numMaterials):
            self.materials[f'Material {i+1}'] = allMaterials[i]
            self.bodies[f'Body {i+1}'] = allBodies[i]

        return  

#**********************************************************************#

    def _assignBoundaryConditions(self):
        """
        Assigns boundary conditions to the surfaces in the geometry 
        based on the number of pads.
        
        Adapts based on whether capacitance matrix is needed.
        """

        self.boundaries = {}

        for i in range(1, self.numPads+3):
            name = self.electrodeMap[i]
            content = {
                'Target Boundaries(1)': i,
                'Name': f'"{name}"',
            }

            if self.capacitance:
                content['Capacitance Body'] = i
            else:
                content['Potential'] = '0.0'

            self.boundaries[f'Boundary Condition {i}'] = content

        dielectricID = self.numPads+3
        boundaryID = self.numPads+4

        self.boundaries[f'Boundary Condition {dielectricID}'] = {
            'Target Boundaries(1)': dielectricID,
            'Name': '"DielectricSurfaceCharge"'
        }
        if not self.capacitance:
            self.boundaries[f'Boundary Condition {dielectricID}'].update({
                'Charge Density': 'Variable Coordinate 1, Coordinate 2, Coordinate 3',
                'File': '"chargeBuildup.dat"'
            })

        self.boundaries[f'Boundary Condition {boundaryID}'] = {
            'Target Boundaries(1)': boundaryID,
            'Name': '"MirrorBoundaries"',
            'Electric Flux': '0.0'
        }

        return

#**********************************************************************#

    def writeAllSIF(self):
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

        Note that grid and cathode potentials are set to 0 by default.
        """

        with open(f'Geometry/{self.sifFilename}', 'w') as f:

            f.write(self.intro)
            f.write(self.header)

            sections = [
                self.simulation, self.constants,
                self.solver, self.equation, self.materials,
                self.bodies, self.boundaries
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

        for i in self.electrodeMap:
            if i == 'Cathode' or i == 'Grid':
                continue
            inElectrode = False

            with open(f'Geometry/{self.name}{i}Weighting.sif', 'w') as f:

                f.write(self.intro)
                f.write(self.header)

                sections = [
                    self.weighting[i], self.constants,
                    self.solver, self.equation, self.materials,
                    self.bodies, self.boundaries
                ]

                for section in sections:
                    for title, content in section.items():
                        f.write(f'{title}\n')

                        for key, value in content.items():

                            #Handle charge density file differently
                            if 'Boundary' in title:
                                if key == 'File':
                                    f.write(f'  {key} {value}\n')

                                elif key == 'Name' and f'"{self.electrodeMap[i]}"' in value:
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

    def runElmer(self):
        """
        Runs the Elmer simulation using the generated SIF file and mesh.
        """

        self._runElmerGrid()
        self._runElmerSolver()

        if not self.capacitance:
            self._runElmerWeighting()

        return

#**********************************************************************#

    def _runElmerGrid(self):

        originalCWD = os.getcwd()
        os.chdir('./Geometry')

        os.makedirs('elmerResults', exist_ok=True)

        try:
            print('\tExecuting ElmerGrid...')
            
            with open(os.path.join(originalCWD, 'log/logElmerGrid.txt'), 'w+') as elmerOutput:
                startTime = time.monotonic()
                subprocess.run(
                    ['ElmerGrid', '14', '2', self.meshFilename, 
                    '-names',
                    '-out', 'elmerResults', 
                    '-autoclean'], 
                    stdout=elmerOutput,
                    check=True
                )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\nElmerGrid run time: {endTime - startTime} s')

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Elmer failed with exit code {e.returncode}.')
        
        finally:
            os.chdir(originalCWD)

        return
        
#**********************************************************************#

    def _runElmerSolver(self):
        """
        Runs the Elmer simulation using the generated SIF file and mesh.
        """
        originalCWD = os.getcwd()
        os.chdir('./Geometry')
            
        try:
            print('\tExecuting ElmerSolver...')

            with open(os.path.join(originalCWD, 'log/logElmerSolver.txt'), 'w+') as elmerOutput:
                startTime = time.monotonic()
                subprocess.run(
                    ['ElmerSolver', self.sifFilename],
                    stdout=elmerOutput,
                    check=True
                )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\nElmerSolver run time: {endTime - startTime} s')

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Elmer failed with exit code {e.returncode}.')
        
        finally:
            os.chdir(originalCWD)

        return
    
#**********************************************************************#

    def _runElmerWeighting(self):
        """
        Runs the Elmer simulation using the weighting files.
        """
        originalCWD = os.getcwd()
        os.chdir('./Geometry')
            
        try:
            print('\tExecuting ElmerSolver for Weightings...')

            with open(os.path.join(originalCWD, 'log/logElmerSolverWeighting.txt'), 'w+') as elmerOutput:
                startTime = time.monotonic()
                for i in self.electrodeMap:
                    if i == 'Cathode' or i == 'Grid':
                        continue
                    subprocess.run(
                        ['ElmerSolver', f'{self.name}{i}Weighting.sif'],
                        stdout=elmerOutput,
                        check=True
                    )
                endTime = time.monotonic()
                elmerOutput.write(f'\n\nElmerSolverWeighting run time: {endTime - startTime} s')

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
            electrode: The name of the electrode to set the potential for.
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

        with open(f'Geometry/{self.sifFilename}', 'r') as f:
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
            
            if inBoundary and strippedLine.startswith('Name') and f'"{electrode}"' in strippedLine:
                newContent += line + '\n'
                newContent += f'  Potential = {potential}\n'
                inBoundary = False
                continue
            
            newContent += line + '\n'

        with open(f'Geometry/{self.sifFilename}', 'w') as f:
            f.write(newContent)

        return
    
#**********************************************************************#

    def resetPotentials(self):

        """
        Resets the potentials for all electrodes in the SIF file.
        """

        with open(f'Geometry/{self.sifFilename}', 'r') as f:
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

        with open(f'Geometry/{self.sifFilename}', 'w') as f:
            f.write(newContent)

        return






