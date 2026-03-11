#################################
# CLASS DEFINITION FOR GEOMETRY #
#################################
from __future__ import annotations

import numpy as np
import os
import math
import gmsh

class gmshClass:

#***********************************************************************************#
    def __init__(self, inputParams=None):
      
        self._occ = gmsh.model.occ

        self._param = inputParams
        self._checkParameters()

#***********************************************************************************#
    def _checkParameters(self):
        """
        Checks that the parameters in self._paramList are valid for creating the geometry.
        """

        neededParameters = [
            'pitch',
            'holeRadius',
            'padLength',
            'gridStandoff',
            'cathodeHeight',
            'gridThickness',
            'dielectricThickness'
        ]

        if self._param is None:
            raise ValueError(
                f"Invalid parameters. Require dict with: {', '.join(neededParameters)}"
            )

        for inParam in self._param:
            if inParam not in neededParameters:
                raise ValueError(f"Parameter '{inParam}' is required but not provided.")  

#***********************************************************************************#
    def _createHexagonalUnitCell(self):
        """
        Create the geometry for a single hexagonal unit cell of the FIMS geometry.
        
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
        hexagonBase = self._makeHexagon(outRadius, -gridStandoff, dielectricThickness)
        padHole = self._makeHexagon(padLength, -gridStandoff, dielectricThickness)
        dielectricVolume, _ = self._occ.cut([(3, hexagonBase)], [(3, padHole)])

        ## Grid
        gridContainer = self._makeHexagon(outRadius, -gridThickness/2, gridThickness)
        gridHole = self._occ.addCylinder(
            0, 0, -gridThickness/2,
            0, 0, gridThickness,
            holeRadius        
        )
        gridVolume, _ = self._occ.cut([(3, gridContainer)], [(3, gridHole)])

        ## Gas
        gasHeight = cathodeHeight + gridStandoff
        gasContainer = self._makeHexagon(outRadius, -gridStandoff, gasHeight)
        gasVolume, _ = self._occ.cut(
            [(3, gasContainer)], 
            [(3, dielectricVolume[0][1]), (3, gridVolume[0][1])], 
            removeObject=True, removeTool=False
        )
        
        padSurface = self._makeHexagon(padLength, -gridStandoff)
        cathodeSurface = self._makeHexagon(outRadius, cathodeHeight)

        cellParts = {
            'Gas': (3, gasVolume[0][1]),
            'Dielectric': (3, dielectricVolume[0][1]),
            'Grid': (3, gridVolume[0][1]),
            'Pad': (2, padSurface),
            'Cathode': (2, cathodeSurface)
        }

        return cellParts
    
#***********************************************************************************#
    def _makeHexagon(self, outRadius, z, zDist=None):
        """
        Makes a hexagon in the xy-plane with the center at the origin.
        Extrudes it in the z-direction if zDist is provided.

        Args:
            outRadius: The distance from the center to the vertices of the hexagon.
            z: The z-coordinate of the hexagon.
            zDist: The distance to extrude the hexagon in the z-direction. 
            If None, the hexagon will not be extruded and will remain a 2D surface.
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
        
#***********************************************************************************#
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
                              where each dictionary contains the parts of a unit cell.
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

#***********************************************************************************#
    def _assignPhysicalGroups(self, entityMap):
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

        globalGroup = {'Gas': [], 'Dielectric': [], 'Grid': [], 'Cathode': []}
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
                gmsh.model.addPhysicalGroup(3, globalGroup[name], name=name)

        # Assign Surfaces
        if globalGroup['Cathode']:
            gmsh.model.addPhysicalGroup(2, globalGroup['Cathode'], name='Cathode')
        if allGridSurfaces:
            gmsh.model.addPhysicalGroup(2, list(set(allGridSurfaces)), name='Grid')
        # Assign Pads individually
        for i in range(min(len(padTags), len(padNames))):
            gmsh.model.addPhysicalGroup(2, padTags[i], name=padNames[i])

        return
    
#***********************************************************************************#
    def _setMeshSizes(self):
        fineMesh = self._param['gridThickness']
        courseMesh = 10*fineMesh
        transitionWidth = 2*self._param['holeRadius']
        
        pipeRadius = max(self._param['holeRadius'], self._param['padLength']) + self._param['gridThickness']
        pillTop = self._param['gridThickness'] + self._param['holeRadius']
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
        gmsh.model.mesh.field.setNumber(3, 'ZMax', pillTop)
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


#***********************************************************************************#
    def createHexagonalGeometry(self, neighborCells=False, runGUI=False):
        """
        Generates a FEM of the hexagonal cell geometry using Gmsh.

        Args:
            neighborCells: If True, also creates the 6 adjacent cells.
            runGUI: If True, runs the Gmsh GUI to visualize the geometry and mesh.

        Returns:
            The base filename of the generated mesh.
        """


        filePath = 'Geometry'
        fileBase = 'FIMSHexagonalCell'
        filename = os.path.join(filePath, f'{fileBase}.msh')

        gmsh.initialize()
        gmsh.model.add(filename)

        _, allCellsMap = self._makeHexagonalCells(neighborCells)
        self._assignPhysicalGroups(allCellsMap)
        self._setMeshSizes()

        gmsh.model.mesh.generate(3)
        gmsh.write(filename)

        if runGUI:
            gmsh.fltk.run()

        gmsh.finalize()

        return fileBase

    
            

#***********************************************************************************#
#***********************************************************************************#
#***********************************************************************************#

class elmerClass:
    def __init__(self, name, numPads, capacitance=False):

        self.capacitance = capacitance

        self.meshFilename = f'{name}.msh'
        if self.capacitance:
            self.name = f'{name}Capacitance'
        else:
            self.name = name
        self.sifFilename = f'{self.name}.sif'

        self.numPads = numPads
        self.numMaterials = 4 if self.numPads == 2 else 3
        
        self._elmerBaseInfo()
        self._selectSolver()
        self._addMaterials()
        self._assignBoundaryConditions()

        return

#***********************************************************************************#
    def _elmerBaseInfo(self):
        
        self.intro = (
            '! This file was generated by the FIMS code.\n'
            '! Do NOT edit manually.\n\n'
        )

        self.header = {
            'Header': {
                'CHECK KEYWORDS': '"Warn"',
                'Mesh DB': '"elmerResults" "."',
                'Include Path': '""',
                'Results Directory': '"elmerResults"',
            }
        }

        self.constants = {
            'Constants': {
                'Permittivity of Vacuum': '8.85418781e-12',
                'Permeability of Vacuum': '1.25663706e-6',
                'Boltzmann Constant': '1.380649e-23',
                'Unit Charge': '1.6021766e-19'
            }
        }

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

        self.equation = {
            'Equation': {
                'Name': '"EField"',
                'Electric Field': 'Computed',
                'Active Solvers(1)': '1'
            }
        }

        return

#***********************************************************************************#
    def _selectSolver(self):

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
    
#***********************************************************************************#
    def _addMaterials(self):

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
    

#***********************************************************************************#
    def _assignBoundaryConditions(self):

        self.boundaries = {}

        electrodeMap = {
            1: 'Cathode', 2: 'Grid', 3: 'CentralPad'
        }

        if self.numPads == 1:
            pass
        if self.numPads == 2:
            electrodeMap[4] = 'CornerPad'
        elif self.numPads == 7:
            electrodeMap.update({
                4: 'TopPad', 5: 'BottomPad',
                6: 'RightTopPad', 7: 'RightBottomPad',
                8: 'LeftTopPad', 9: 'LeftBottomPad'
            })
        else:
            raise ValueError('Number of pads must be 1, 2 or 7.')

        for i in range(1, self.numPads+3):
            name = electrodeMap[i]
            content = {
                'Target Boundaries(1)': i,
                'Name': f'"{name}"',
            }

            if self.capacitance:
                content['Capacitance Body'] = i
            else:
                content['Potential'] = '0.0'

            self.boundaries[f'Boundary {i}'] = content

        dielectricID = self.numPads+3
        boundaryID = self.numPads+4

        self.boundaries[f'Boundary {dielectricID}'] = {
            'Target Boundaries(1)': dielectricID,
            'Name': '"DielectricSurfaceCharge"'
        }
        if not self.capacitance:
            self.boundaries[f'Boundary {dielectricID}'].update({
                'Charge Density': 'Variable Coordinate 1, Coordinate 2, Coordinate 3',
                'File': '"chargeBuildup.dat"'
            })

        self.boundaries[f'Boundary {boundaryID}'] = {
            'Target Boundaries(1)': boundaryID,
            'Name': '"MirrorBoundaries"',
            'Electric Flux': '0.0'
        }

        return

        
        
#***********************************************************************************#
    def writeSIF(self):
        with open(f'Geometry/{self.sifFilename}', 'w') as f:

            f.write(self.intro)

            sections = [
                self.header, self.simulation, self.constants,
                self.solver, self.equation, self.materials,
                self.bodies, self.boundaries
            ]

            for section in sections:
                for title, content in section.items():
                    f.write(f'{title}\n')

                    for key, value in content.items():

                    #Handle charge density file differently since it doesn't use an equals sign
                        if 'Boundary' in title and key == 'File':
                            f.write(f'  {key} {value}\n')
                        else:
                            f.write(f'  {key} = {value}\n')
                            
                    f.write('End\n\n')







