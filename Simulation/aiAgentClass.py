from __future__ import annotations

import json
import math
import os
import time
import hashlib
import asyncio

import pandas as pd
import numpy as np
from configs import UnitCell, NumPyEncoder, aiModel
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool
)

class FIMS_AI_Agent:
    """
    AI-agent driven optimization of the FIMS amplification-grid hole shape.

    History is appended to ../Data/AI/holeShapeHistory.jsonl and mirrored to
    ../Data/AI/holeShapeHistory.csv for human review.
    
    methods:
        _setupAiInfo
        _describeConstraints
        _loadHistory
        _renderHistory
        _createShape
        _proposeShape
        updateHistory
        getCustomShape
        
    """
    
#**********************************************************************#

    def __init__(self, geoConfig, geoParams, resume=True):
        """Initializes a FIMS AI Agent."""
        self.bestEntry = None        
        self.maxProposals = 5
        self.history = []
        
        # Setup geometry
        self._geoConfig = geoConfig
        self._geoParams = geoParams

        # Setup Ai agent
        self.aiModel = aiModel.getAIModel()
        self._setupAiInfo()
        
        # Setup files
        aiPath = os.path.join('..', 'Data', 'AI')
        os.makedirs(aiPath, exist_ok=True)
    
        self.historyJSONL = os.path.join(aiPath, 'holeShapeHistory.jsonl')
        self.historyCSV = os.path.join(aiPath, 'holeShapeHistory.csv')
        self.idealShapePath = os.path.join(aiPath, 'bestHoleShape.json')
        self._customShapePath = os.path.join('Geometry', 'customShape.json')
        
        if resume:
            self._loadHistory()
            completed = [e for e in self.history if e.get('status') == 'ok']
            self.bestEntry = min(completed, key=lambda e: e['ibn']) if completed else None
            print(f'Resuming from {len(self.history)} recorded trials.')
        
        elif os.path.exists(self.historyJSONL):
            print(
                f'Warning: {self.historyJSONL} exists and will be appended to. '
                'Initialize with "resume = True" to let the agent see those trials.'
            )
        
        return

#**********************************************************************#

    def _setupAiInfo(self):
        """
        Gets the information to be fed to the AI agent.
        
        returns:
            aiPrompt (str): text description of what the agent should do.
            shapeTool (str): formatting of the hole shape the agent should use.
        """
        self.aiPrompt = """
            You are designing the hole outline in the amplification grid of a FIMS
            micro-pattern gaseous detector. Your objective is to MINIMIZE the ion backflow
            number (IBN). Lower is better.

            PHYSICS

            The amplification grid is a thin conducting sheet perforated by a lattice of
            holes. Electrons drifting down from the drift region are funneled through a
            hole into the amplification gap below, where they avalanche. The positive ions
            created in that avalanche drift back up. Any ion that makes it through the hole
            and into the drift region is backflow; any ion that lands on grid material is
            not. IBN is the mean number of backflowing ions per avalanche, so lower is
            better.

            The grid is arranged in a hexagonal pattern, so the holes must be able to be repeated
            in that pattern. Generally speaking, larger holes have an easier time collecting 
            electrons above the grid, which enables a lower field and smaller avalanches. 
            Smaller avalanches mean less ions are created which therefore lowers the IBN. However,
            larger holes also make it easier for ions created to backflow, so larger isn't 
            universally better. 

            You may also assume that ions are created with a 2D Gaussian distribution, with
            the majority created at the center of the hexagonal unit cell. This may mean
            that multiple holes per unit cell offset from the center could be ideal, but
            don't assume that.

            Treat all of the above as a starting intuition only. The simulation is the
            authority. If the measured history contradicts it, follow the history.

            RULES
             - You have exactly one action: propose_hole_shape. You cannot read files, run
               code, or change any other parameter.
              - Call propose_hole_shape exactly once, and then stop. Do not perform any other
                actions or attempt any other tasks.
             - IBN is reported with a Monte Carlo uncertainty. Two designs whose IBN differ
               by less than the quadrature sum of their errors are NOT distinguishable. Do
               not build a conclusion on such a gap; if you need to resolve one, say so in
               your hypothesis.
             - Each simulation costs between 3 minutes and 2 hours. Every proposal must
               earn its place. State a hypothesis that could turn out to be wrong.
             - While the history is short, spread out: change the shape family, not one
               number. Once a family clearly wins, perturb it deliberately along one axis
               at a time so you can attribute the change.
             - Never repeat an outline already in the history.
             - Respect the geometric constraints exactly. A violated constraint is caught
               before the simulator runs and simply wastes a turn.
            """

        holeSpec = {
            'type': 'object',
            'description': (
                'One hole. Every hole is described on its own: its own outline '
                'type, its own parameters, and its own position. Holes in the same '
                'cell need not resemble each other in any way.'
            ),
            'properties': {
                'type': {
                    'type': 'string',
                    'enum': ['polygon', 'polar', 'fourier'],
                    'description': (
                        'polygon: explicit x,y vertices. '
                        'polar: radius/angle pairs, auto-sorted by angle, which '
                        'cannot self-intersect. '
                        'fourier: a smooth closed curve '
                        'r(theta) = meanRadius + sum a_n cos(n*theta + phase_n); '
                        'the most compact way to describe lobed or rounded '
                        'outlines.'
                    ),
                },
                'offset': {
                    'type': 'array',
                    'items': {'type': 'number'},
                    'minItems': 2,
                    'maxItems': 2,
                    'description': (
                        '[dx, dy] of this hole center from the cell center, in '
                        'microns. Use [0, 0] to put it at the center.'
                    ),
                },
                'vertices': {
                    'type': 'array',
                    'minItems': 3,
                    'maxItems': 360,
                    'items': {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'minItems': 2,
                        'maxItems': 2,
                    },
                    'description': (
                        'For type "polygon" only: [[x, y], ...] in microns, ordered '
                        'around the outline and measured from THIS hole center, not '
                        'the cell center.'
                    ),
                },
                'points': {
                    'type': 'array',
                    'minItems': 3,
                    'maxItems': 360,
                    'items': {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'minItems': 2,
                        'maxItems': 2,
                    },
                    'description': (
                        'For type "polar" only: [[radius, angleDegrees], ...] about '
                        'this hole center. Radii must be positive; angles must be '
                        'distinct.'
                    ),
                },
                'meanRadius': {
                    'type': 'number',
                    'description': (
                        'For type "fourier" only: the mean radius of this hole in '
                        'microns.'
                    ),
                },
                'harmonics': {
                    'type': 'array',
                    'maxItems': 8,
                    'items': {
                        'type': 'object',
                        'properties': {
                            'n': {
                                'type': 'integer',
                                'minimum': 1,
                                'maximum': 24,
                                'description': 'Number of lobes.',
                            },
                            'amplitude': {
                                'type': 'number',
                                'description': (
                                    'Lobe depth in microns. The sum of the absolute '
                                    'amplitudes must stay below meanRadius.'
                                ),
                            },
                            'phaseDeg': {
                                'type': 'number',
                                'description': 'Phase offset in degrees.',
                            },
                        },
                        'required': ['n', 'amplitude'],
                    },
                    'description': (
                        'For type "fourier" only. An empty list gives a circle.'
                    ),
                },
                'numSamples': {
                    'type': 'integer',
                    'minimum': 12,
                    'maximum': 360,
                    'description': (
                        'For type "fourier" only: how many points to sample this '
                        'curve at. 120 is a good default; use more only for high '
                        'lobe counts.'
                    ),
                },
                'rotationDeg': {
                    'type': 'number',
                    'description': (
                        'Optional rotation of this outline about its own center, in '
                        'degrees.'
                    ),
                },
            },
            'required': ['type', 'offset'],
        }
        
        self.shapeTool = {
            'name': 'propose_hole_shape',
            'description': (
                'Propose the next grid hole pattern to simulate. This is your only '
                'available action. A pattern is a list of holes in one unit cell. '
                'Each hole is specified independently, so you may mix shapes, sizes '
                'and types freely within a cell. All lengths are in microns.'
            ),
            'input_schema': {
                'type': 'object',
                'properties': {
                    'hypothesis': {
                        'type': 'string',
                        'description': (
                            'What you expect this pattern to do to the IBN relative '
                            'to the current best, and the mechanism you think is '
                            'responsible. One or two sentences. Must be falsifiable '
                            'by the result.'
                        ),
                    },
                    'rationale': {
                        'type': 'string',
                        'description': (
                            'What in the history led you here. Name the specific runs '
                            'you are reasoning from.'
                        ),
                    },
                    'shape': {
                        'type': 'object',
                        'description': 'The hole pattern for one unit cell.',
                        'properties': {
                            'holes': {
                                'type': 'array',
                                'minItems': 1,
                                'maxItems': 24,
                                'items': holeSpec,
                                'description': (
                                    'Every hole in the cell. One entry gives a single '
                                    'hole; several entries give several fully '
                                    'independent holes, which may overlap.'
                                ),
                            },
                            'numHoles': {
                                'type': 'integer',
                                'minimum': 1,
                                'maximum': 24,
                                'description': (
                                    'Optional cross-check: must equal the length of '
                                    '"holes" if both are given.'
                                ),
                            },
                        },
                        'required': ['holes'],
                    },
                },
                'required': ['hypothesis', 'rationale', 'shape'],
            },
        }
        
        return

#**********************************************************************#

    def _describeConstraints(self):
        """
        Builds the geometric constraint block from the live parameters.
     
        returns:
            constraints (str): The rendered constraints.
        """
        pitch = float(self._geoParams['pitch'])
        padLength = float(self._geoParams['padLength'])
     
        cornerRadius = pitch/math.sqrt(3.)
        edgeRadius = pitch/2.
        cellLimit = edgeRadius*0.95
     
        constraints = (
            f'  Unit cell: regular hexagon, pitch = {pitch:.1f} um (cell center to '
            f'cell center).\n'
            f'    Corners sit at radius {cornerRadius:.2f} um, at 0, 60, 120, 180, '
            f'240 and 300 deg.\n'
            f'    Edge midpoints sit at radius {edgeRadius:.2f} um, at 30, 90, 150, '
            f'210, 270 and 330 deg.\n'
            f'  Pad length: {padLength:.1f} um.\n'
            f'  The whole pattern is tiled onto every neighboring cell.\n'
            f'  Each hole outline is measured from ITS OWN center, then moved to '
            f'its offset.\n'
            f'  CONTAINMENT: every vertex of every hole, after its offset is '
            f'applied, must satisfy\n'
            f'    x*cos(a) + y*sin(a) < {cellLimit:.2f} um   for a = 30, 90, 150, '
            f'210, 270, 330 deg.\n'
            f'    That is the hexagon shrunk by a 5% wall of grid material. Note '
            f'a vertex aimed at a\n'
            f'    corner may sit further from the center than one aimed at an '
            f'edge.\n'
            f'  Holes may overlap each other freely. Two holes that come within '
            f'0.5 um WITHOUT\n'
            f'    overlapping are rejected, because the sliver of grid metal '
            f'between them cannot be meshed.\n'
            f'  At most 24 holes, 360 vertices per hole and 1440 vertices in the '
            f'whole pattern.\n'
            f'  No edge shorter than 0.15 um. No outline may cross itself or pass '
            f'through its own center.'
        )
     
        return constraints

    #**********************## History handling ##**************************#

    def _loadHistory(self):
        """
        Reads an existing JSONL history file.

        Returns:
            history (list): List of history records, empty if the file does not exist.
        """
        if not os.path.exists(self.historyJSONL):
            return

        with open(self.historyJSONL, 'r') as inFile:
            for line in inFile:
                line = line.strip()
                if line:
                    self.history.append(json.loads(line))
        
        return

    #**********************************************************************#

    def updateHistory(self, record):
        """
        Updates history file and mirrors the history to a CSV file.

        Args:
            record (dict): The record to append to JSON file.
        """
        self.history.append(record)
        
        # Update JSON file
        with open(self.historyJSONL, 'a') as outFile:
            outFile.write(json.dumps(record) + '\n')
            outFile.flush()
            os.fsync(outFile.fileno())
        
        # Update CSV file
        allRows = []
        for entry in self.history:
            summary = entry.get('summary') or {}
            allRows.append({
                'iteration': entry.get('iteration'),
                'status': entry.get('status'),
                'runNumber': entry.get('runNumber'),
                'IBN': entry.get('ibn'),
                'IBN Error': entry.get('ibnError'),
                'shapeType': entry.get('shapeType'),
                'shapeHash': entry.get('shapeHash'),
                'numVertices': summary.get('numVertices'),
                'area': summary.get('area'),
                'perimeter': summary.get('perimeter'),
                'openAreaFraction': summary.get('openAreaFraction'),
                'numHoles': summary.get('numHoles'),
                'maxRadius': summary.get('maxRadius'),
                'minRadius': summary.get('minRadius'),
                'areaIsExact': summary.get('areaIsExact'),
                'duration': entry.get('duration'),
                'hypothesis': entry.get('hypothesis'),
                'rationale': entry.get('rationale'),
                'message': entry.get('message'),
                'spec': json.dumps(entry.get('spec'))
            })

        pd.DataFrame(allRows).to_csv(self.historyCSV, index=False)
        
        # Identify best entry and update path file
        completed = [e for e in self.history if e.get('status') == 'ok']
        if not completed:
            return None
        
        newBestEntry = min(completed, key=lambda e: e['ibn'])
        
        if newBestEntry != self.bestEntry:
            self.bestEntry = newBestEntry
            with open(self.idealShapePath, 'w') as outFile:
                json.dump(self.bestEntry, outFile, indent=2)

        return

    #**********************************************************************#

    def _renderHistory(self, numDetailed=3):
        """
        Renders the trial history into a compact block for the agent.

        A summary table covers every attempt; the full outline is included only
        for the best few and the most recent, which keeps the prompt bounded no
        matter how many iterations have run.

        Args:
            numDetailed (int): How many of the best shapes to spell out in full.

        Returns:
            str: The rendered history.
        """
        if not self.history:
            return (
                'Nothing has been simulated yet. Propose a starting point and say '
                'what you expect from it.'
            )

        header = (
            f"{'it':>3}  {'status':<8} {'IBN':>10} {'+/-':>8} {'N':>2} "
            f"{'maxR':>6} {'minR':>6} {'area':>8} {'perim':>7} {'open':>6}  types"
        )
        allRows = [header, '-'*len(header)]

        for entry in self.history:
            summary = entry.get('summary') or {}
            nan = float('nan')

            if entry.get('status') == 'ok':
                ibnText = f"{entry['ibn']:.5g}"
                errText = f"{entry['ibnError']:.3g}"
            else:
                ibnText, errText = '-', '-'

            allRows.append(
                f"{entry.get('iteration', -1):>3}  "
                f"{entry.get('status', '?'):<8} {ibnText:>10} {errText:>8} "
                f"{summary.get('numHoles', 0):>2d} "
                f"{summary.get('maxRadius', nan):>6.2f} "
                f"{summary.get('minRadius', nan):>6.2f} "
                f"{summary.get('area', nan):>8.1f} "
                f"{summary.get('perimeter', nan):>7.1f} "
                f"{summary.get('openAreaFraction', nan):>6.3f}  "
                f"{entry.get('shapeType', '?')}"
            )

            if entry.get('status') != 'ok' and entry.get('message'):
                allRows.append(f"       -> {entry['message']}")

        # Spell out the outlines worth perturbing
        completed = [e for e in self.history if e.get('status') == 'ok']
        detailIds = [
            e['iteration']
            for e in sorted(completed, key=lambda e: e['ibn'])[:numDetailed]
        ]
        for entry in self.history[-2:]:
            if entry.get('status') == 'ok' and entry['iteration'] not in detailIds:
                detailIds.append(entry['iteration'])

        if detailIds:
            allRows.append('')
            allRows.append('FULL OUTLINES (best first, then most recent)')
            for entry in self.history:
                if entry['iteration'] not in detailIds:
                    continue
                allRows.append(
                    f"\n  iteration {entry['iteration']}  "
                    f"IBN = {entry['ibn']:.5g} +/- {entry['ibnError']:.3g}"
                )
                allRows.append(f"  spec: {json.dumps(entry['spec'])}")
                if entry.get('hypothesis'):
                    allRows.append(f"  you predicted: {entry['hypothesis']}")

        return '\n'.join(allRows)

    #**********************************************************************#

    def _createShape(self, shapeSummary):
        """
        Creates a custom hole pattern and writes it to a given file path.

        A bare outline with no 'holes' entry is read as a single hole at
        the cell center.

        Note: intended to be used by AI agent.

        args:
            shapeSummary (dict): details of the custom shape.
                {'holes': [holeSpec, holeSpec, ...]}
                Each holeSpec is an independent outline:
                    'type'
                    'vertices'
                    'points'
                    'meanRadius'
                    'harmonics'
                    'numSamples'
                    'rotationDeg'
                    'offset'
                Optional: 'numHoles' (int), cross-checked against len(holes).

        returns:
            validSummary (dict): the shape details after being validated.
                'numHoles'
                'numVertices'
                'maxRadius'
                'minRadius'
                'area'
                'areaIsExact'
                'perimeter'
                'openAreaFraction'
                'shapeHash'
                'holes'
        """
        allHoles = self._resolveHolePattern(shapeSummary)
        validSummary = self._validateHolePattern(allHoles)

        # Create hash of every x,y point in the pattern
        canonicalForm = json.dumps([
            [[round(x, 6), round(y, 6)] for x, y in hole['vertices']]
            for hole in allHoles
        ])
        validSummary['shapeHash'] = hashlib.sha1(
            canonicalForm.encode()
        ).hexdigest()[:12]

        payload = {
            'spec': shapeSummary,
            'vertices': [[float(x), float(y)] for x, y in allHoles[0]['vertices']],
            'summary': validSummary,
            'pitch': float(self._geoParams['pitch']),
            'unitCell': self._geoConfig.unitCell.value,
            'holes': [
                {
                    'type': hole['type'],
                    'offset': hole['offset'],
                    'vertices': [[float(x), float(y)] for x, y in hole['vertices']],
                } 
                for hole in allHoles
            ],
        }

        with open(self._customShapePath, 'w') as outFile:
            json.dump(payload, outFile, indent=2, cls=NumPyEncoder)

        # Use maximum radial position for future radius checks.
        self._geoParams['holeRadius'] = validSummary['maxRadius']

        return validSummary

    #**********************************************************************#

    def _resolveShapeSpec(self, shapeSpec):
        """
        Converts a hole shape into a list of (x, y) vertices.
        
        Note: list is closed and counter-clockwise.
        
        args:
            shapeSpec (dict):

        returns:
            shapeList (list): List of (x, y) tuples describing the hole outline.
        """
        shapeType = str(shapeSpec.get('type', '')).strip().lower()

        match shapeType:
            case 'polygon':
                rawVertices = shapeSpec.get('vertices')
                if not rawVertices:
                    raise ValueError("Error: 'polygon' requires 'vertices'.")

                points = [
                    (float(vertex[0]), float(vertex[1]))
                    for vertex in rawVertices
                ]

            case 'polar':
                rawPoints = shapeSpec.get('points')
                if not rawPoints:
                    raise ValueError("Error: 'polar' requires 'points'.")

                polarPoints = sorted(
                    (float(theta) % 360., float(radius))
                    for radius, theta in rawPoints
                )

                allAngles = [theta for theta, _ in polarPoints]
                if len(set(allAngles)) != len(allAngles):
                    raise ValueError('Error: repeated angle in polar outline.')
                if any(radius <= 0. for _, radius in polarPoints):
                    raise ValueError('Error: polar radii must be positive.')

                points = [
                    (
                        radius*math.cos(math.radians(theta)),
                        radius*math.sin(math.radians(theta))
                    )
                    for theta, radius in polarPoints
                ]

            case 'fourier':
                meanRadius = float(shapeSpec.get('meanRadius', 0.))
                harmonics = shapeSpec.get('harmonics', [])
                numSamples = int(shapeSpec.get('numSamples', 120))

                if meanRadius <= 0.:
                    raise ValueError("Error: 'meanRadius' must be positive.")
                if not 12 <= numSamples <= 360:
                    raise ValueError("Error: 'numSamples' must be between 12 and 360.")

                points = []
                for i in range(numSamples):
                    theta = 2.*math.pi*i/numSamples
                    radius = meanRadius

                    for term in harmonics:
                        order = int(term['n'])
                        amplitude = float(term['amplitude'])
                        phase = math.radians(float(term.get('phaseDeg', 0.)))
                        radius += amplitude*math.cos(order*theta + phase)

                    if radius <= 0.:
                        raise ValueError(
                            'Error: Harmonics drive the radius to or below '
                            f'zero at {math.degrees(theta):.1f} deg. Reduce '
                            'the amplitudes or raise the mean radius.'
                        )

                    points.append(
                        (radius*math.cos(theta), radius*math.sin(theta))
                    )

            case _:
                raise ValueError(
                    f"Error: Unsupported hole shape type: '{shapeType}'."
                )

        rotation = math.radians(float(shapeSpec.get('rotationDeg', 0.)))
        if rotation:
            cosAngle = math.cos(rotation)
            sinAngle = math.sin(rotation)
            points = [
                (x*cosAngle - y*sinAngle, x*sinAngle + y*cosAngle)
                for x, y in points
            ]

        # Ensure points are arranged counter-clockwise
        if self._signedArea(points) < 0.:
            points.reverse()

        return points

    #**********************************************************************#

    def _resolveHolePattern(self, shapeSpec):
        """
        Expands a pattern specification into a list of placed holes.

        args:
            shapeSpec (dict): details of the hole pattern.

        returns:
            allHoles (list): one dict per hole containing 'type', 'offset',
                'vertices', 'edges', 'boundRadius', 'minLocalRadius', 'area'
                and 'perimeter'.
        """
        if not isinstance(shapeSpec, dict):
            raise ValueError('Error: Shape specification must be a dictionary.')

        allSpecs = shapeSpec.get('holes')
        if allSpecs is None:
            # A bare outline is read as one hole at the cell center
            allSpecs = [shapeSpec]

        if not isinstance(allSpecs, list) or not allSpecs:
            raise ValueError("Error: 'holes' must be a non-empty list.")

        declaredCount = shapeSpec.get('numHoles')
        if declaredCount is not None and int(declaredCount) != len(allSpecs):
            raise ValueError(
                'Error: Number of holes given does not match the number of holes declared:\n'
                f'\tDeclared: {declaredCount} =/= Actual: {len(allSpecs)}'
            )

        allHoles = []
        for index, holeSpec in enumerate(allSpecs): 
            if not isinstance(holeSpec, dict):
                raise ValueError(f'Error: Hole {index} is not a dictionary.')
            
            # Resolve the outline of the hole
            try:
                outline = self._resolveShapeSpec(holeSpec)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                detail = str(error).replace('Error: ', '', 1)
                raise ValueError(f'Error: Hole {index}: {detail}') from None
            
            # Offset the hole from the center
            rawOffset = holeSpec.get('offset', [0., 0.])
            if len(rawOffset) != 2:
                raise ValueError(
                    f'Error: Hole {index} offset must be [dx, dy].'
                )
            offsetX = float(rawOffset[0])
            offsetY = float(rawOffset[1])

            vertices = [(x + offsetX, y + offsetY) for x, y in outline]

            numVertices = len(vertices)
            allEdges = [
                math.dist(vertices[i], vertices[(i + 1) % numVertices])
                for i in range(numVertices)
            ]
            localRadii = [math.hypot(x, y) for x, y in outline]

            allHoles.append({
                'type': str(holeSpec.get('type', '')).strip().lower(),
                'offset': [offsetX, offsetY],
                'vertices': vertices,
                'edges': allEdges,
                'boundRadius': max(localRadii),
                'minLocalRadius': min(localRadii),
                'area': abs(self._signedArea(vertices)),
                'perimeter': sum(allEdges),
            })

        return allHoles

    #**********************************************************************#

    def _validateHolePattern(self, allHoles, minEdge=0.15, minHoleGap=0.5):
        """
        Verifies that a hole pattern fits the unit cell and can be meshed.

            Checks each hole individually, and then verifies that the full
        pattern fits within the unit cell. Also verifies that the gap
        between adjacent (but not overlapping) holes isn't too small.

        args:
            allHoles (list): List of all holes.
            minEdge (float): Shortest permitted outline edge, in microns.
                This only rejects degenerate edges; it is not a
                resolution limit.
            minHoleGap (float): Smallest permitted gap between holes
                that do not overlap, in microns.

        returns:
            validSummary (dict): Summary of the validated pattern.
        """
        safetyBuffer = .05
        maxPatternVertices=1440
        
        totalVertices = sum(len(hole['vertices']) for hole in allHoles)
        if totalVertices > maxPatternVertices:
            raise ValueError(
                f'Error: Pattern uses {totalVertices} vertices across '
                f'{len(allHoles)} holes; the budget is {maxPatternVertices}. '
                'Lower numSamples or use fewer holes.'
            )

        # Check each individual hole for self consistency
        for index, hole in enumerate(allHoles):
            label = f'Hole {index} ({hole["type"]})'
            numVertices = len(hole['vertices'])

            if numVertices < 3:
                raise ValueError(f'Error: {label} needs at least 3 vertices.')
            if numVertices > 360:
                raise ValueError(f'Error: {label} exceeds 360 vertices.')

            # Ensure the edge of the hole does not touch its own center.
            if hole['minLocalRadius'] <= 1e-6:
                raise ValueError(f'Error: {label} touches its own axis.')

            # Ensure all edges are above the minimum size threshold.
            shortestEdge = min(hole['edges'])
            if shortestEdge < minEdge:
                raise ValueError(
                    f'Error: {label} has an edge of {shortestEdge:.3f} um; '
                    f'the minimum is {minEdge:.2f} um.'
                )

            # Ensure hole doesn't cross itself.
            if self._hasSelfIntersection(hole['vertices']):
                raise ValueError(f'Error: {label} outline crosses itself.')

            # Ensure hole exists
            if hole['area'] <= 0.:
                raise ValueError(f'Error: {label} encloses no area.')

        # Check all holes vs the unit cell
        pitch = float(self._geoParams['pitch'])
        cellLimit = (pitch/2.)*(1. - safetyBuffer)

        for index, hole in enumerate(allHoles):
            for normalX, normalY, angleDeg in self._cellEdges():
                reach = max(
                    x*normalX + y*normalY for x, y in hole['vertices']
                )
                if reach >= cellLimit:
                    raise ValueError(
                        f'Error: Hole {index} at offset '
                        f'({hole["offset"][0]:.2f}, {hole["offset"][1]:.2f}) '
                        f'reaches {reach:.2f} um toward the cell edge at '
                        f'{angleDeg:.0f} deg; the limit is {cellLimit:.2f} um. '
                        'Move it inward or make it smaller.'
                    )

        # Ensure holes have a minimum gap between them
        for i in range(len(allHoles)):
            for j in range(i + 1, len(allHoles)):
                first = allHoles[i]
                second = allHoles[j]

                centerGap = math.dist(first['offset'], second['offset'])
                boundSum = first['boundRadius'] + second['boundRadius']

                # Far enough apart that only the bounding circles matter
                if centerGap > boundSum:
                    separation = centerGap - boundSum
                    if separation < minHoleGap:
                        raise ValueError(
                            f'Error: Holes {i} and {j} are separated by only '
                            f'{separation:.3f} um. Either overlap them or '
                            f'leave at least {minHoleGap:.2f} um between them.'
                        )
                    continue

                if self._holesOverlap(first, second):
                    continue

                gap = self._outlineGap(first['vertices'], second['vertices'])
                if gap < minHoleGap:
                    raise ValueError(
                        f'Error: Holes {i} and {j} come within {gap:.3f} um '
                        'without overlapping. Either overlap them or leave '
                        f'at least {minHoleGap:.2f} um between them.'
                    )

        # Calculate optical transparency and return full pattern
        allRadii = [
            math.hypot(x, y) for hole in allHoles for x, y in hole['vertices']
        ]
        area, areaIsExact = self._getCustomArea(allHoles)

        # Area of the unit cell this pattern sits in
        if self._geoConfig.unitCell == UnitCell.HEXAGON:
            cellArea = math.sqrt(3)/2.*pitch**2
        else:
            cellArea = pitch**2

        validSummary = {
            'numHoles': len(allHoles),
            'numVertices': totalVertices,
            'maxRadius': max(allRadii),
            'minRadius': min(hole['minLocalRadius'] for hole in allHoles),
            'area': area,
            'areaIsExact': areaIsExact,
            'perimeter': sum(hole['perimeter'] for hole in allHoles),
            'openAreaFraction': area/cellArea,
            'holes': [
                {
                    'type': hole['type'],
                    'offset': hole['offset'],
                    'area': hole['area'],
                    'perimeter': hole['perimeter'],
                    'boundRadius': hole['boundRadius'],
                }
                for hole in allHoles
            ],
        }

        return validSummary

    #**********************************************************************#

    def _cellEdges(self):
        """
        Gets the distances for the center of each edge of the unit cell.

        returns:
            edges (list): (normalX, normalY, angleDegrees) per cell edge.
        """
        match self._geoConfig.unitCell:
            case UnitCell.HEXAGON:
                allAngles = [30., 90., 150., 210., 270., 330.]

            case UnitCell.SQUARE:
                allAngles = [0., 90., 180., 270.]

            case _:
                raise ValueError(
                    f'Error: Unsupported unit cell: '
                    f'{self._geoConfig.unitCell}'
                )

        edges = [
            (
                math.cos(math.radians(angle)),
                math.sin(math.radians(angle)),
                angle
            )
            for angle in allAngles
        ]

        return edges

    #**********************************************************************#

    @staticmethod
    def _holesOverlap(firstHole, secondHole):
        """
        Tests whether two placed holes share any area.

        Outlines are densely sampled, so testing whether either vertex set
        falls inside the other polygon is sufficient in practice.

        args:
            firstHole, secondHole (dict): Placed holes.

        returns:
            bool: True if the two holes overlap.
        """
        from matplotlib.path import Path

        firstPath = Path(firstHole['vertices'])
        secondPath = Path(secondHole['vertices'])

        if firstPath.contains_points(secondHole['vertices']).any():
            return True
        if secondPath.contains_points(firstHole['vertices']).any():
            return True

        return False

    #**********************************************************************#

    @staticmethod
    def _outlineGap(firstVertices, secondVertices):
        """
        Finds the smallest distance between two outlines.

        Measures every vertex of each outline against the edge of an
        adjacent hole.

        args:
            firstVertices, secondVertices (list): (x, y) vertex tuples.

        returns:
            gap (float): Closest approach in microns.
        """
        def pointsToEdges(rawPoints, rawPolygon):
            points = np.asarray(rawPoints, dtype=float)
            starts = np.asarray(rawPolygon, dtype=float)
            edges = np.roll(starts, -1, axis=0) - starts

            lengthSq = (edges**2).sum(axis=1)
            lengthSq[lengthSq == 0.] = 1e-30

            offsets = points[:, None, :] - starts[None, :, :]
            along = (offsets*edges[None, :, :]).sum(axis=2)/lengthSq[None, :]
            along = np.clip(along, 0., 1.)

            closest = starts[None, :, :] + along[:, :, None]*edges[None, :, :]
            gaps = np.sqrt(((points[:, None, :] - closest)**2).sum(axis=2))

            return gaps.min()

        gap = float(min(
            pointsToEdges(firstVertices, secondVertices),
            pointsToEdges(secondVertices, firstVertices),
        ))

        return gap

    #**********************************************************************#

    @staticmethod
    def _getCustomArea(allHoles, numSamples=800):
        """
        Open area of the union of every hole in the pattern.

        Note: when two or more holes overlap, the union is estimated
        on a regular grid. Otherwise, the area is exact.

        args:
            allHoles (list): Placed holes from _resolveHolePattern().
            numSamples (int): Grid resolution per axis for the estimate.

        returns:
            tuple: (area, areaIsExact)
        """
        if len(allHoles) == 1:
            return allHoles[0]['area'], True

        # Check if holes overlap
        canTouch = False
        for i in range(len(allHoles)):
            for j in range(i + 1, len(allHoles)):
                centerGap = math.dist(
                    allHoles[i]['offset'], allHoles[j]['offset']
                )
                boundSum = (
                    allHoles[i]['boundRadius'] + allHoles[j]['boundRadius']
                )
                if centerGap <= boundSum:
                    canTouch = True
                    break
            if canTouch:
                break

        if not canTouch:
            return sum(hole['area'] for hole in allHoles), True

        from matplotlib.path import Path

        allPoints = np.asarray(
            [vertex for hole in allHoles for vertex in hole['vertices']],
            dtype=float
        )
        xMin, yMin = allPoints.min(axis=0)
        xMax, yMax = allPoints.max(axis=0)

        xGrid = np.linspace(xMin, xMax, numSamples)
        yGrid = np.linspace(yMin, yMax, numSamples)
        meshX, meshY = np.meshgrid(xGrid, yGrid)
        samplePoints = np.column_stack([meshX.ravel(), meshY.ravel()])

        inside = np.zeros(samplePoints.shape[0], dtype=bool)
        for hole in allHoles:
            inside |= Path(hole['vertices']).contains_points(samplePoints)

        sampleArea = (
            (xMax - xMin)/(numSamples - 1)*(yMax - yMin)/(numSamples - 1)
        )

        return float(inside.sum()*sampleArea), False

    #**********************************************************************#

    @staticmethod
    def _signedArea(points):
        """
        Computes the signed area of a closed polygon (shoelace formula).

        args:
            points (list): List of (x, y) vertex tuples.

        returns:
            area (float): Signed area. Positive for counter-clockwise ordering.
        """
        total = 0.
        numPoints = len(points)

        for i in range(numPoints):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % numPoints]
            total += x1*y2 - x2*y1
        area = total/2.
        
        return area

    #**********************************************************************#

    @staticmethod
    def _hasSelfIntersection(vertices):
        """
        Tests a closed polygon for crossing edges.

        Edges that share a vertex are skipped. This is a strict crossing
        test and does not flag co-linear overlap.

        args:
            vertices (list): List of (x, y) vertex tuples.

        returns:
            bool: True if any two non-adjacent edges cross.
        """
        numVertices = len(vertices)
        crosses = False

        def orientation(pointA, pointB, pointC):
            value = (
                (pointB[0] - pointA[0])*(pointC[1] - pointA[1])
                - (pointB[1] - pointA[1])*(pointC[0] - pointA[0])
            )
            if abs(value) < 1e-12:
                return 0

            return 1 if value > 0 else -1
        
        for i in range(numVertices):
            firstStart = vertices[i]
            firstEnd = vertices[(i + 1) % numVertices]

            for j in range(i + 1, numVertices):
                # Skip edges sharing a vertex with edge i
                if j == (i + 1) % numVertices or (j + 1) % numVertices == i:
                    continue

                secondStart = vertices[j]
                secondEnd = vertices[(j + 1) % numVertices]

                d1 = orientation(secondStart, secondEnd, firstStart)
                d2 = orientation(secondStart, secondEnd, firstEnd)
                d3 = orientation(firstStart, firstEnd, secondStart)
                d4 = orientation(firstStart, firstEnd, secondEnd)

                if d1 != d2 and d3 != d4:
                    crosses = True
                    return crosses

        return crosses

    #***********************## Run Simulation ##***************************#

    def _proposeShape(self, iteration, maxIterations, feedback=None):
        """
        Asks the model for exactly one new hole outline.

        Each iteration is a fresh single-turn call rather than a growing
        conversation. The history is re-rendered compactly every time, so the
        prompt stays the same size whether this is iteration 3 or 300.

        args:
            iteration (int): The iteration about to be run.
            maxIterations (int): Total iteration budget.
            constraints (str): Rendered geometric constraints.
            feedback (str): Text describing rejected attempts this iteration.
            
        returns:
            proposal (dict): The tool input, containing 'shape', 'hypothesis', 'rationale'.
        """
        bestText = (
            f'Best so far: iteration {self.bestEntry["iteration"]} at '
            f'IBN = {self.bestEntry["ibn"]:.5g} +/- {self.bestEntry["ibnError"]:.3g}.'
            if self.bestEntry else 'No successful run yet.'
        )
        
        constraints = self._describeConstraints()
        
        userBlock = '\n\n'.join(filter(None, [
            f'Iteration {iteration} of {maxIterations}. {bestText}',
            'GEOMETRIC CONSTRAINTS\n' + constraints,
            'HISTORY\n' + self._renderHistory(),
            feedback,
            'Propose the next outline.',
        ]))
        
        proposal = asyncio.run(self._runAgentTurn(userBlock))
        
        return proposal

    #**********************************************************************#

    async def _runAgentTurn(self, userBlock):
        """
        Runs one single-turn agent query and returns the proposal.

        Args:
            userBlock (str): The rendered user turn.

        Returns:
            dict: The tool input, containing 'shape', 'hypothesis', 'rationale'.
        """
        proposal = {}

        @tool(
            self.shapeTool['name'],
            self.shapeTool['description'],
            self.shapeTool['input_schema'],
        )
        async def proposeHoleShape(args):
            proposal.update(args)
            return {
                'content': [
                    {'type': 'text', 'text': 'Proposal recorded. Stop here.'}
                ]
            }

        shapeServer = create_sdk_mcp_server(
            name='fims',
            version='1.0.0',
            tools=[proposeHoleShape],
        )

        options = ClaudeAgentOptions(
            model=self.aiModel,
            system_prompt=self.aiPrompt,
            mcp_servers={'fims': shapeServer},
            allowed_tools=[f'mcp__fims__{self.shapeTool["name"]}'],
            disallowed_tools=[
                'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
                'WebFetch', 'WebSearch', 'Task', 'TodoWrite',
            ],
            permission_mode='dontAsk',
            setting_sources=[],
            max_turns=2,
        )

        lastResult = None
        async for message in query(prompt=userBlock, options=options):
            if isinstance(message, ResultMessage):
                lastResult = message

        if not proposal:
            detail = f' ({lastResult.subtype})' if lastResult is not None else ''
            raise RuntimeError(
                f'Error: Model returned no hole shape proposal{detail}.'
            )

        return dict(proposal)
    #**********************************************************************#

    def getCustomShape(self, iteration, finalIteration):
        """
        Gets a custom shape description.
        
        args:
            iteration (int): the current interation number.
            finalIteration (int): the final iteration number.
        """
        feedback = None
        rejected = []
        proposal = None
        
        # Propose hole shape
        for attempt in range(self.maxProposals):
            # First iteration is a circle
            if (not self.history and attempt == 0):
                proposal = {
                    'shape': {
                        'holes': [{
                            'type': 'fourier',
                            'meanRadius': float(self._geoParams['holeRadius']),
                            'harmonics': [],
                            'numSamples': 120,
                            'offset': [0., 0.],
                        }],
                    },
                    'hypothesis': 'Baseline circular hole, seeded by the script.',
                    'rationale': 'Reference point for every later design.'
                }
            
            # Second iteration is 6 equilateral triangles in a hexagonal pattern.
            elif (len(self.history) == 1 and attempt == 0):
                patternScale = 0.94
                holeGap = 0.75

                ringRadius = patternScale*pitch/3.
                triRadius = ringRadius - holeGap

                allHoles = []
                for index in range(6):
                    ringAngle = math.radians(30. + 60.*index)
                    allHoles.append({
                        'type': 'polar',
                        'points': [
                            [triRadius, 90.],
                            [triRadius, 210.],
                            [triRadius, 330.],
                        ],
                        # Every other triangle is flipped
                        'rotationDeg': 180.*(index % 2),
                        'offset': [
                            ringRadius*math.cos(ringAngle),
                            ringRadius*math.sin(ringAngle),
                        ],
                    })
                
                proposal = {
                    'shape': {
                        'holes': allHoles,
                        'numHoles': 6,
                    },
                    'hypothesis': (
                        'Six triangles tiled into a hexagon hold roughly the '
                        'same open area as the circle but add six thin grid '
                        'spokes running from the cell center outward. If ion '
                        'collection is driven by grid material sitting above '
                        'the avalanche, IBN should fall below iteration 1. If '
                        'it does not, open area rather than grid topology is '
                        'what sets IBN.'
                    ),
                    'rationale': 'Seeded by the script as a second reference point.'
                }
            
            else:
                proposal = self._proposeShape(iteration, finalIteration, feedback=feedback)
            
            # Create hole shape
            try:
                shapeSummary = self._createShape(proposal['shape'])
                
                # Ensure shape is new
                priorHashes = {e.get('shapeHash') for e in self.history if e.get('shapeHash')}
                if shapeSummary['shapeHash'] not in priorHashes:
                    return proposal, shapeSummary
                
                else:
                    message = 'This outline is already in the history.'
            
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                message = str(error)

            print(f'\t Rejected proposal: {message}')
            rejected.append(f'\t - {json.dumps(proposal["shape"])}\n \t{message}')
            feedback = (
                'Your previous proposals this turn were rejected before '
                'reaching the simulator. Fix the problem and try again:\n'
                + '\n'.join(rejected)
            )

        return None, None

#**********************************************************************#
