"""
AI-agent driven optimization of the FIMS amplification-grid hole shape.

Run from the Simulation directory:

    export ANTHROPIC_API_KEY=...
    python optimizeHoleShape.py --max-iterations 25

History is appended to ../Data/holeShapeHistory.jsonl after every
iteration (crash safe, resumable) and mirrored to
../Data/holeShapeHistory.csv for human review and plotting.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import pandas as pd
import anthropic

from simulationClass import FIMS_Simulation
from configs import GeometryConfiguration, UnitCell, HoleShape, PadShape, ScaleOption

simDir = os.getcwd()
analysisDir = os.path.join(simDir, '..', 'Analysis')
sys.path.append(analysisDir)
from runDataClass import runData

#*******************## AI-Agent Information ##**************************#

aiModel = os.environ.get('FIMS_AGENT_MODEL', 'claude-opus-4-5') # TODO: get model number

#**********************************************************************#

aiPrompt = """
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
   code, or change any other parameter. Pitch, grid thickness, gas mixture and
   field ratio are set outside your control.
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

#**********************************************************************#

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
 
#**********************************************************************#

shapeTool = {
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
 
#**********************************************************************#

def describeConstraints(simObject):
    """
    Builds the geometric constraint block from the live parameters.
 
    args:
        simObject: The configured simulation object.
 
    returns:
        constraints (str): The rendered constraints.
    """
    pitch = float(simObject.getParam('pitch'))
    padLength = float(simObject.getParam('padLength'))
 
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

def loadHistory(path):
    """
    Reads an existing JSONL history file.

    Args:
        path (str): Path to the history file.

    Returns:
        history (list): List of history records, empty if the file does not exist.
    """
    if not os.path.exists(path):
        return []

    history = []
    with open(path, 'r') as inFile:
        for line in inFile:
            line = line.strip()
            if line:
                history.append(json.loads(line))

    return history

#**********************************************************************#

def appendHistory(path, record):
    """
    Appends one record to the JSONL history.

    Args:
        path (str): Path to the history file.
        record (dict): The record to append.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, 'a') as outFile:
        outFile.write(json.dumps(record) + '\n')
        outFile.flush()
        os.fsync(outFile.fileno())

    return

#**********************************************************************#

def writeHistoryCsv(path, history):
    """
    Mirrors the history to a flat CSV.

    Args:
        history (list): List of history records.
        path (str): Destination CSV path.
    """
    allRows = []
    for entry in history:
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

    pd.DataFrame(allRows).to_csv(path, index=False)

    return

#**********************************************************************#

def bestEntry(history):
    """
    Returns the completed record with the lowest IBN, or None.

    Args:
        history (list): List of history records.

    Returns:
        dict or None: The best record so far.
    """
    completed = [e for e in history if e.get('status') == 'ok']
    if not completed:
        return None

    return min(completed, key=lambda e: e['ibn'])

#**********************************************************************#

def renderHistory(history, numDetailed=3):
    """
    Renders the trial history into a compact block for the model.

    A summary table covers every attempt; the full outline is included only
    for the best few and the most recent, which keeps the prompt bounded no
    matter how many iterations have run.

    Args:
        history (list): List of history records.
        numDetailed (int): How many of the best shapes to spell out in full.

    Returns:
        str: The rendered history.
    """
    if not history:
        return (
            'Nothing has been simulated yet. Propose a starting point and say '
            'what you expect from it.'
        )

    header = (
        f"{'it':>3}  {'status':<8} {'IBN':>10} {'+/-':>8} {'N':>2} "
        f"{'maxR':>6} {'minR':>6} {'area':>8} {'perim':>7} {'open':>6}  types"
    )
    allRows = [header, '-'*len(header)]

    for entry in history:
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
    completed = [e for e in history if e.get('status') == 'ok']
    detailIds = [
        e['iteration']
        for e in sorted(completed, key=lambda e: e['ibn'])[:numDetailed]
    ]
    for entry in history[-2:]:
        if entry.get('status') == 'ok' and entry['iteration'] not in detailIds:
            detailIds.append(entry['iteration'])

    if detailIds:
        allRows.append('')
        allRows.append('FULL OUTLINES (best first, then most recent)')
        for entry in history:
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

#***********************## Run Simulation ##***************************#

def parseArguments():
    """Parses the command line options."""
    parser = argparse.ArgumentParser(
        description='AI-agent optimization of the FIMS grid hole shape.'
    )
    
    parser.add_argument(
        '--max-iterations', dest='maxIterations', type=int, default=100,
        help='Number of simulations to run (default 100).'
    )
    
    parser.add_argument(
        '--max-proposals', dest='maxProposals', type = int, default=5,
        help='Number of times a proposal can be retried (default 5).'
    )

    parser.add_argument(
        '--resume', dest='resume', action='store_true',
        help='Continue from the existing history file.'
    )
    
    return parser.parse_args()

#**********************************************************************#

def proposeShape(client, model, history, iteration, maxIterations,
                 constraints, feedback=None):
    """
    Asks the model for exactly one new hole outline.

    Each iteration is a fresh single-turn call rather than a growing
    conversation. The history is re-rendered compactly every time, so the
    prompt stays the same size whether this is iteration 3 or 300.

    Args:
        client (anthropic.Anthropic): The API client.
        model (str): Model ID.
        history (list): List of history records.
        iteration (int): The iteration about to be run.
        maxIterations (int): Total iteration budget.
        constraints (str): Rendered geometric constraints.
        feedback (str): Text describing rejected attempts this iteration.
        
    Returns:
        dict: The tool input, containing 'shape', 'hypothesis', 'rationale'.

    Raises:
        RuntimeError: If the model returns no proposal.
    """
    best = bestEntry(history)
    bestText = (
        f'Best so far: iteration {best["iteration"]} at IBN = '
        f'{best["ibn"]:.5g} +/- {best["ibnError"]:.3g}.'
        if best else 'No successful run yet.'
    )

    userBlock = '\n\n'.join(filter(None, [
        f'Iteration {iteration} of {maxIterations}. {bestText}',
        'GEOMETRIC CONSTRAINTS\n' + constraints,
        'HISTORY\n' + renderHistory(history),
        feedback,
        'Propose the next outline.',
    ]))

    requestArgs = {
        'model': model,
        'max_tokens': 8000,
        'system': aiPrompt,
        'tools': [shapeTool],
        'messages': [{'role': 'user', 'content': userBlock}],
    }

    requestArgs['tool_choice'] = {
        'type': 'tool', 'name': shapeTool['name']
    }

    response = client.messages.create(**requestArgs)

    for block in response.content:
        if block.type == 'tool_use' and block.name == shapeTool['name']:
            return dict(block.input)

    raise RuntimeError('Error: Model returned no hole shape proposal.')

#**********************************************************************#

def runHoleShapeOptimizer():
    """Runs the optimization loop."""
    startTime = time.perf_counter()
    
    # Configure simulation
    args = parseArguments()
    client = anthropic.Anthropic()
    geoConfig = GeometryConfiguration(
        unitCell=UnitCell.HEXAGON,
        holeShape=HoleShape.CUSTOM,
        padShape=PadShape.HEXAGON,
        scale=ScaleOption.HALF,
    )
    
    FIMS = FIMS_Simulation()
    FIMS.setGeometry(geoConfig)
    constraints = describeConstraints(FIMS)
    
    # Setup history files
    aiPath = os.path.join('..', 'Data', 'AI')
    os.makedirs(aiPath, exist_ok=True)
    
    historyJSONL = os.path.join(aiPath, 'holeShapeHistory.jsonl')
    historyCSV = os.path.join(aiPath, 'holeShapeHistory.csv')
    idealShapePath = os.path.join(aiPath, 'bestHoleShape.json')

    if args.resume:
        history = loadHistory(historyJSONL)
        print(f'Resuming from {len(history)} recorded trials.')
    
    elif os.path.exists(historyJSONL):
        print(
            f'Warning: {historyJSONL} exists and will be appended to. '
            'Pass --resume to let the agent see those trials.'
        )
        history = []
    
    else:
        history = []
    
    # Get current iteration and final number
    startIteration = 1 + max(
        (elem.get('iteration', 0) for elem in history), default=0
    )
    finalIteration = startIteration + args.maxIterations - 1

    # Begin AI Agent loop
    for iteration in range(startIteration, finalIteration + 1):

        print(f'\n{"*"*72}')
        print(f'Starting Iteration {iteration} of {finalIteration}')
        print(f'{"*"*72}')

        # Get a proposal
        feedback = None
        rejected = []
        proposal = None
        for attempt in range(args.maxProposals):
            # First iteration is a circle
            if (iteration == startIteration and not history and attempt == 0):
                proposal = {
                    'shape': {
                        'holes': [{
                            'type': 'fourier',
                            'meanRadius': float(FIMS.getParam('holeRadius')),
                            'harmonics': [],
                            'numSamples': 120,
                            'offset': [0., 0.],
                        }],
                    },
                    'hypothesis': 'Baseline circular hole, seeded by the script.',
                    'rationale': 'Reference point for every later design.'
                }
            
            # Create a hole shape
            else:
                proposal = proposeShape(
                    client, aiModel, history, iteration, finalIteration,
                    constraints, feedback=feedback
                )
            shapeProposed = proposal['shape']

            try:
                shapeSummary = FIMS.createCustomShape(shapeProposed)
                
                # Ensure shape is new
                priorHashes = {e.get('shapeHash') for e in history if e.get('shapeHash')}
                if shapeSummary['shapeHash'] not in priorHashes:
                    break
                
                else:
                    message = 'This outline is already in the history.'
            
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                message = str(error)

            print(f'\t Rejected proposal: {message}')
            rejected.append(f'\t - {json.dumps(shapeProposed)}\n \t{message}')
            feedback = (
                'Your previous proposals this turn were rejected before '
                'reaching the simulator. Fix the problem and try again:\n'
                + '\n'.join(rejected)
            )
            shapeProposed = None
        ## end proposal loop ##    
        
        # Record shape details
        if shapeProposed is None:
            record = {
                'iteration': iteration,
                'status': 'rejected',
                'message': (
                    f'No valid proposal after {args.maxProposals} attempts.'
                ),
                'spec': None,
                'summary': None,
            }
            history.append(record)
            appendHistory(historyJSONL, record)
            writeHistoryCsv(historyCSV, history)
            print('Giving up on this iteration.')
            continue
        
        print(f'\tHypothesis: {proposal["hypothesis"]}')
        print(
            f'\tPattern: {shapeSummary["numHoles"]} hole(s), '
            f'maxR = {shapeSummary["maxRadius"]:.2f} um, '
            f'area = {shapeSummary["area"]:.1f} um^2, '
            f'open = {shapeSummary["openAreaFraction"]:.3f}'
        )
        for index, hole in enumerate(shapeSummary['holes']):
            print(
                f'\t  hole {index}: {hole["type"]:<8} offset '
                f'({hole["offset"][0]:+.2f}, {hole["offset"][1]:+.2f}) um, '
                f'area {hole["area"]:.1f} um^2, '
                f'perimeter {hole["perimeter"]:.1f} um'
            )

        # Run the simulation
        record = {
            'iteration': iteration,
            'shapeType': '+'.join(sorted({hole['type'] for hole in shapeSummary['holes']})),
            'shapeHash': shapeSummary['shapeHash'],
            'summary': shapeSummary,
            'spec': shapeProposed,
            'hypothesis': proposal['hypothesis'],
            'rationale': proposal['rationale'],
        }

        iterationStart = time.perf_counter()
        try:
            runNumber = FIMS.runForEfficiency()
            
            simData = runData(runNumber)
            ibn = float(simData.getCalcParameter('Average IBN'))
            ibnError = float(simData.getCalcParameter('IBN Error'))
            fieldRatio = float(simData.getRunParameter('fieldRatio'))
            
            record.update({
                'status': 'ok',
                'ibn': ibn,
                'ibnError': ibnError,
                'runNumber': runNumber,
                'fieldRatio': fieldRatio,
            })
            print(
                f'  Result: IBN = {ibn:.5g} +/- {ibnError:.3g} '
                f'(run {runNumber}, field ratio {fieldRatio:g})'
            )

        except Exception as error:
            record.update({
                'status': 'failed',
                'message': f'{type(error).__name__}: {error}',
            })
            print(f'\t Simulation failed: {record["message"]}')

        record['duration'] = round(time.perf_counter() - iterationStart, 1)
        print(f'\t Duration: {record["duration"]:.1f} s')
        
        # Update history files
        history.append(record)
        appendHistory(historyJSONL, record)
        writeHistoryCsv(historyCSV, history)

        best = bestEntry(history)
        if best is not None:
            print(
                f'Best so far: {best["iteration"]}:\n',
                f'\tIBN = {best["ibn"]:.5g} +/- {best["ibnError"]:.3g}'
            )
            with open(idealShapePath, 'w') as outFile:
                json.dump(best, outFile, indent=2)

    # Print a summary of all iterations
    print(f'\n{"*"*72}')
    completed = [e for e in history if e.get('status') == 'ok']
    print(
        f'{len(completed)} of {len(history)} trials produced an IBN. '
        f'Total agent run time: {(time.perf_counter() - startTime)/3600.:.2f} h'
    )

    best = bestEntry(history)
    if best is not None:
        print(
            f'Lowest IBN: {best["ibn"]:.5g} +/- {best["ibnError"]:.3g} '
            f'at iteration {best["iteration"]} (run {best.get("runNumber")})'
        )
        print(f'Best outline written to {idealShapePath}')

    print(f'History written to: {historyCSV}')

    return

#**********************************************************************#

runHoleShapeOptimizer()
