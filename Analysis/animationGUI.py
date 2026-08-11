import sys
import os
import uproot
import pandas as pd
import numpy as np
import awkward as ak
import math

import matplotlib
import matplotlib.patches as patches
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QLabel, QSpinBox, QPushButton, QStackedWidget, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QButtonGroup
)
from PyQt6.QtCore import Qt, QTimer

CMTOMICRON = 1e4



# ==========================================
# DATA HANDLER CLASS
# ==========================================
class AnimationData:
    DEFAULT_FILENAME = 'animationData.root'

    def __init__(self, fileName: str = None):
        self.fileName = fileName if fileName else self.DEFAULT_FILENAME
        self.simData = None
        self.avalancheData = None
        self.fieldStengths = None
        self.fieldLines = None
        self.particleData = None
        self.signalData = None
        
        self.loadRootData()

        return

#**********************************************************************#
    def loadRootData(self):
        """
        Reads ROOT trees into Pandas DataFrames.
        """
        dataPath = '../Data/'
        filePath = os.path.join(dataPath, self.fileName)
        
        if not os.path.exists(filePath):
            print(f"Warning: '{filePath}' does not exist.")
            return

        with uproot.open(filePath) as file:

            if 'simDataTree' in file:
                tree = file['simDataTree']
                self.simData = {
                    key: val[0] 
                    for key, val in tree.arrays(library='np').items()
                }

            if 'avalancheTree' in file:
                self.avalancheData = file['avalancheTree'].arrays(
                    ['AvalancheID', 'Gain'], 
                    library='pd'
                )

            if 'fieldTree' in file:
                fieldDF = file['fieldTree'].arrays(
                    ['x', 'y', 'z', 'Ex', 'Ey', 'Ez', 'Weighting'], 
                    library='pd'
                )
                fieldDF[['x', 'y', 'z']] *= CMTOMICRON
                self.fieldStrengths = fieldDF

            if 'fieldLineTree' in file:
                lineDF = file['fieldLineTree'].arrays(
                    ['FieldLineID', 'FieldStart', 'x', 'y', 'z'],
                    library='pd'
                )
                lineDF[['x', 'y', 'z']] *= CMTOMICRON
                self.fieldLines = lineDF

            if 'particleDataTree' in file:
                pData = file['particleDataTree'].arrays(
                    ['AvalancheID', 'FrameID', 'Time', 'ParticleType', 'x', 'y', 'z'],
                    library='ak'
                )
                particleDF = self._flattenBranch(pData)
                particleDF[['x', 'y', 'z']] *= CMTOMICRON
                self.particleData = particleDF

            if 'signalDataTree' in file:
                signalTree = file['signalDataTree']
                self.signalData = signalTree.arrays(
                    signalTree.keys(), 
                    library='pd'
                )

        return

#**********************************************************************#
    @staticmethod
    def _flattenBranch(pData) -> pd.DataFrame:
        """
        Flattens nested C++ std::vector particle branches.
        """

        flatAvalanche = ak.flatten(ak.broadcast_arrays(pData['AvalancheID'], pData['x'])[0])
        flatFrame = ak.flatten(ak.broadcast_arrays(pData['FrameID'], pData['x'])[0])
        flatTime = ak.flatten(ak.broadcast_arrays(pData['Time'], pData['x'])[0])

        allData = {
            'AvalancheID': ak.to_numpy(flatAvalanche),
            'FrameID': ak.to_numpy(flatFrame),
            'Time': ak.to_numpy(flatTime),
            'ParticleType': ak.to_numpy(ak.flatten(pData['ParticleType'])),
            'x': ak.to_numpy(ak.flatten(pData['x'])),
            'y': ak.to_numpy(ak.flatten(pData['y'])),
            'z': ak.to_numpy(ak.flatten(pData['z']))
        }
        return pd.DataFrame(allData)


# ==========================================
# MATPLOTLIB CANVAS CANVAS WIDGET
# ==========================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, is3D=True):
        self.fig = Figure()
        super().__init__(self.fig)
        self.is3D = is3D

        self.ax = self.setupAxes(is2D=False)
        return

    def setupAxes(self, is2D=False):
        """Clears the figure and prepares 3D or dual 2D subplot axes."""
        self.fig.clear()

        if is2D:
            gs = self.fig.add_gridspec(2, 2)
            xz = self.fig.add_subplot(gs[0, 0])
            yz = self.fig.add_subplot(gs[1, 0])
            xy = self.fig.add_subplot(gs[:, 1])
            return xz, yz, xy
        else:
            if self.is3D:
                self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
            else:
                self.ax = self.fig.add_subplot(1, 1, 1)
            return self.ax


# ==========================================
# MAIN GUI WINDOW
# ==========================================
class FIMSVisualizer(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FIMS Visualizer')
        self.resize(1500, 750)

        # Load Data
        self.data = AnimationData()

        # Animation states
        self.animationTimer = QTimer()
        self.animationTimer.timeout.connect(self._nextFrame)
        self.curFrameID = 0
        self.allFrames = []

        self._init_UI()

        return

#**********************************************************************#
    def _init_UI(self):
        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)
        mainLayout = QHBoxLayout(mainWidget)

        # --- LEFT SIDEBAR (Controls) ---
        sidebar = QGroupBox('Visualization Settings')
        sidebarLayout = QVBoxLayout(sidebar)

        # Data Selection
        sidebarLayout.addWidget(QLabel('Select Data:'))
        self.viewSelector = QComboBox()
        self.viewSelector.addItems([ # Match with order in _onViewChange
            'Simulation Parameters', 
            'Field Lines', 
            'Electron Avalanche', 
            'Induced Signals'
        ])
        self.viewSelector.currentIndexChanged.connect(self._onViewChange)
        sidebarLayout.addWidget(self.viewSelector)

        # Projection Mode - 2D vs 3D
        projLayout = QHBoxLayout()
        self.chk3D = QCheckBox('3D View')
        self.chk2D = QCheckBox('2D Projections')
        self.chk3D.setChecked(True)

        self.projGroup = QButtonGroup(self)
        self.projGroup.addButton(self.chk3D)
        self.projGroup.addButton(self.chk2D)

        self.chk3D.toggled.connect(self._onProjectionModeChanged)

        projLayout.addWidget(self.chk3D)
        projLayout.addWidget(self.chk2D)
        sidebarLayout.addLayout(projLayout)

        # Show Geometry
        self.chkGeometry = QCheckBox('Show Geometry')
        self.chkGeometry.setChecked(True)
        self.chkGeometry.toggled.connect(self._onGeometryToggled)
        sidebarLayout.addWidget(self.chkGeometry)

        # Controls Stack for mode-specific options
        self.controlsStack = QStackedWidget()

        #========================================
        # Params options
        viewWidgetParams = QWidget()
        self.controlsStack.addWidget(viewWidgetParams)

        # Field lines options
        viewWidgetFieldLines = QWidget()
        self.controlsStack.addWidget(viewWidgetFieldLines)

        # Avalanche Animation controls
        viewWidgetAvalanche = QWidget()
        layoutViewAvalanche = QVBoxLayout(viewWidgetAvalanche)
        layoutViewAvalanche.setContentsMargins(0, 0, 0, 0)
        
        layoutViewAvalanche.addWidget(QLabel('AvalancheID:'))
        self.avalancheSpinBox = QSpinBox()
        if self.data.particleData is not None and not self.data.particleData.empty:
            maxID = int(self.data.particleData['AvalancheID'].max())
            self.avalancheSpinBox.setRange(0, maxID)
        else:
            self.avalancheSpinBox.setRange(0, 0)
            self.avalancheSpinBox.setEnabled(False)
        self.avalancheSpinBox.valueChanged.connect(self._resetAvalancheAnimation)
        layoutViewAvalanche.addWidget(self.avalancheSpinBox)

        self.playButton = QPushButton('Play Animation')
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self._toggleAnimation)
        layoutViewAvalanche.addWidget(self.playButton)
        
        self.controlsStack.addWidget(viewWidgetAvalanche)

        # Signals controls
        viewWidgetSignal = QWidget()
        self.controlsStack.addWidget(viewWidgetSignal)
        #========================================

        sidebarLayout.addWidget(self.controlsStack)
        sidebarLayout.addStretch()

        # Reload File Button
        reloadButton = QPushButton('Reload ROOT File')
        reloadButton.clicked.connect(self._reloadData)
        sidebarLayout.addWidget(reloadButton)

        mainLayout.addWidget(sidebar, stretch=1)

        # --- RIGHT SIDE (Display Stack) ---
        displayPanel = QWidget()
        displayLayout = QVBoxLayout(displayPanel)

        self.canvasStack = QStackedWidget()

        #========================================
        # Simulation Parameters Table Widget
        self.simParamTable = QTableWidget()
        self.simParamTable.setColumnCount(2)
        self.simParamTable.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.simParamTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.simParamTable.verticalHeader().setVisible(False)
        self.canvasStack.addWidget(self.simParamTable)

        # Field lines
        self.canvasField = MplCanvas(is3D=True)
        self.canvasStack.addWidget(self.canvasField)

        # Avalanche
        self.canvasAvalanche = MplCanvas(is3D=True)
        self.canvasStack.addWidget(self.canvasAvalanche)

        # Signals
        self.canvasSignal = MplCanvas(is3D=False)
        self.canvasStack.addWidget(self.canvasSignal)
        #========================================

        # Navigation Toolbar initialized
        self.toolbar = NavigationToolbar(self.canvasField, self)
        self.toolbar.hide()
        
        displayLayout.addWidget(self.toolbar)
        displayLayout.addWidget(self.canvasStack)

        mainLayout.addWidget(displayPanel, stretch=4)

        # Render initial view (Parameters table)
        self._plotSimParams()

        return

    # ==========================================
    # SLOTS AND RENDER METHODS
    # ==========================================
    
#**********************************************************************#
    def _onViewChange(self, index):
        self.animationTimer.stop()
        self.playButton.setChecked(False)
        self.playButton.setText('Play Animation')
        
        self.controlsStack.setCurrentIndex(index)
        self.canvasStack.setCurrentIndex(index)

        is3DView = index in (1, 2)
        self.chk3D.setEnabled(is3DView)
        self.chk2D.setEnabled(is3DView)

        # Match index order from _init_UI()
        if index == 0:
            self.toolbar.hide()
            self._plotSimParams()
        else:
            currentCanvas = self.canvasStack.currentWidget()
            self.toolbar.canvas = currentCanvas
            self.toolbar.update()
            self.toolbar.show()

            # Plot something
            if index == 1:
                self._plotFieldLines()
            elif index == 2:
                self._resetAvalancheAnimation()
            elif index == 3:
                self._plotSignals()

        return

#**********************************************************************#
    def _onProjectionModeChanged(self):
        """Re-render current view when switching between 3D and 2D mode."""
        idx = self.viewSelector.currentIndex()
        if idx == 1:
            self._plotFieldLines()
        elif idx == 2:
            self._renderParticleFrame()
        return
    
#**********************************************************************#
    def _onGeometryToggled(self):
        """Triggers re-render when the geometry overlay is toggled."""
        idx = self.viewSelector.currentIndex()
        if idx == 1:
            self._plotFieldLines()
        elif idx == 2:
            self._renderParticleFrame()
        return
    
#**********************************************************************#
    def _reloadData(self):
        self.data.loadRootData()
        self._onViewChange(self.viewSelector.currentIndex())
        return



# --- Plotting logic ---
#**********************************************************************#
    def _plotSimParams(self):
        """Populates the main display table with simulation parameters."""
        # Clear existing rows
        self.simParamTable.setRowCount(0)
        if not self.data.simData:
            return

        self.simParamTable.setRowCount(len(self.data.simData))
        # Populate parameter names and values
        for row, (key, value) in enumerate(self.data.simData.items()):
            valStr = f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
            
            keyItem = QTableWidgetItem(str(key))
            valItem = QTableWidgetItem(valStr)
            
            keyItem.setFlags(keyItem.flags() ^ Qt.ItemFlag.ItemIsEditable)
            valItem.setFlags(valItem.flags() ^ Qt.ItemFlag.ItemIsEditable)

            self.simParamTable.setItem(row, 0, keyItem)
            self.simParamTable.setItem(row, 1, valItem)

        return
    
#**********************************************************************#
    def _plotFieldLines(self):

        use2D = self.chk2D.isChecked()

        colorMap = {
            0: 'blue',      #Cathode
            1: 'red',       #Above the grid
            -1: 'green'     #Below the grid
        }


        groupedLines = self.data.fieldLines.groupby(['FieldLineID', 'FieldStart'])

        if use2D:
            xz, yz, xy = self.canvasField.setupAxes(is2D=True)
            if self.data.fieldLines is None or self.data.fieldLines.empty:
                xy.text(0, 0, 0, 'No Field Line Data Found', color='r')
                return

            for (lineID, startVal), lineData in groupedLines:
                color = colorMap.get(startVal, 'k')
                xz.plot(lineData['x'], lineData['z'], c=color)
                yz.plot(lineData['y'], lineData['z'], c=color)
                xy.plot(lineData['x'], lineData['y'], c=color)            

            self._formatAxes((xz, yz, xy))
            if self.chkGeometry.isChecked():
                self._drawGeometry((xz, yz, xy))

        else:
            ax = self.canvasField.setupAxes(is2D=False)
            if self.data.fieldLines is None or self.data.fieldLines.empty:
                ax.text(0, 0, 0, 'No Field Line Data Found', color='r')
                return

            for (lineID, startVal), lineData in groupedLines:
                ax.plot(
                    lineData['x'], lineData['y'], lineData['z'], 
                    c=colorMap.get(startVal, 'k')
                )

            self._formatAxes(ax)
            if self.chkGeometry.isChecked():
                self._drawGeometry(ax)

        self.canvasField.fig.tight_layout()
        self.canvasField.draw()
        return

#**********************************************************************#
    def _resetAvalancheAnimation(self):
        self.animationTimer.stop()
        self.playButton.setChecked(False)
        self.playButton.setText('Play Animation')

        if self.data.particleData is None or self.data.particleData.empty:
            ax = self.canvasAvalanche.ax
            ax.clear()
            ax.text(0, 0, 0, 'No Particle Data Found', color='r')
            self.canvas3DPart.draw()
            return

        avID = self.avalancheSpinBox.value()
        self.inData = self.data.particleData[self.data.particleData['AvalancheID'] == avID]
        self.allFrames = sorted(self.inData['FrameID'].unique())
        self.curFrameID = 0

        self._renderParticleFrame()

        return

#**********************************************************************#
    def _renderParticleFrame(self):
        if not self.allFrames:
            return

        use2D = self.chk2D.isChecked()

        frameID = self.allFrames[self.curFrameID]
        inFrameData = self.inData[self.inData['FrameID'] == frameID]

        elec = inFrameData[inFrameData['ParticleType'] == 0]
        ions = inFrameData[inFrameData['ParticleType'] != 0]


        if use2D:
            pass
        else:
            ax = self.canvasField.setupAxes(is2D=False)
            ax.clear()
            
            ax.scatter(
                elec['x'], elec['y'], elec['z'], 
                c='b', s=8, label='Electrons'
            )
            ax.scatter(
                ions['x'], ions['y'], ions['z'], 
                c='r', s=12, label='Ions'
            )

            # Fix spatial limits using full dataset bounds -- TODO use pitch?
            df = self.inData
            ax.set_xlim(df['x'].min(), df['x'].max())
            ax.set_ylim(df['y'].min(), df['y'].max())
            ax.set_zlim(df['z'].min(), df['z'].max())

            inTime = inFrameData['Time'].iloc[0] if not inFrameData.empty else 0.0
            ax.set_title(f'Avalanche {self.avalancheSpinBox.value()} | Frame {frameID} ({inTime:.2f} ns)')
            ax.set_xlabel(r'x ($\mu$m)')
            ax.set_ylabel(r'y ($\mu$m)')
            ax.set_zlabel(r'z ($\mu$m)')
            ax.legend(loc='upper right')

            self.canvas3DPart.draw()

        return

#**********************************************************************#
    def _toggleAnimation(self, checked):
        if checked:
            self.playButton.setText('Pause')
            self.animationTimer.start(80)  # ~12 FPS frame rate
        else:
            self.playButton.setText('Play Animation')
            self.animationTimer.stop()

        return

#**********************************************************************#
    def _nextFrame(self):
        if not self.allFrames:
            return
        self.curFrameID = (self.curFrameID + 1) % len(self.allFrames)
        self._renderParticleFrame()

        return

#**********************************************************************#
    def _plotSignals(self):
        ax = self.canvas2DSignal.ax
        ax.clear()

        if self.data.signalData is None or self.data.signalData.empty:
            ax.text(0.5, 0.5, 'No Signal Data Found', ha='center', va='center', color='r')
        else:
            inData = self.data.signalData
            time = inData['Time'] if 'Time' in inData.columns else inData.index

            # Plot all dynamic signal branches
            for col in inData.columns:
                if col in ['AvalancheID', 'Time']:
                    continue
                ax.plot(time, inData[col], label=col)

            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Signal')
            ax.set_title('Induced Signals')
            ax.grid()
            ax.legend(loc='upper right')

        self.canvas2DSignal.draw()

        return

#**********************************************************************#
    def _drawGeometry(self, axes):
        """
        TODO
        """
        if self.data.simData is None:
            return
        
        padLength = self.data.simData['padLength']*CMTOMICRON
        
        self._addGrid(axes)
        self._addPads(axes)
        self._addUnitCell(axes)

        return

#**********************************************************************#
    def _addGrid(self, axes):
        pitch = self.data.simData['pitch']*CMTOMICRON
        holeRadius = self.data.simData['holeRadius']*CMTOMICRON
        gridThickness = self.data.simData['gridThickness']*CMTOMICRON
        
        gridSize = gridThickness/2

        sqrt3 = math.sqrt(3)
        centers = pitch * np.array([
            (0,0),
            (0, 1), (0, -1),
            (sqrt3/2, .5), (sqrt3/2, -.5),
            (-sqrt3/2, .5), (-sqrt3/2, -.5),
        ])

        xScale = pitch*sqrt3/2
        yScale = pitch
                    
        gridRes = 501
        x = np.linspace(-xScale, xScale, gridRes)
        y = np.linspace(-yScale, yScale, gridRes)
        xGrid, yGrid = np.meshgrid(x, y)

        inHole = np.zeros(xGrid.shape, dtype=bool)
        holeR2 = holeRadius**2
        for center in centers:
            R2 = (xGrid - center[0])**2 + (yGrid - center[1])**2
            inHole |= R2 < holeR2

        zTop = np.ma.masked_where(inHole, np.full_like(xGrid, gridSize))
        zBot = np.ma.masked_where(inHole, np.full_like(xGrid, -gridSize))

        if isinstance(axes, tuple):
            xz, yz, xy = axes

            gridMask = np.ma.masked_where(inHole, np.ones_like(xGrid))
            midIdx = gridRes // 2

            xz.fill_between(
                x, zBot[midIdx, :], zTop[midIdx, :], 
                color='grey'
            )
            yz.fill_between(
                y, zBot[:, midIdx], zTop[:, midIdx], 
                color='grey'
            )
            xy.pcolormesh(
                xGrid, yGrid, gridMask, 
                cmap='Greys', vmin=0, vmax=2, alpha=0.5, shading='auto'
            )

        else:
            ax = axes

            ax.plot_surface(
                xGrid, yGrid, zTop,
                color='grey', alpha=0.5, edgecolor='none', shade=True
            )
            ax.plot_surface(
                xGrid, yGrid, zBot,
                color='grey', alpha=0.5, edgecolor='none', shade=True
            )

        return

#**********************************************************************#
    def _addPads(self, axes):
        pitch = self.data.simData['pitch']*CMTOMICRON
        padLength = self.data.simData['padLength']*CMTOMICRON

        #padThickness = self.data.simData['padThickness']*CMTOMICRON
        amplificationGap = self.data.simData['amplificationGap']*CMTOMICRON

        zHeight = -amplificationGap

        sqrt3 = math.sqrt(3)
        centers = pitch * np.array([
            (0,0),
            (0, 1), (0, -1),
            (sqrt3/2, .5), (sqrt3/2, -.5),
            (-sqrt3/2, .5), (-sqrt3/2, -.5),
        ])

        if isinstance(axes, tuple):
            xz, yz, xy = axes

            for (i, j) in centers:
                xLocs, yLocs = hexXY(padLength, i, j)
                line = '-' if i==0 and j==0 else ':'
                xz.plot(xLocs, zHeight*np.ones(len(xLocs)), c='m', ls=line)
                yz.plot(yLocs, zHeight*np.ones(len(yLocs)), c='m', ls=line)
                xy.plot(xLocs, yLocs, c='m', ls=line)

        else:
            ax = axes

            for (i, j) in centers:
                xLocs, yLocs = hexXY(padLength, i, j)
                line = '-' if i==0 and j==0 else ':'
                ax.plot(
                    xLocs, yLocs, zHeight*np.ones(len(xLocs)),
                    c='m', ls=line
                )

        return

#**********************************************************************#
    def _addUnitCell(self, axes):
        pitch = self.data.simData['pitch']*CMTOMICRON

        sqrt3 = math.sqrt(3)
        xLocs, yLocs = hexXY(pitch/sqrt3, 0, 0)

        amplificationGap = self.data.simData['amplificationGap']*CMTOMICRON
        driftLength = self.data.simData['driftLength']*CMTOMICRON
        padHeight = -amplificationGap#Not quite but okay

        if isinstance(axes, tuple):
            xz, yz, xy = axes
            xy.plot(xLocs, yLocs, c='c')

        else:
            ax = axes
            ax.plot(xLocs, yLocs, 0*np.ones(len(xLocs)), c='c')
            ax.plot(xLocs, yLocs, padHeight*np.ones(len(xLocs)), c='c')
            ax.plot(xLocs, yLocs, driftLength*np.ones(len(xLocs)), c='c')

        return

#**********************************************************************#
    def _formatAxes(self, axes):
        """
        TODO
        """
        xLabel = r'x ($\mu$m)'
        yLabel = r'y ($\mu$m)'
        zLabel = r'z ($\mu$m)'

        pitch = self.data.simData['pitch']*CMTOMICRON
        xScale = pitch*math.sqrt(3)/2
        yScale = pitch
        
        if isinstance(axes, tuple):
            xz, yz, xy = axes
        
            xz.set_xlabel(xLabel)
            xz.set_ylabel(zLabel)
            xz.set_xlim([-xScale, xScale])

            yz.set_xlabel(yLabel)
            yz.set_ylabel(zLabel)
            yz.set_xlim([-yScale, yScale])

            xy.set_xlabel(xLabel)
            xy.set_ylabel(yLabel)
            xy.set_xlim([-xScale, xScale])
            xy.set_ylim([-yScale, yScale])
            xy.set_aspect('equal')

            for ax in axes:
                ax.grid(alpha=.5, ls=':')

        else:
            ax = axes

            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.set_zlabel(zLabel)

            ax.set_xlim([-xScale, xScale])
            ax.set_ylim([-yScale, yScale])

            ax.grid(alpha=.5, ls=':')

        return


# ==========================================
# Helper functions
# ==========================================
def hexXY(length, xCenter, yCenter):
    sqrt3 = math.sqrt(3)

    xLocs = length * np.array([1, .5, -.5, -1, -.5, .5, 1]) + xCenter
    yLocs = length * sqrt3 * np.array([0, .5, .5, 0, -.5, -.5, 0]) + yCenter

    return xLocs, yLocs

# ==========================================
# Main to run
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FIMSVisualizer()
    window.show()
    sys.exit(app.exec())
