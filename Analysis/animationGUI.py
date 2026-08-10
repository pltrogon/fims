import sys
import os
import uproot
import pandas as pd
import numpy as np
import awkward as ak

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QLabel, QSpinBox, QPushButton, QStackedWidget, QGroupBox
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
    def __init__(self, is3D=True):
        self.fig = Figure(figsize=(8, 6))
        if is3D:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


# ==========================================
# MAIN GUI WINDOW
# ==========================================
class FIMSVisualizer(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FIMS Visualizer')
        self.resize(1000, 700)

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

        # Mode Selection
        sidebarLayout.addWidget(QLabel('Select View:'))
        self.viewSelector = QComboBox()
        self.viewSelector.addItems(['Field Lines 3D', 'Particle Avalanche 3D', 'Induced Signals 2D'])
        self.viewSelector.currentIndexChanged.connect(self._onViewChange)
        sidebarLayout.addWidget(self.viewSelector)

        # Controls Stack for mode-specific options
        self.controlsStack = QStackedWidget()
        
        # View 0: Field lines options
        viewWidget0 = QWidget()
        self.controlsStack.addWidget(viewWidget0)

        # View 1: Avalanche Animation controls
        viewWidget1 = QWidget()
        layoutView1 = QVBoxLayout(viewWidget1)
        layoutView1.setContentsMargins(0, 0, 0, 0)
        
        layoutView1.addWidget(QLabel('AvalancheID:'))
        self.avalancheSpinBox = QSpinBox()
        if self.data.particleData is not None and not self.data.particleData.empty:
            maxID = int(self.data.particleData['AvalancheID'].max())
            self.avalancheSpinBox.setRange(0, maxID)
        else:
            self.avalancheSpinBox.setRange(0, 0)
            self.avalancheSpinBox.setEnabled(False)
        self.avalancheSpinBox.valueChanged.connect(self._resetAvalancheAnimation)
        layoutView1.addWidget(self.avalancheSpinBox)

        self.playButton = QPushButton('Play Animation')
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self._toggleAnimation)
        layoutView1.addWidget(self.playButton)
        
        self.controlsStack.addWidget(viewWidget1)

        # View 2: Signals controls
        viewWidget2 = QWidget()
        self.controlsStack.addWidget(viewWidget2)

        sidebarLayout.addWidget(self.controlsStack)
        sidebarLayout.addStretch()

        # Reload File Button
        reloadButton = QPushButton('Reload ROOT File')
        reloadButton.clicked.connect(self._reloadData)
        sidebarLayout.addWidget(reloadButton)

        mainLayout.addWidget(sidebar, stretch=1)

        # --- RIGHT SIDE (Matplotlib Display Stack) ---
        displayPanel = QWidget()
        displayLayout = QVBoxLayout(displayPanel)

        self.canvasStack = QStackedWidget()
        
        # Canvas 0: 3D Field lines
        self.canvas3DField = MplCanvas(is3D=True)
        self.canvasStack.addWidget(self.canvas3DField)

        # Canvas 1: 3D Particles
        self.canvas3DPart = MplCanvas(is3D=True)
        self.canvasStack.addWidget(self.canvas3DPart)

        # Canvas 2: 2D Signals
        self.canvas2DSignal = MplCanvas(is3D=False)
        self.canvasStack.addWidget(self.canvas2DSignal)

        # Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas3DField, self)
        
        displayLayout.addWidget(self.toolbar)
        displayLayout.addWidget(self.canvasStack)

        mainLayout.addWidget(displayPanel, stretch=4)

        # Render initial view
        self._plotFieldLines()

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

        # Rebind active toolbar to canvas
        currentCanvas = self.canvasStack.currentWidget()
        self.toolbar.setParent(None)
        self.toolbar = NavigationToolbar(currentCanvas, self)

        if index == 0:
            self._plotFieldLines()
        elif index == 1:
            self._resetAvalancheAnimation()
        elif index == 2:
            self._plotSignals()

        return

#**********************************************************************#
    def _reloadData(self):
        self.data.loadRootData()
        self._onViewChange(self.viewSelector.currentIndex())
        return

    # --- Plotting logic ---
#**********************************************************************#
    def _plotFieldLines(self):
        ax = self.canvas3DField.ax
        ax.clear()

        colorMap = {
            0: 'blue',      #Cathode
            1: 'red',       #Above the grid
            -1: 'green'     #Below the grid
        }

        if self.data.fieldLines is None or self.data.fieldLines.empty:
            ax.text(0, 0, 0, 'No Field Line Data Found', color='r')
        else:
            grouped = self.data.fieldLines.groupby(['FieldLineID', 'FieldStart'])

            for (lineID, startVal), lineData in grouped:
                ax.plot(
                    lineData['x'], lineData['y'], lineData['z'], 
                    c=colorMap.get(startVal, 'k')
                )

            ax.set_xlabel(r'x ($\mu$m)')
            ax.set_ylabel(r'y ($\mu$m)')
            ax.set_zlabel(r'z ($\mu$m)')

        self.canvas3DField.draw()

        return

#**********************************************************************#
    def _resetAvalancheAnimation(self):
        self.animationTimer.stop()
        self.playButton.setChecked(False)
        self.playButton.setText('Play Animation')

        if self.data.particleData is None or self.data.particleData.empty:
            ax = self.canvas3DPart.ax
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

        ax = self.canvas3DPart.ax
        ax.clear()

        frameID = self.allFrames[self.curFrameID]
        inFrameData = self.inData[self.inData['frameID'] == frameID]

        elec = inFrameData[inFrameData['ParticleType'] == 0]
        ions = inFrameData[inFrameData['ParticleType'] != 0]

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


# ==========================================
# Main to run
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FIMSVisualizer()
    window.show()
    sys.exit(app.exec())
