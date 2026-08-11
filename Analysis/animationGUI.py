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
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QButtonGroup,
    QRadioButton
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
                fieldDF[['Ex', 'Ey', 'Ez']] /= 1e3
                fieldDF['E'] = np.sqrt(fieldDF['Ex']**2 + fieldDF['Ey']**2 + fieldDF['Ez']**2)
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
        if hasattr(self, 'cbar') and self.cbar is not None:
            self.cbar.remove()
            self.cbar = None

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

# **********************************************************************#
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
        self.viewSelector.addItems([  # Match with order in _onViewChange
            'Simulation Parameters',
            'Field Lines',
            'Field Strengths',
            'Electron Avalanche',
            'Induced Signals',
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

        # ========================================
        # Params options
        viewWidgetParams = QWidget()
        self.controlsStack.addWidget(viewWidgetParams)

        # Field lines options
        viewWidgetFieldLines = QWidget()
        self.controlsStack.addWidget(viewWidgetFieldLines)

        # Field Strength options
        viewWidgetFieldStrengths = QWidget()
        layoutFieldStrengths = QVBoxLayout(viewWidgetFieldStrengths)
        layoutFieldStrengths.setContentsMargins(0, 0, 0, 0)

        self.chkEField = QRadioButton('Electric Field')
        self.chkWField = QRadioButton('Weighting Field')
        self.chkEField.setChecked(True)

        self.fieldGroup = QButtonGroup(self)
        self.fieldGroup.setExclusive(True)
        self.fieldGroup.addButton(self.chkEField)
        self.fieldGroup.addButton(self.chkWField)
        self.fieldGroup.buttonClicked.connect(self._onFieldModeChanged)

        layoutFieldStrengths.addWidget(self.chkEField)
        layoutFieldStrengths.addWidget(self.chkWField)
        self.controlsStack.addWidget(viewWidgetFieldStrengths)

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
        layoutSignals = QVBoxLayout(viewWidgetSignal)
        layoutSignals.setContentsMargins(0, 0, 0, 0)

        self.chkSignal = QRadioButton('Induced Signal')
        self.chkCharge = QRadioButton('Total Charge')
        self.chkSignal.setChecked(True)

        self.signalGroup = QButtonGroup(self)
        self.signalGroup.setExclusive(True)
        self.signalGroup.addButton(self.chkSignal)
        self.signalGroup.addButton(self.chkCharge)
        self.signalGroup.buttonClicked.connect(self._onSignalChange)

        layoutSignals.addWidget(self.chkSignal)
        layoutSignals.addWidget(self.chkCharge)
        self.controlsStack.addWidget(viewWidgetSignal)
        # ========================================

        sidebarLayout.addWidget(self.controlsStack)
        sidebarLayout.addStretch()

        # Reload File Button
        reloadButton = QPushButton('Reload ROOT File')
        reloadButton.clicked.connect(self._reloadData)
        sidebarLayout.addWidget(reloadButton)

        mainLayout.addWidget(sidebar, stretch=1)

        # --- RIGHT SIDE (Display Area) ---
        displayPanel = QWidget()
        displayLayout = QVBoxLayout(displayPanel)

        # Simplified display stack: 0 = Parameters Table, 1 = Shared Matplotlib Canvas
        self.displayStack = QStackedWidget()

        # View 0: Table
        self.simParamTable = QTableWidget()
        self.simParamTable.setColumnCount(2)
        self.simParamTable.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.simParamTable.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.simParamTable.verticalHeader().setVisible(False)
        self.displayStack.addWidget(self.simParamTable)

        # View 1: Single Shared Canvas
        self.canvas = MplCanvas()
        self.displayStack.addWidget(self.canvas)

        # Navigation Toolbar tied directly to the single canvas
        self.toolbar = NavigationToolbar(self.canvas, self)

        displayLayout.addWidget(self.toolbar)
        displayLayout.addWidget(self.displayStack)

        mainLayout.addWidget(displayPanel, stretch=4)

        # Render initial view (Parameters table)
        self._plotSimParams()

        return

    # ==========================================
    # SLOTS AND RENDER METHODS
    # ==========================================
    
#**********************************************************************#
    def _onViewChange(self, index):
        """Handles main view changes from the viewSelector dropdown."""
        # Stop animation when leaving/entering views
        self.animationTimer.stop()
        self.playButton.setChecked(False)
        self.playButton.setText('Play Animation')

        # Update control sidebar options
        self.controlsStack.setCurrentIndex(index)

        # Enable 2D/3D toggle
        is3DView = index in (1, 2, 3)
        self.chk3D.setEnabled(is3DView)
        self.chk2D.setEnabled(is3DView)

        # Enable Geometry toggle
        isGeo = index in (1, 2, 3)
        self.chkGeometry.setEnabled(isGeo)

        if index == 0:
            # Show parameters table (index 0 in displayStack)
            self.displayStack.setCurrentIndex(0)
            self.toolbar.hide()
            self._plotSimParams()
        else:
            # Show Matplotlib canvas (index 1 in displayStack)
            self.displayStack.setCurrentIndex(1)
            self.toolbar.show()

            # Match exact index order from _init_UI()
            if index == 1:
                self._plotFieldLines()
            elif index == 2:
                self._plotFields()
            elif index == 3:
                self._resetAvalancheAnimation()
            elif index == 4:
                self._plotSignals()

        return

#**********************************************************************#
    def _onProjectionModeChanged(self):
        """Re-render current view when switching between 3D and 2D mode."""
        idx = self.viewSelector.currentIndex()
        if idx == 1:
            self._plotFieldLines()
        elif idx == 2:
                    self._plotFields()
        elif idx == 3:
            self._renderParticleFrame()
        return
    
#**********************************************************************#
    def _onGeometryToggled(self):
        """Triggers re-render when the geometry overlay is toggled."""
        idx = self.viewSelector.currentIndex()
        if idx == 1:
            self._plotFieldLines()
        elif idx == 2:
            self._plotFields()
        elif idx == 3:
            self._renderParticleFrame()
        return

#**********************************************************************#
    def _onFieldModeChanged(self, *args):
        """Triggers re-render when the field is changed."""
        idx = self.viewSelector.currentIndex()
        if idx==2:
            self._plotFields()
        return

#**********************************************************************#
    def _onSignalChange(self, *args):
        """Triggers re-render when the signal is changed."""
        idx = self.viewSelector.currentIndex()
        if idx==4:
            self._plotSignals()
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
            xz, yz, xy = self.canvas.setupAxes(is2D=True)

            for (lineID, startVal), lineData in groupedLines:
                color = colorMap.get(startVal, 'k')
                xz.plot(lineData['x'], lineData['z'], c=color)
                yz.plot(lineData['y'], lineData['z'], c=color)
                xy.plot(lineData['x'], lineData['y'], c=color)            

            if self.chkGeometry.isChecked():
                self._drawGeometry((xz, yz, xy))
            self._formatAxes((xz, yz, xy))

        else:
            ax = self.canvas.setupAxes(is2D=False)

            for (lineID, startVal), lineData in groupedLines:
                ax.plot(
                    lineData['x'], lineData['y'], lineData['z'], 
                    c=colorMap.get(startVal, 'k')
                )

            if self.chkGeometry.isChecked():
                self._drawGeometry(ax)
            self._formatAxes(ax)
            

        self.canvas.fig.tight_layout()
        self.canvas.draw()
        return

#**********************************************************************#
    def _plotFields(self):
        """TODO"""

        use2D = self.chk2D.isChecked()
        isEField = self.chkEField.isChecked()
        fieldData = self.data.fieldStrengths.copy()

        plotData = fieldData['E'] if isEField else fieldData['Weighting']

        if isEField:
            vmin = np.nanmin(plotData)
            vmax = np.nanmax(plotData)
            lineLevels = None
        else:
            vmin, vmax = 0.0, 1.0
            lineLevels = np.arange(0.2, 1.2, 0.2)

        contour = None
        if use2D:
            xz, yz, xy = self.canvas.setupAxes(is2D=True)
        
            xzMask = np.isclose(fieldData['y'], 0, atol=1e-6)
            contour = xz.tricontourf(
                fieldData['x'][xzMask], fieldData['z'][xzMask], plotData[xzMask], 
                levels=101, cmap='viridis', vmin=vmin, vmax=vmax
            )
            if not isEField:
                xzLines = xz.tricontour(
                    fieldData['x'][xzMask], fieldData['z'][xzMask], plotData[xzMask], 
                    levels=lineLevels, colors='c', lw=0.5, vmin=vmin, vmax=vmax
                )
                xz.clabel(xzLines, inline=True, fontsize=8, fmt='%.1f')

            yzMask = np.isclose(fieldData['x'], 0, atol=1e-6)
            contour = yz.tricontourf(
                fieldData['y'][yzMask], fieldData['z'][yzMask], plotData[yzMask], 
                levels=101, cmap='viridis', vmin=vmin, vmax=vmax
            )
            if not isEField:
                yzLines = yz.tricontour(
                    fieldData['y'][yzMask], fieldData['z'][yzMask], plotData[yzMask], 
                    levels=lineLevels, colors='c', lw=0.5, vmin=vmin, vmax=vmax
                )
                yz.clabel(yzLines, inline=True, fontsize=8, fmt='%.1f')

            if self.chkGeometry.isChecked():
                self._drawGeometry((xz, yz, xy))
            self._formatAxes((xz, yz, xy))
            xy.set_visible(False)

            if contour is not None:
                self.cbar = self.canvas.fig.colorbar(
                    contour, 
                    ax=[xz, yz], 
                    orientation='vertical', 
                    fraction=0.03, 
                    pad=0.04
                )
            
        else:
            ax = self.canvas.setupAxes(is2D=False)

            mappable = ax.scatter(
                fieldData['x'], fieldData['y'], fieldData['z'],
                c=plotData, cmap='viridis', s=15
            )

            if self.chkGeometry.isChecked():
                self._drawGeometry(ax)
            self._formatAxes(ax)

            self.cbar = self.canvas.fig.colorbar(
                mappable,
                ax=ax,
                orientation='vertical',
                fraction=0.03,
                pad=0.04
            )

        if self.cbar is not None:
            label = 'Field Strength (kV/cm)' if isEField else 'Weighting Potential'
            self.cbar.set_label(label)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        return

#**********************************************************************#
    def _resetAvalancheAnimation(self):
        self.animationTimer.stop()
        self.playButton.setChecked(False)
        self.playButton.setText('Play Animation')

        if self.data.particleData is None or self.data.particleData.empty:
            ax = self.canvas.ax
            ax.clear()
            ax.text(0, 0, 0, 'No Particle Data Found', color='r')
            self.canvas.draw()
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
            ax = self.canvas.setupAxes(is2D=False)
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

            self.canvas.draw()

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
        #TODO: checkboxes for total/ion/electron signals.
        #All signals or just individual pads

        isSignal = self.chkSignal.isChecked()

        xz, yz, xy = self.canvas.setupAxes(is2D=True)

        inData = self.data.signalData

        # Plot all dynamic signal branches
        for col in inData.columns:
            if col in ['AvalancheID', 'Time']:
                continue
            xy.plot(
                inData['Time'], 
                inData[col] if isSignal else inData[col].cumsum(),
                label=col
            )

        xy.set_xlabel('Time (ns)')
        xy.set_ylabel('Signal (fC/ns)' if isSignal else 'Charge (fC)')
        xy.grid()
        xy.legend(loc='upper right')

        xz.set_visible(False)
        yz.set_visible(False)

        self.canvas.draw()

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
