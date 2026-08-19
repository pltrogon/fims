import sys
import os
import uproot
import pandas as pd
import numpy as np
import awkward as ak
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QLabel, QSpinBox, QPushButton, QStackedWidget, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QButtonGroup,
    QRadioButton, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer

CMTOMICRON = 1e4
VCMTOkVCM = 1e-3



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
        self.animationData = None
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

            # Simulation Metadata
            if 'simDataTree' in file:
                simDataDF = {
                    k: v[0]
                    for k, v in file['simDataTree'].arrays(library='np').items()
                }
                geoKeys = [
                    'padLength', 'pitch', 'holeRadius',
                    'amplificationGap', 'driftLength',
                    'gridThickness', 'padThickness',
                    'thicknessSiO2', 'pillarRadius'
                ]
                fieldKeys = ['driftField', 'amplificationField']
                for key in geoKeys:
                    if key in simDataDF:
                        simDataDF[key] *= CMTOMICRON
                for key in fieldKeys:
                    if key in simDataDF:
                        simDataDF[key] *= VCMTOkVCM
                self.simData = simDataDF

            # Avalanche Overview Data
            if 'avalancheDataTree' in file:
                self.avalancheData = file['avalancheDataTree'].arrays(
                    ['AvalancheID', 'Gain', 'numPosIons', 'numNegIons'], library='pd'
                )

            # Electric & Weighting Fields
            if 'fieldTree' in file:
                fieldDF = file['fieldTree'].arrays(library='pd')
                fieldDF[['x', 'y', 'z']] *= CMTOMICRON
                fieldDF[['Ex', 'Ey', 'Ez']] *= VCMTOkVCM
                fieldDF['E'] = np.linalg.norm(
                    fieldDF[['Ex', 'Ey', 'Ez']].values, axis=1
                )
                self.fieldStrengths = fieldDF

            # Electric Field Lines
            if 'fieldLineTree' in file:
                lineDF = file['fieldLineTree'].arrays(
                    ['FieldLineID', 'FieldStart', 'x', 'y', 'z'], library='pd'
                )
                lineDF[['x', 'y', 'z']] *= CMTOMICRON
                self.fieldLines = lineDF

            # Particle Tracks (Awkward Array -> Flattened DataFrame)
            if 'animationDataTree' in file:
                anData = file['animationDataTree'].arrays(
                    [
                        'AvalancheID', 'FrameID',
                        'Time', 'ParticleType',
                        'x', 'y', 'z',
                    ],
                    library='ak',
                )
                animationDF = self._flattenBranch(anData)
                animationDF[['x', 'y', 'z']] *= CMTOMICRON
                self.animationData = animationDF

            # Induced Signal Traces
            if 'signalDataTree' in file:
                self.signalData = file['signalDataTree'].arrays(library='pd')

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

        self.ax = self.setupAxes(is3D=True)
        return

    def setupAxes(self, is3D=True, numPlots=3):
            """Clears the figure and prepares 3D or dual 2D subplot axes."""
            self.fig.clear()
            if hasattr(self, 'cbar') and self.cbar is not None:
                self.cbar.remove()
                self.cbar = None
    
            if is3D:
                self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
                return self.ax

            else:
                if numPlots == 3:
                    gs = self.fig.add_gridspec(2, 2)
                    xz = self.fig.add_subplot(gs[0, 0])
                    yz = self.fig.add_subplot(gs[1, 0])
                    xy = self.fig.add_subplot(gs[:, 1])
                    return xz, yz, xy
                
                elif numPlots == 2:
                    xz = self.fig.add_subplot(1, 2, 1)
                    yz = self.fig.add_subplot(1, 2, 2)
                    return xz, yz

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
        self.viewSelector.addItems([  # Match with order in _onChange
            'Simulation Parameters',
            'Geometry',
            'Field Lines',
            'Field Strengths',
            'Electron Avalanche',
            'Induced Signals',
            'Avalanche And Signal',
            'Avalanche Info'
        ])
        self.viewSelector.currentIndexChanged.connect(self._onChange)
        sidebarLayout.addWidget(self.viewSelector)

        # Projection Mode - 2D vs 3D
        projLayout = QHBoxLayout()
        self.chk3D = QCheckBox('3D View')
        self.chk2D = QCheckBox('2D Projections')
        self.chk3D.setChecked(True)

        self.projGroup = QButtonGroup(self)
        self.projGroup.addButton(self.chk3D)
        self.projGroup.addButton(self.chk2D)

        self.chk3D.toggled.connect(self._onChange)

        projLayout.addWidget(self.chk3D)
        projLayout.addWidget(self.chk2D)
        sidebarLayout.addLayout(projLayout)

        # Show Geometry
        self.chkGeometry = QCheckBox('Show Geometry')
        self.chkGeometry.setChecked(True)
        self.chkGeometry.toggled.connect(self._onChange)
        sidebarLayout.addWidget(self.chkGeometry)

        # Global Avalanche ID Selector
        sidebarLayout.addWidget(QLabel('Avalanche ID:'))
        self.avalancheSpinBox = QSpinBox()
        if self.data.animationData is not None and not self.data.animationData.empty:
            maxID = int(self.data.animationData['AvalancheID'].max())
            self.avalancheSpinBox.setRange(0, maxID)
        else:
            self.avalancheSpinBox.setRange(0, 0)
            self.avalancheSpinBox.setEnabled(False)

        self.avalancheSpinBox.valueChanged.connect(self._onChange)
        sidebarLayout.addWidget(self.avalancheSpinBox)

        # Controls Stack for mode-specific options
        self.controlsStack = QStackedWidget()

        # ========================================
        # Params options
        viewWidgetParams = QWidget()
        self.controlsStack.addWidget(viewWidgetParams)

        # Geometry options
        viewWidgetGeometry = QWidget()
        self.controlsStack.addWidget(viewWidgetGeometry)
        
        # Field lines options
        viewWidgetFieldLines = QWidget()
        layoutFieldLines = QVBoxLayout(viewWidgetFieldLines)

        self.chkCathodeLines = QCheckBox('Cathode')
        self.chkCathodeLines.setChecked(True)
        self.chkCathodeLines.toggled.connect(self._onChange)
        self.chkAboveGridLines = QCheckBox('Above Grid')
        self.chkAboveGridLines.setChecked(False)
        self.chkAboveGridLines.toggled.connect(self._onChange)
        self.chkBelowGridLines = QCheckBox('Below Grid')
        self.chkBelowGridLines.setChecked(False)
        self.chkBelowGridLines.toggled.connect(self._onChange)

        layoutFieldLines.addWidget(self.chkCathodeLines)
        layoutFieldLines.addWidget(self.chkAboveGridLines)
        layoutFieldLines.addWidget(self.chkBelowGridLines)
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
        self.fieldGroup.buttonClicked.connect(self._onChange)

        layoutFieldStrengths.addWidget(self.chkEField)
        layoutFieldStrengths.addWidget(self.chkWField)

        layoutVMax = QHBoxLayout()
        lblVMax = QLabel('Max Field:')
        self.txtVMax = QLineEdit()
        self.txtVMax.setPlaceholderText("Auto")
        self.txtVMax.editingFinished.connect(self._onChange)
        layoutVMax.addWidget(lblVMax)
        layoutVMax.addWidget(self.txtVMax)
        layoutFieldStrengths.addLayout(layoutVMax)

        self.chkContours = QCheckBox('Weighting Contours')
        self.chkContours.setChecked(True)
        self.chkContours.toggled.connect(self._onChange)
        self.chkAdjacent = QCheckBox('Adjacent Contour')
        self.chkAdjacent.setChecked(False)
        self.chkAdjacent.toggled.connect(self._onChange)
        layoutFieldStrengths.addWidget(self.chkContours)
        layoutFieldStrengths.addWidget(self.chkAdjacent)

        self.controlsStack.addWidget(viewWidgetFieldStrengths)

        # Avalanche Animation controls
        viewWidgetAvalanche = QWidget()
        layoutViewAvalanche = QVBoxLayout(viewWidgetAvalanche)
        layoutViewAvalanche.setContentsMargins(0, 0, 0, 0)

        # Play / Pause Button
        self.playButton = QPushButton('Play Animation')
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self._toggleAnimation)
        layoutViewAvalanche.addWidget(self.playButton)

        # Step Navigation Buttons
        stepLayout = QHBoxLayout()
        self.prevFrameButton = QPushButton('Prev')
        self.prevFrameButton.clicked.connect(self._prevFrame)
        self.nextFrameButton = QPushButton('Next')
        self.nextFrameButton.clicked.connect(self._nextFrame)
        stepLayout.addWidget(self.prevFrameButton)
        stepLayout.addWidget(self.nextFrameButton)
        layoutViewAvalanche.addLayout(stepLayout)

        # Jump to Time Layout
        jumpLayout = QHBoxLayout()
        lblJump = QLabel('Time (ns):')
        self.timeInput = QLineEdit()
        self.timeInput.setPlaceholderText('1.0')

        self.jumpButton = QPushButton('Jump')
        self.jumpButton.clicked.connect(self._jumpToTime)
        self.timeInput.returnPressed.connect(self._jumpToTime)

        jumpLayout.addWidget(lblJump)
        jumpLayout.addWidget(self.timeInput)
        jumpLayout.addWidget(self.jumpButton)
        layoutViewAvalanche.addLayout(jumpLayout)

        self.chkWeighing = QCheckBox('Weighting Contours')
        self.chkWeighing.setChecked(False)
        self.chkWeighing.toggled.connect(self._onChange)
        layoutViewAvalanche.addWidget(self.chkWeighing)

        self.chkAvSig = QCheckBox('Induced Signal')
        self.chkAvSig.setChecked(False)
        self.chkAvSig.toggled.connect(self._onChange)
        layoutViewAvalanche.addWidget(self.chkAvSig)

        # Add spacer to keep controls pushed to the top
        layoutViewAvalanche.addStretch()

        self.controlsStack.addWidget(viewWidgetAvalanche)

        # Signals controls
        viewWidgetSignal = QWidget()
        layoutSignals = QVBoxLayout(viewWidgetSignal)
        layoutSignals.setContentsMargins(0, 0, 0, 0)

        self.chkLog = QCheckBox('Log Scale')
        self.chkLog.setChecked(True)
        self.chkLog.toggled.connect(self._onChange)

        self.chkSignal = QRadioButton('Induced Signal')
        self.chkCharge = QRadioButton('Total Charge')
        self.chkSignal.setChecked(True)

        self.chkElecSignal = QCheckBox('Show Electron Signal')
        self.chkElecSignal.setChecked(False)
        self.chkElecSignal.toggled.connect(self._onChange)
        self.chkIonSignal = QCheckBox('Show Ion Signal')
        self.chkIonSignal.setChecked(False)
        self.chkIonSignal.toggled.connect(self._onChange)

        self.signalGroup = QButtonGroup(self)
        self.signalGroup.setExclusive(True)
        self.signalGroup.addButton(self.chkSignal)
        self.signalGroup.addButton(self.chkCharge)
        self.signalGroup.buttonClicked.connect(self._onChange)

        layoutSignals.addWidget(self.chkLog)
        layoutSignals.addWidget(self.chkSignal)
        layoutSignals.addWidget(self.chkCharge)
        layoutSignals.addWidget(self.chkElecSignal)
        layoutSignals.addWidget(self.chkIonSignal)

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
    
# **********************************************************************#
    def _onChange(self, *args):
        """Unified handler for view selection, projection, geometry, field, and signal changes."""
        idx = self.viewSelector.currentIndex()
        controlsIdx = 4 if idx == 6 else idx
        self.controlsStack.setCurrentIndex(controlsIdx)

        # Reset animation state on any view change
        self.animationTimer.stop()
        self.playButton.setChecked(False)
        self.playButton.setText("Play Animation")

        # Update sidebar and control button states based on view index
        is3DView = idx in (1, 2, 3, 4, 6)
        self.chk3D.setEnabled(is3DView)
        self.chk2D.setEnabled(is3DView)
        self.chkGeometry.setEnabled(idx in (1, 2, 3, 4, 6))

        # Handle non-plot view (Parameters Table)
        if controlsIdx in [0, 7]:
            self.displayStack.setCurrentIndex(0)
            self.toolbar.hide()
            if controlsIdx == 0:
                self._plotSimParams()
            elif controlsIdx == 7:
                self._plotAvalancheInfo()
            return

        # Handle plot views (Matplotlib Canvas)
        self.displayStack.setCurrentIndex(1)
        self.toolbar.show()

        # View-to-plotting-function mapping
        viewRenderers = {
            1: self._plotGeometry,
            2: self._plotFieldLines,
            3: self._plotFields,
            4: self._resetAvalancheAnimation,
            5: self._plotSignals,
            6: self._resetAvalancheAnimation,
        }

        # Dispatch execution
        renderFunction = viewRenderers.get(idx)
        if renderFunction:
            renderFunction()

        return
    
#**********************************************************************#
    def _reloadData(self):
        self.data.loadRootData()
        self._onChange(self.viewSelector.currentIndex())
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
    def _plotAvalancheInfo(self):
        """Populates the main display table with avalanmche info."""
        self.simParamTable.setRowCount(0)
        
        if self.data.avalancheData is None or self.data.avalancheData.empty:
            return

        df = self.data.avalancheData
        
        self.simParamTable.setRowCount(len(df))
        self.simParamTable.setColumnCount(len(df.columns))
        self.simParamTable.setHorizontalHeaderLabels(list(df.columns))

        for row_idx, row_data in df.iterrows():
            for col_idx, value in enumerate(row_data):
                valStr = f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
                
                item = QTableWidgetItem(valStr)
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                
                self.simParamTable.setItem(row_idx, col_idx, item)

        return
    
#**********************************************************************#
    def _plotGeometry(self):
        use2D = self.chk2D.isChecked()
        if use2D:
            xz, yz, xy = self.canvas.setupAxes(is3D=False)
            self._drawGeometry((xz, yz, xy))
            self._formatAxes((xz, yz, xy))

        else:
            ax = self.canvas.setupAxes()
            self._drawGeometry(ax)
            self._formatAxes(ax)

        self.canvas.fig.tight_layout()
        self.canvas.draw()
        return

#**********************************************************************#
    def _plotFieldLines(self):

        use2D = self.chk2D.isChecked()

        lineSettings = {
            0: {'plot': self.chkCathodeLines.isChecked(), 'c': 'b'}, #Cathode
            1: {'plot': self.chkAboveGridLines.isChecked(), 'c': 'r'}, #Above the grid
            -1: {'plot': self.chkBelowGridLines.isChecked(), 'c': 'g'} #Below the grid
        }

        groupedLines = self.data.fieldLines.groupby(['FieldLineID', 'FieldStart'])

        if use2D:
            xz, yz, xy = self.canvas.setupAxes(is3D=False)

            for (lineID, startVal), lineData in groupedLines:
                setting = lineSettings.get(startVal)
                if not setting or not setting['plot']:
                    continue
                xz.plot(lineData['x'], lineData['z'], c=setting['c'])
                yz.plot(lineData['y'], lineData['z'], c=setting['c'])
                xy.plot(lineData['x'], lineData['y'], c=setting['c'])            

            if self.chkGeometry.isChecked():
                self._drawGeometry((xz, yz, xy))
            self._formatAxes((xz, yz, xy))

        else:
            ax = self.canvas.setupAxes()

            for (lineID, startVal), lineData in groupedLines:
                setting = lineSettings.get(startVal)
                if not setting or not setting['plot']:
                    continue
                ax.plot(
                    lineData['x'], lineData['y'], lineData['z'], 
                    c=setting['c']
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

        plotData = fieldData['E'] if isEField else fieldData['Weight_TopPad']

        if isEField:
            textVal = self.txtVMax.text().strip()
            vmin = np.nanmin(plotData)
            vmax = float(textVal) if textVal else np.nanmax(plotData)

        else:
            vmin, vmax = 0.0, 1.0

        contour = None
        if use2D:
            xz, yz = self.canvas.setupAxes(is3D=False, numPlots=2)
            layout = [
                (xz, 'y', 'x', 'z'),
                (yz, 'x', 'y', 'z')
            ]

            for ax, fixed, x, y in layout:
                mask = np.isclose(fieldData[fixed], 0, atol=1e-6)
                xData, yData = fieldData[x][mask], fieldData[y][mask]
                zData = plotData[mask]

                contour = ax.tricontourf(
                    xData, yData, zData, 
                    levels=101, cmap='viridis', vmin=vmin, vmax=vmax
                )
                if not isEField:
                    if self.chkContours.isChecked():
                        self._plotContours((xz, yz))
                    if self.chkAdjacent.isChecked():
                        self._plotContours((xz, yz), pad='RightBottomPad', color='m')

            if self.chkGeometry.isChecked():
                self._drawGeometry((xz, yz))
            self._formatAxes((xz, yz))

            if contour is not None:
                divider = make_axes_locatable(yz)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                self.cbar = self.canvas.fig.colorbar(contour, cax=cax)
            
        else: 
            ax = self.canvas.setupAxes()
            
            mappable = ax.scatter(
                fieldData['x'], fieldData['y'], fieldData['z'],
                c=plotData, cmap='viridis', s=15, alpha=.75
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
    def _plotContours(self, axes, pad='TopPad', color='c'):
        """
        Plots contour lines with inline labels on 2D spatial axes (xz and yz).
        """
        xz, yz = axes
        
        layout = [
            (xz, 'y', 'x', 'z'),
            (yz, 'x', 'y', 'z')
        ]

        fieldData = self.data.fieldStrengths
        plotData = self.data.fieldStrengths[f'Weight_{pad}']

        for ax, fixed, x, y in layout:
            mask = np.isclose(fieldData[fixed], 0, atol=1e-6)
            xData = fieldData[x][mask]
            yData = fieldData[y][mask]
            zData = plotData[mask]

            sep=0.05
            lines = ax.tricontour(
                xData, yData, zData,  
                levels=np.arange(0, 1+sep, sep), colors=color, linewidths=0.5, 
                vmin=0, vmax=1
            )
            ax.clabel(lines, inline=True, fontsize=8, fmt='%.2f')

        return
    
#**********************************************************************#
    def _plotSignals(self): 
        #TODO: All signals or just individual pads

        isSignal = self.chkSignal.isChecked()
        logScale = self.chkLog.isChecked()
        pltElectrons = self.chkElecSignal.isChecked()
        pltIons = self.chkIonSignal.isChecked()

        ax = self.canvas.setupAxes(is3D=False, numPlots=1)

        avID = self.avalancheSpinBox.value()
        self.signalData = self.data.signalData[self.data.signalData['AvalancheID'] == avID]
        inSignalData = self.signalData

        time = inSignalData['Time']

        padList = list(
            dict.fromkeys(
                col.split("_")[1]
                for col in inSignalData.columns
                if col not in ["AvalancheID", "Time"]
            )
        )

        styles = {
            'Signal': '-',
            'Electron': '--',
            'Ion': ':'
        }

        colors = plt.cm.tab10.colors
        padColor = {
            pad: colors[i % len(colors)] for i, pad in enumerate(padList)
        }

        # Plot all dynamic signal branches
        for col in inSignalData.columns:
            if col in ['AvalancheID', 'Time']:
                continue

            prefix, pad = col.split("_")

            if prefix == 'Electron' and not pltElectrons:
                continue
            if prefix == 'Ion' and not pltIons:
                continue

            signal = inSignalData[col] if isSignal else inSignalData[col].cumsum()
            ax.plot(
                time+0.001, signal,
                ls=styles.get(prefix, "-"), c=padColor[pad],
                label=pad if prefix == 'Signal' else None
            )

        if logScale:
            ax.set_xscale('log')

        ax.set_xlim([.1, None])

        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Signal (fC/ns)' if isSignal else 'Charge (fC)')
        ax.grid()
        ax.legend()

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        return

#**********************************************************************#
    def _resetAvalancheAnimation(self):
        # Reset animation UI & Timer
        self.animationTimer.stop()
        self.playButton.setChecked(False)
        self.playButton.setText('Play Animation')

        # Reset common state variables
        avID = self.avalancheSpinBox.value()
        self.inAvData = self.data.animationData[self.data.animationData['AvalancheID'] == avID]
        self.signalData = self.data.signalData[self.data.signalData['AvalancheID'] == avID]
        
        self.allFrames = sorted(self.inAvData['FrameID'].unique())
        self.curFrameID = 0

        # Route to the active view renderer
        self._renderAnimation()

        return

#**********************************************************************#
    def _renderAnimation(self):
        idx = self.viewSelector.currentIndex()
        if idx == 6:
            self._plotAvalancheSignal()
        else:
            self._renderParticleFrame()

#**********************************************************************#
    def _nextFrame(self):
        if not self.allFrames:
            return
        self.curFrameID = (self.curFrameID + 1) % len(self.allFrames)
        self._renderAnimation()

        return

#**********************************************************************#
    def _prevFrame(self):
        if not self.allFrames:
            return
        self.curFrameID = (self.curFrameID - 1) % len(self.allFrames)
        self._renderAnimation()

        return

# **********************************************************************#
    def _jumpToTime(self):
        """Finds the frame closest to the target time entered."""
        if not self.allFrames or self.inAvData.empty:
            return

        text = self.timeInput.text().strip()
        if not text:
            return

        try:
            targetTime = float(text)
        except ValueError:
            return


        # Calculate absolute difference to find the nearest frame index
        frameTimes = (
            self.inAvData.groupby('FrameID')['Time'].first().reindex(self.allFrames)
        )
        closestFrameID = (frameTimes - targetTime).abs().idxmin()
        self.curFrameID = self.allFrames.index(closestFrameID)

        # Render updated frame
        self._renderAnimation()
        return

#**********************************************************************#
    def _toggleAnimation(self, checked):
        if checked:
            self.playButton.setText('Pause')
            self.animationTimer.start(80)  # 80=~12 FPS frame rate ()lower is faster
        else:
            self.playButton.setText('Play Animation')
            self.animationTimer.stop()

        return

#**********************************************************************#
    def _renderParticleFrame(self):
        if not self.allFrames:
            return

        frameID = self.allFrames[self.curFrameID]
        inFrameData = self.inAvData[self.inAvData['FrameID'] == frameID]

        particleConfig = [
            {'ID': 0, 'c': 'b', 's': 10, 'label': 'Electrons'},
            {'ID': 1, 'c': 'r', 's': 15, 'label': 'Positive Ions'},
            {'ID': -1, 'c': 'g', 's': 15, 'label': 'Negative Ions'},
        ]
        
        if self.chk2D.isChecked():
            xz, yz, xy = self.canvas.setupAxes(is3D=False)
            allAxs = (xz, yz, xy)

            plots = [
                (xz, ('x', 'z')),
                (yz, ('y', 'z')),
                (xy, ('x', 'y')),
            ]
            labelAx = xz
                
            if self.chkWeighing.isChecked():
                self._plotContours((xz, yz))

        else:
            ax = self.canvas.setupAxes()
            allAxs = ax
            plots = [(ax, ('x', 'y', 'z'))]
            labelAx = ax

        for inAx, cols in plots:
            for p in particleConfig:
                subData = inFrameData[inFrameData['ParticleType'] == p['ID']]
                coords = [subData[col] for col in cols]
                inAx.scatter(*coords, c=p['c'], s=p['s'], label=p['label'])

        if self.chkGeometry.isChecked():
            self._drawGeometry(allAxs)
        self._formatAxes(allAxs)

        inTime = inFrameData['Time'].iloc[0] if not inFrameData.empty else -1
        timeLabel = f'{inTime:.2f} ns' if inTime <= 250 else rf'{inTime/1e3:.2f} $\mu$s'
        labelAx.set_title(f'Time = ({timeLabel}) (Frame ID: {frameID})')
        labelAx.legend()

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        return

#**********************************************************************#
    def _plotAvalancheSignal(self):
        xz, yz, sig = self.canvas.setupAxes(is3D=False)

        frameID = self.allFrames[self.curFrameID]
        inFrameData = self.inAvData[self.inAvData['FrameID'] == frameID]
        inSignalData = self.signalData

        particleConfig = [
            {'ID':  0, 'c': 'b', 's': 10, 'label': 'Electrons'},
            {'ID':  1, 'c': 'r', 's': 15, 'label': 'Positive Ions'},
            {'ID': -1, 'c': 'g', 's': 15, 'label': 'Negative Ions'},
        ]

        if self.chkWeighing.isChecked():
            self._plotContours((xz, yz))

        for ax, x, y in [(xz, 'x', 'z'), (yz, 'y', 'z')]:
            for p in particleConfig:
                subData = inFrameData[inFrameData['ParticleType'] == p['ID']]
                label = p['label'] if ax == xz else None
                ax.scatter(subData[x], subData[y], c=p['c'], s=p['s'], label=label)

        if self.chkGeometry.isChecked():
            self._drawGeometry((xz, yz))
        self._formatAxes((xz, yz))

        padList = list({col.split('_')[1] for col in inSignalData.columns if col not in ['AvalancheID', 'Time']})
        colors = plt.cm.tab10.colors
        padColor = {pad: colors[i % len(colors)] for i, pad in enumerate(padList)}

        styles = {'Signal': '-', 'Electron': '--', 'Ion': ':'}
        isSignal = self.chkAvSig.isChecked()
        time = inSignalData['Time']

        activePrefixes = {'Signal'}
        for col in inSignalData.columns:
            if col in ['AvalancheID', 'Time']:
                continue

            prefix, pad = col.split('_')
            if prefix not in activePrefixes:
                continue

            signal = inSignalData[col] if isSignal else inSignalData[col].cumsum()
            sig.plot(
                time+0.001, signal,
                ls=styles.get(prefix, '-'), c=padColor[pad],
                label=pad if prefix == 'Signal' else None
            )

        inTime = inFrameData['Time'].iloc[0] if not inFrameData.empty else -1
        
        sig.axvline(inTime, c='r', ls='--')

        sig.set_xscale('log')
        sig.set_xlim([.1, None])

        sig.set_xlabel('Time (ns)')
        sig.set_ylabel('Signal (fC/ns)' if isSignal else 'Charge (fC)')
        sig.grid()
        sig.legend()

        timeLabel = f'{inTime:.2f} ns' if inTime <= 250 else rf'{inTime/1e3:.2f} $\mu$s'
        xz.set_title(f'Time = ({timeLabel}) (Frame ID: {frameID})')
        xz.legend()

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        return

#----- Formatting figures -----
#**********************************************************************#
    def _drawGeometry(self, axes):
        """
        TODO
        """
        if self.data.simData is None:
            return
        
        padLength = self.data.simData['padLength']
        
        self._addGrid(axes)
        self._addPads(axes)
        self._addUnitCell(axes)

        return

#**********************************************************************#
    def _addGrid(self, axes):
        pitch = self.data.simData['pitch']
        holeRadius = self.data.simData['holeRadius']
        gridThickness = self.data.simData['gridThickness']
        
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
            xz, yz, *rest = axes

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
            if rest:
                rest[0].pcolormesh(
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
        pitch = self.data.simData['pitch']
        padLength = self.data.simData['padLength']

        #padThickness = self.data.simData['padThickness']
        amplificationGap = self.data.simData['amplificationGap']

        zHeight = -amplificationGap

        sqrt3 = math.sqrt(3)
        centers = pitch * np.array([
            (0,0),
            (0, 1), (0, -1),
            (sqrt3/2, .5), (sqrt3/2, -.5),
            (-sqrt3/2, .5), (-sqrt3/2, -.5),
        ])

        if isinstance(axes, tuple):
            xz, yz, *rest = axes

            for (i, j) in centers:
                xLocs, yLocs = hexXY(padLength, i, j)
                line = '-' if i==0 and j==0 else ':'
                xz.plot(xLocs, zHeight*np.ones(len(xLocs)), c='m', ls=line)
                yz.plot(yLocs, zHeight*np.ones(len(yLocs)), c='m', ls=line)

                if rest:
                    rest[0].plot(xLocs, yLocs, c='m', ls=line)

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
        pitch = self.data.simData['pitch']

        sqrt3 = math.sqrt(3)
        xLocs, yLocs = hexXY(pitch/sqrt3, 0, 0)

        amplificationGap = self.data.simData['amplificationGap']
        driftLength = self.data.simData['driftLength']
        padHeight = -amplificationGap#Not quite but okay

        if isinstance(axes, tuple):
            xz, yz, *rest = axes
            xz.axvline(xLocs[0], c='c', ls='--')
            xz.axvline(-xLocs[0], c='c', ls='--')
            xz.axvline(xLocs[1], c='c', ls=':')
            xz.axvline(-xLocs[1], c='c', ls=':')
            yz.axvline(pitch/2, c='c', ls='--')
            yz.axvline(-pitch/2, c='c', ls='--')
            yz.axvline(0, c='c', ls=':')
            if rest:
                rest[0].plot(xLocs, yLocs, c='c')

        else:
            ax = axes
            ax.plot(xLocs, yLocs, 0*np.ones(len(xLocs)), c='c')
            ax.plot(xLocs, yLocs, padHeight*np.ones(len(xLocs)), c='c')
            ax.plot(xLocs, yLocs, driftLength*np.ones(len(xLocs)), c='c')
            for x, y in zip(xLocs, yLocs):
                xLine = [x, x]
                yLine = [y, y]
                zLine = [driftLength, padHeight]
                ax.plot(xLine, yLine, zLine, c='c', ls='--')

        return

#**********************************************************************#
    def _formatAxes(self, axes):
        """
        TODO
        """
        xLabel = r'x ($\mu$m)'
        yLabel = r'y ($\mu$m)'
        zLabel = r'z ($\mu$m)'

        pitch = self.data.simData['pitch']
        xScale = pitch*math.sqrt(3)/2
        yScale = pitch

        amplificationGap = self.data.simData['amplificationGap']
        driftLength = self.data.simData['driftLength']
        zBuffer=5
        zLim = [-amplificationGap-zBuffer, driftLength+zBuffer]
        
        if isinstance(axes, tuple):
            xz, yz, *rest = axes

            xz.set(
                xlabel=xLabel, ylabel=zLabel, 
                xlim=[-xScale, xScale], ylim=zLim
            )
            yz.set(
                xlabel=yLabel, ylabel=zLabel, 
                xlim=[-yScale, yScale], ylim=zLim
            )
            if rest: #xy
                rest[0].set(
                    xlabel=xLabel,
                    ylabel=yLabel,
                    xlim=[-xScale, xScale],
                    ylim=[-yScale, yScale],
                    aspect='equal',
                )

            for ax in axes:
                ax.axvline(0, c='k', ls=':', alpha=.75)
                ax.axhline(0, c='k', ls=':', alpha=.75)
                ax.grid(alpha=.25, ls=':')

        else:
            ax = axes

            ax.set(
                xlabel=xLabel, ylabel=yLabel, zlabel=zLabel, 
                xlim=[-xScale, xScale], ylim=[-yScale, yScale], zlim=zLim
            )
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
