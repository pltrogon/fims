import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from pandasgui import show
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from runDataClass import runData


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulation Data GUI")
        self.geometry("2000x1000")

        self.simData = None
        self.currentPlot = None
        self.currentFig = None
        self.binWidthValue = 1
        self.thresholdValue = 0
        self.curAvalancheID = tk.IntVar(value=0)
        self.curParticle = tk.StringVar(self)
        self.curParticle.set('-- Select --')

        # --- Frames ---
        self.runFrame = tk.Frame(self)
        self.runFrame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.mainFrame = tk.Frame(self)
        self.mainFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.optionsFrame = tk.Frame(self.mainFrame)
        self.optionsFrame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.listFrame = tk.Frame(self.optionsFrame)
        self.listFrame.pack(side=tk.TOP, fill=tk.Y, padx=10, pady=10)

        self.binFrame = tk.Frame(self.optionsFrame)
        self.binFrame.pack(side=tk.TOP, fill=tk.Y, padx=5, pady=5)

        self.threshFrame = tk.Frame(self.optionsFrame)
        self.threshFrame.pack(side=tk.TOP, fill=tk.Y, padx=5, pady=5)

        self.avalancheFrame = tk.Frame(self.optionsFrame)
        self.avalancheFrame.pack(side=tk.TOP, fill=tk.Y, padx=5, pady=5)

        self.particleFrame = tk.Frame(self.optionsFrame)
        self.particleFrame.pack(side=tk.TOP, fill=tk.Y, padx=5, pady=5)

        self.buttonFrame = tk.Frame(self.optionsFrame)
        self.buttonFrame.pack(side=tk.TOP, fill=tk.Y, padx=10)

        self.metaDataFrame = tk.Frame(self.mainFrame)
        self.metaDataFrame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.plottingFrame = tk.Frame(self.mainFrame)
        self.plottingFrame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # --- Run number entry ---
        runLabel = tk.Label(
            self.runFrame, 
            text="Run Number:", 
            font=("Arial", 20, "bold"))
        runLabel.pack(side=tk.LEFT, padx=5)

        self.runEntry = tk.Entry(self.runFrame, width=10)
        self.runEntry.pack(side=tk.LEFT, padx=5)
        self.runEntry.insert(0, "1000")

        self.loadRun = tk.Button(
            self.runFrame, 
            text="Load", 
            command=self.loadData)
        self.loadRun.pack(side=tk.LEFT)
        self.runEntry.bind("<Return>", self.loadData)


        # --- Treeview for metaData ---
        metaDataLabel = tk.Label(
            self.metaDataFrame, 
            text="MetaData:", 
            font=("Arial", 20, "bold"))
        metaDataLabel.pack(side=tk.TOP, pady=5)

        self.metaDataTree = ttk.Treeview(self.metaDataFrame, columns=('Value'), show='tree headings')
        self.metaDataTree.heading('#0', text='Parameter')
        self.metaDataTree.heading('Value', text='Value')
        self.metaDataTree.column('#0', width=150)
        self.metaDataTree.column('Value', width=200)

        self.metaDataTree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Listbox for DataFrames ---
        dataLabel = tk.Label(
            self.listFrame, 
            text='DataFrames:', 
            font=("Arial", 20, "bold"))
        dataLabel.pack(side=tk.TOP, pady=(0, 5))
        
        self.dataList = tk.Listbox(self.listFrame, selectmode=tk.SINGLE)
        self.dataList.pack(side=tk.TOP, fill=tk.Y, expand=True)
        self.dataList.bind('<<ListboxSelect>>', self.selectData)

        # --- Bin width ---
        binLabel = tk.Label(self.binFrame, text='Histogram Bin Width:')
        binLabel.pack(side=tk.LEFT, pady=(2, 2))

        self.binWidth = tk.Spinbox(
            self.binFrame, 
            from_=1, to=100, 
            wrap=False, width=2, 
            command=self.updateBinWidth)
        self.binWidth.pack(side=tk.LEFT)
        self.binWidth.delete(0, "end")
        self.binWidth.insert(0, 1)    

        # --- Threshold ---
        threshLabel = tk.Label(self.threshFrame, text='Threshold:')
        threshLabel.pack(side=tk.LEFT, pady=(2, 2))

        self.threshold = tk.Spinbox(
            self.threshFrame, 
            from_=0, to=1000, 
            wrap=False, width=3, 
            command=self.updateThreshold)
        self.threshold.pack(side=tk.LEFT)
        self.threshold.delete(0, "end")
        self.threshold.insert(0, 0) 

        # --- Avalanche number ---
        avalancheLabel = tk.Label(self.avalancheFrame, text='Avalanche ID:')
        avalancheLabel.pack(side=tk.LEFT, pady=(2, 2))

        self.avalancheID = tk.Spinbox(
            self.avalancheFrame, 
            from_=0, to=999, ##TODO - make this maximum match the value in metadata. Or alternative just be very large
            wrap=False, width=5, 
            textvariable=self.curAvalancheID,
            command=self.updateAvalancheID)
        self.avalancheID.pack(side=tk.LEFT)
        self.avalancheID.bind("<Return>", self.updateAvalancheID)
        
        # --- Particle Options ---
        particleLabel = tk.Label(self.particleFrame, text='Particle Choice:')
        particleLabel.pack(side=tk.LEFT, pady=(2, 2))

        particleOptions = ['electron', 'posIon', 'negIon']
        self.particleSelection = tk.OptionMenu(
            self.particleFrame, 
            self.curParticle, 
            *particleOptions)
        self.particleSelection.pack(side=tk.RIGHT, pady=5, padx=5)
        self.curParticle.trace_add('write', self.updateParticle)

        # --- Plotting Buttons ---
        plotButtons = tk.Label(
            self.buttonFrame, 
            text='Plots:', 
            font=("Arial", 20, "bold"))
        plotButtons.pack(side=tk.TOP, padx=5, pady=5)

        self.showGeometry = tk.Button(
            self.buttonFrame, text='Geometry', 
            command=lambda: self.plotButton('Geometry')
            )
        self.showGeometry.pack(side=tk.TOP)

        self.fieldLineFrame = tk.Frame(self.buttonFrame)
        self.fieldLineFrame.pack(side=tk.TOP, fill=tk.Y)

        self.showFieldLines = tk.Button(
            self.fieldLineFrame, text='All Field Lines', 
            command=lambda: self.plotButton('FieldLines')
            )
        self.showFieldLines.pack(side=tk.TOP)

        self.cathodeLineFrame = tk.Frame(self.buttonFrame)
        self.cathodeLineFrame.pack(side=tk.TOP, fill=tk.Y)

        self.showCathodeLines = tk.Button(
            self.cathodeLineFrame, text='Cathode', 
            command=lambda: self.plotButton('Cathode')
            )
        self.showCathodeLines.pack(side=tk.LEFT)

        self.showEdgeLines = tk.Button(
            self.cathodeLineFrame, text='Edge', 
            command=lambda: self.plotButton('Edge')
            )
        self.showEdgeLines.pack(side=tk.RIGHT)

        self.gridLineFrame = tk.Frame(self.buttonFrame)
        self.gridLineFrame.pack(side=tk.TOP, fill=tk.Y)

        self.showAboveGrid = tk.Button(
            self.fieldLineFrame, text='AboveGrid', 
            command=lambda: self.plotButton('AboveGrid')
            )
        self.showAboveGrid.pack(side=tk.LEFT)

        self.showBelowGrid = tk.Button(
            self.fieldLineFrame, text='BelowGrid', 
            command=lambda: self.plotButton('BelowGrid')
            )
        self.showBelowGrid.pack(side=tk.LEFT)

        self.avalancheFrame = tk.Frame(self.buttonFrame)
        self.avalancheFrame.pack(side=tk.TOP, fill=tk.Y)

        self.showAvalanche = tk.Button(
            self.avalancheFrame, text='Avalanche Size', 
            command=lambda: self.plotButton('AvalancheSize')
            )
        self.showAvalanche.pack(side=tk.LEFT)

        self.showEfficiency = tk.Button(
            self.avalancheFrame, text='Efficiency', 
            command=lambda: self.plotButton('Efficiency')
            )
        self.showEfficiency.pack(side=tk.LEFT)

        self.showAvalancheTrack = tk.Button(
            self.buttonFrame, text='Avalanche Tracks', 
            command=lambda: self.plotButton('AvalancheTracks')
            )
        self.showAvalancheTrack.pack(side=tk.TOP)

        self.showHeatmap = tk.Button(
            self.buttonFrame, text='Particle Heatmap', 
            command=lambda: self.plotButton('Heatmap')
            )
        self.showHeatmap.pack(side=tk.TOP)

        self.showDiffusion = tk.Button(
            self.buttonFrame, text='Particle Diffusion', 
            command=lambda: self.plotButton('Diffusion')
            )
        self.showDiffusion.pack(side=tk.TOP)

        self.signalFrame = tk.Frame(self.buttonFrame)
        self.signalFrame.pack(side=tk.TOP, fill=tk.Y)

        self.showInducedSignal = tk.Button(
            self.signalFrame, text='Induced Signal', 
            command=lambda: self.plotButton('InducedSignal')
            )
        self.showInducedSignal.pack(side=tk.LEFT)

        self.showAverageSignal = tk.Button(
            self.signalFrame, text='Average', 
            command=lambda: self.plotButton('AverageSignal')
            )
        self.showAverageSignal.pack(side=tk.RIGHT)

        self.showSignalvsGain = tk.Button(
            self.buttonFrame, text='Signal vs. Gain', 
            command=lambda: self.plotButton('SignalvsGain')
            )
        self.showSignalvsGain.pack(side=tk.TOP)


    def loadData(self, *args):
        """
        Loads data and updates the GUI.
        """
        inRunNo = self.runEntry.get()
        if not inRunNo.isdigit():
            messagebox.showerror("Invalid Input", "Please enter a valid number.")
            return

        runNo = int(inRunNo)
        
        try:
            self.simData = runData(runNo)

            if self.simData:
                if self.currentPlot is None:
                    self.currentPlot = 'Geometry'
                self.getPlot()

            self.updateGUI()

            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load data: {e}")


    def updateGUI(self):
        """
        Updates the the listbox and metadata with the loaded data.
        """
        
        # Clear existing widgets
        for item in self.metaDataTree.get_children():
            self.metaDataTree.delete(item)
        self.dataList.delete(0, tk.END)
        for widget in self.plottingFrame.winfo_children():
            widget.destroy()

        # Populate listbox with dataframe names - skip metadata
        if self.simData and hasattr(self.simData, 'dataTrees'):
            for name in self.simData.dataTrees:
                if name != 'metaData':
                    self.dataList.insert(tk.END, name)

        # Populate the Treeview with metaData values
        if self.simData and hasattr(self.simData, 'getMetaData'):
            metaDataDict = self.simData.getMetaData()
            for key, value in metaDataDict.items():
                self.metaDataTree.insert('', tk.END, text=key, values=(value,))

        if self.simData and self.currentFig:
            canvas = FigureCanvasTkAgg(self.currentFig, master=self.plottingFrame)
            canvasWidget = canvas.get_tk_widget()
            canvasWidget.pack(fill=tk.BOTH, expand=True)
            canvas.draw()


    def getPlot(self):
        """
        Matches the currentPlot with the appropriate method. Gets the figure.
        """
        if self.simData:
            match self.currentPlot:
                case 'Geometry':
                    self.currentFig = self.simData.plotCellGeometry()

                case 'AvalancheSize':
                    self.currentFig = self.simData.plotAvalancheFits(binWidth=self.binWidthValue)
                
                case 'Efficiency':
                    self.currentFig = self.simData.plotEfficiency(binWidth=self.binWidthValue, threshold=self.thresholdValue)

                case 'FieldLines':
                    self.currentFig = self.simData.plotAllFieldLines()

                case 'Cathode':
                    self.currentFig = self.simData.plot2DFieldLines('Cathode')
                case 'Edge':
                    self.currentFig = self.simData.plot2DFieldLines('Edge')
                
                case 'AboveGrid':
                    self.currentFig = self.simData.plot2DFieldLines('AboveGrid')
                case 'BelowGrid':
                    self.currentFig = self.simData.plot2DFieldLines('BelowGrid')

                case 'AvalancheTracks':
                    self.currentFig = self.simData.plotAvalanche2D(avalancheID=self.curAvalancheID.get())

                case 'Heatmap':
                    self.currentFig = self.simData.plotParticleHeatmaps(self.curParticle.get())

                case 'Diffusion':
                    self.currentFig = self.simData.plotDiffusion(self.curParticle.get())

                case 'InducedSignal':
                    self.currentFig = self.simData.plotAvalancheSignal(avalancheID=self.curAvalancheID.get())

                case 'AverageSignal':
                    self.currentFig = self.simData.plotAverageSignal()

                case 'SignalvsGain':
                    self.currentFig = self.simData.plotSignalvsGain()

                case _:
                    print('Incorrect plot - defaulting')
                    self.currentFig = self.simData.plotCellGeometry()


            
    def selectData(self, event):
        """
        Opens the selected dataframe in pandasgui.
        """
        # Get the selected index from the listbox
        selectIndex = self.dataList.curselection()
        if not selectIndex:
            return
        
        index = selectIndex[0]
        # Get the item text from the listbox
        selectedName = self.dataList.get(index)

        dataFrame = None
        if self.simData and hasattr(self.simData, 'getDataFrame'):
            dataFrame = self.simData.getDataFrame(selectedName)
        
        if dataFrame is not None:
            show(dataFrame)
        else:
            messagebox.showerror("Error", f"DataFrame '{selectedName}' not found in simData.")

    def updateBinWidth(self):
        """
        Saves the spinbox value to and updates the plot if data is loaded.
        """
        try:
            # Get the value from the spinbox
            value = int(self.binWidth.get())
            
            if value >= 1:
                self.binWidthValue = value
            else:
                self.binWidth.delete(0, "end")
                self.binWidth.insert(0, 1)
                self.binWidthValue = 1

            if self.simData and self.currentPlot:
                self.getPlot()

            self.updateGUI()
       
        except ValueError:
            messagebox.showerror("Invalid Input", "Bin width must be an integer.")
            self.binWidth.delete(0, "end")
            self.binWidth.insert(0, self.binWidthValue)

    def updateThreshold(self):
        """
        Saves the spinbox value to and updates the plot if data is loaded.
        """
        try:
            # Get the value from the spinbox
            value = int(self.threshold.get())
            
            if value >= 1:
                self.thresholdValue = value
            else:
                self.threshold.delete(0, "end")
                self.threshold.insert(0, 1)
                self.thresholdValue = 0

            if self.simData and self.currentPlot:
                self.getPlot()

            self.updateGUI()
       
        except ValueError:
            messagebox.showerror("Invalid Input", "Threshold must be an integer.")
            self.threshold.delete(0, "end")
            self.threshold.insert(0, self.thresholdValue)

    
    def updateAvalancheID(self, *args):
        """
        Updates the Avalanche ID based on the Spinbox value.
        """
        try:
            value = self.curAvalancheID.get()
            
            if value < 0:
                self.curAvalancheID.set(0)

            if self.simData and self.currentPlot:
                self.getPlot()

            self.updateGUI()
    
        except tk.TclError:
            messagebox.showerror("Invalid Input", "Avalanche ID must be an integer.")
            self.curAvalancheID.set(0)

    def updateParticle(self, *args):
        """
        """
        self.getPlot()
        self.updateGUI()


    def plotButton(self, plotToGet):
        self.currentPlot = plotToGet
        self.getPlot()
        self.updateGUI()

if __name__ == "__main__":
    app = App()
    app.mainloop()