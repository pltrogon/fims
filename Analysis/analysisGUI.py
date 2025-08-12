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
        self.geometry("1400x600")

        self.simData = None

        # --- Frames ---
        self.runFrame = tk.Frame(self)
        self.runFrame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.mainFrame = tk.Frame(self)
        self.mainFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.listFrame = tk.Frame(self.mainFrame)
        self.listFrame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.metaDataFrame = tk.Frame(self.mainFrame)
        self.metaDataFrame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.plotFrame = tk.Frame(self.mainFrame)
        self.plotFrame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # --- Run number entry ---
        runLabel = tk.Label(self.runFrame, text="Run Number:")
        runLabel.pack(side=tk.LEFT, padx=5)

        self.runEntry = tk.Entry(self.runFrame)
        self.runEntry.pack(side=tk.LEFT, padx=5)
        self.runEntry.insert(0, "1000")

        self.loadRun = tk.Button(self.runFrame, text="Load", command=self.loadData)
        self.loadRun.pack(side=tk.LEFT)


        # --- Treeview for metaData ---
        metaDataLabel = tk.Label(self.metaDataFrame, text="MetaData:", font=("Arial", 12, "bold"))
        metaDataLabel.pack(side=tk.TOP, pady=5)

        # Create a Treeview widget with two columns
        self.metaDataTree = ttk.Treeview(self.metaDataFrame, columns=('Value'), show='tree headings')
        self.metaDataTree.heading('#0', text='Parameter')
        self.metaDataTree.heading('Value', text='Value')
        self.metaDataTree.column('#0', width=150)
        self.metaDataTree.column('Value', width=200)

        # Pack the Treeview and its scrollbar
        self.metaDataTree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Listbox for DataFrames ---
        dataLabel = tk.Label(self.listFrame, text="Available DataFrames", font=("Arial", 12, "bold"))
        dataLabel.pack(side=tk.TOP, pady=(0, 5))
        
        self.dataList = tk.Listbox(self.listFrame, selectmode=tk.SINGLE)
        self.dataList.pack(side=tk.TOP, fill=tk.Y, expand=True)
        self.dataList.bind('<<ListboxSelect>>', self.selectData)

    def loadData(self):
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
            # Call a separate method to handle GUI updates
            self.updateGUI()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load data: {e}")

    def updateGUI(self):
        """
        Updates the plot and the listbox with the loaded data.
        """
        # Clear existing widgets
        for widget in self.plotFrame.winfo_children():
            widget.destroy()
        for item in self.metaDataTree.get_children():
            self.metaDataTree.delete(item)
        self.dataList.delete(0, tk.END)


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

        # Display the geometry plot
        if self.simData and hasattr(self.simData, 'plotCellGeometry'):
            fig = self.simData.plotCellGeometry()
            canvas = FigureCanvasTkAgg(fig, master=self.plotFrame)
            canvasWidget = canvas.get_tk_widget()
            canvasWidget.pack(fill=tk.BOTH, expand=True)
            canvas.draw()
            
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

if __name__ == "__main__":
    app = App()
    app.mainloop()