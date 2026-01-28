############################################
# CLASS DEFINITION FOR GAS SIMULATION DATA #
############################################
from __future__ import annotations

import os
import uproot
import pandas as pd
import awkward_pandas
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import glob

from collections import defaultdict

from scipy.stats import shapiro

#********************************************************************************#
#********************************************************************************#
#********************************************************************************#
#********************************************************************************#
#********************************************************************************#
class gasData:
    """
    Class representing simulation data from scanning through a specified gas.

    Handles data from parallel plate simulations where one gas 
    component is scanned while another remains constant.

    Attributes:
        gasID (str): Identifier for the specific simulation run.
        scannedGas (str): The name of the gas being varied.
        constGasFraction (int): Percentage of the non-scanned gas component.
        gasEfficiencyData (pd.DataFrame): Data filtered for 95% efficiency criteria.
        fields (list[int]): Sorted list of available field strengths (kV/cm).
        gasFieldData (dict): Dictionary mapping field strengths to DataFrames 
            containing scan results.
    """


#********************************************************************************#
    # Initialize
    def __init__(self):
        """
        Initializes a scanned gas data object.
        """

        self.gasID = None
        self.scannedGas = ""
        self.constGasFraction = None

        self._gasComposition = ""

        self.gasEfficiencyData = None
        self.fields = None
        self.gasFieldData = None
        


#********************************************************************************#
    def loadGasScans(self, gasID=None, scannedGas=None, constGasFraction=None):
        """        
        Loads all simulated data from scanning a given gas and given constant gas. 

        Assumes simulations are available at both a constant field strength
        and with fields necessary for 95% effieincy.

        Args:
            gasID (str): Identifier of the gas mixture.
            scannedGas (str): Name of the scanned gas.
            constGasFraction (int): Amount for the constant, non-scanned gas.
        """

        if gasID is None or scannedGas is None or constGasFraction is None:
            raise ValueError("Error - No inputs.")
        
        self.gasID = gasID
        self.scannedGas = scannedGas
        self.constGasFraction = constGasFraction


        self._checkGas()
        self._gasComposition = self._getGasString()

        #Get results from efficiency scans
        self.gasEfficiencyData = self._readEfficiencyScans()

        #Get results from all constant-field scans avalable
        self.fields = self._getFieldStrengths()
        self.gasFieldData = self._readGasScans()

        return

#********************************************************************************#
    def _checkGas(self):
        """
        Checks if the gas scan string is valid.
        """

        gasScans = [
                'isobutane',
                'CF4'
            ]
        
        if self.scannedGas not in gasScans:
            raise ValueError('Error - Invalid gas scan.')
        
        return
        

#********************************************************************************#
    def _getGasString(self):
        """
        Gets the gas composition string based on the scanned gas.

        Returns:
            str: Gas composition identifier string.
        """

        self._checkGas()

        match self.scannedGas:
            case 'isobutane':
                scanGas = '.scanIsobutane'
                constantGas = f'.with{self.constGasFraction}CF4'
            case 'CF4':
                scanGas = '.scanCF4'
                constantGas = f'.with{self.constGasFraction}isobutane'
            case _:
                raise ValueError('Error - Invalid gas composition')

        return f'{self.gasID}{scanGas}{constantGas}'


#********************************************************************************#
    def _readEfficiencyScans(self):
        """
        Reads all efficiency scan data from a .pkl file.

        Returns:
            pd.Dataframe: Datframe witht the efficiency scan results.
        """

        filePath = '../Data/ParallelPlate'
        fileName = f'{self._gasComposition}.Efficiency.pkl'
        filename = os.path.join(filePath, fileName)

        inData = pd.read_pickle(filename)
        gasData = self._calcVariability(inData)

        return gasData
    
#********************************************************************************#
    def _calcVariability(self, gasData):
        """
        Calculate the gain variability, defined as 1 - n_95 / nbar.
        Appends these results to the input dataframe.

        Args:
            gasData (pd.dataframe): Dataframe containing the gas-scan data

        Returns:
            pd.dataframe: Dataframe containing the updated gas-scan data
        """

        gasData['variability'] = 1 - gasData['n_95']/gasData['meanGain']
        A = gasData['gainErr']/gasData['meanGain']
        B = gasData['n_95Err']/gasData['n_95']
        gasData['variabilityErr'] = np.abs(gasData['n_95']/gasData['meanGain'])*np.sqrt(A**2 + B**2)

        return gasData


#********************************************************************************#
    def _getFieldStrengths(self):
        """
        Gets all field strengths with scans associated with the given gas.

        Returns:
            list: List of field strengths.
        """

        try:
            filePath = '../Data/ParallelPlate'
            allFiles = os.listdir(filePath)

        except FileNotFoundError:
            print(f'Error: Directory not found at {filePath}')
            return []
        

        fieldStrengths = set()
        prefix = f'{self._gasComposition}.at'
        suffix = 'kVcm.pkl'

        for inFile in allFiles:
            if inFile.startswith(prefix) and inFile.endswith(suffix):
                try:
                    fieldString = inFile.replace(prefix, '', 1) 
                    fieldString = fieldString.replace(suffix, '', 1)
                
                    fieldInt = int(fieldString)
                    fieldStrengths.add(fieldInt)
                
                except ValueError:
                    print(f"Warning: Could not parse strength from file: {inFile}")
                    continue

        fields = sorted(list(fieldStrengths))
        print(f'Field strengths are: {fields}')
        
        return fields

#********************************************************************************#
    def _readGasScans(self):
        """
        Reads all scan data from avaialble .pkl files into a dictionary.
        The keys are the field strengths.

        Returns:
            dict: Dictionary containing pandas dataframes with the scan results.
        """
        gasData = {}
        filePath = '../Data/ParallelPlate'
        for inField in self.fields:
            
            try:
                fileName = f'{self._gasComposition}.at{inField}kVcm.pkl'
                filename = os.path.join(filePath, fileName)

                inData = pd.read_pickle(filename)
                inData = self._calcVariability(inData)
                gasData[inField] = inData


            except Exception as e:
                print(f"Error reading pickle file {filename}: {e}. Skipping.")
                continue

        return gasData
    

#********************************************************************************#
    def plotGasScan(self, fieldStrength):
        """
        Plots the Isobutane scan data at a particular field strength.

        Incldues plots for the gain, threshold for 95% efficiency, and gain variability
        as functions of isobutane concentration.

        Args:
            fieldStrength (int): Field strength in kV/cm
        """

        if fieldStrength not in self.fields:
            raise ValueError('Error - Invalid field strength.')

        plotData = self.gasFieldData[fieldStrength]

        fig = plt.figure(figsize=(14, 4))
        fig.suptitle(f'Gas Scan (Plate Separation=100um, Field Strength=25kV/cm)')
        gain = fig.add_subplot(131)
        threshold = fig.add_subplot(132)
        variability = fig.add_subplot(133)

        label = f'CF4 = {plotData['CF4'].iloc[0]}%'
        if plotData['CF4'].iloc[0] == 3:
            label = label+' (T2K)'
            
        gain.errorbar(
            x=plotData['Isobutane'], y=plotData['meanGain'],
            xerr=None, yerr=plotData['gainErr'],
            ls='', marker='x',
            label=label
        )
        threshold.errorbar(
            x=plotData['Isobutane'], y=plotData['n_95'],
            xerr=None, yerr=plotData['n_95Err'],
            ls='', marker='x',
        )
        variability.errorbar(
            x=plotData['Isobutane'], y=plotData['variability'],
            xerr=None, yerr=plotData['variabilityErr'],
            ls='', marker='x',
        )


        gain.set_ylabel('nbar')
        threshold.set_ylabel('n_95')
        variability.set_ylabel('1 - n_95/nbar')

        gain.set_yscale('log')
        threshold.set_yscale('log')

        gain.set_title('Gas Gain')
        threshold.set_title('95% Threshold')
        variability.set_title('Gain Variabilty')

        for inax in [gain, threshold, variability]:
            inax.axvline(x=2, c='m', ls=':', label='T2K Isobutane %')
            inax.set_xlabel('Isobutane Concentration (%)')
            inax.grid()
        gain.legend()

        plt.tight_layout()
        return fig
    
#********************************************************************************#
    def plotGasEfficiencyScan(self):
        """
        Plots the Isobutane scan data with a 95% efficiency at 10e threshold requirement

        Incldues plots for the fieldm gain, threshold for 95% efficiency, and gain variability
        as functions of isobutane concentration.
        """
        plotData = self.gasEfficiencyData

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f'Gas 95% Efficiency Scan (Plate Separation=100um)')
        field = fig.add_subplot(221)
        gain = fig.add_subplot(222)
        threshold = fig.add_subplot(223)
        variability = fig.add_subplot(224)

        label = f'CF4 = {plotData['CF4'].iloc[0]}%'
        if plotData['CF4'].iloc[0] == 3:
            label = label+' (T2K)'
            
        field.errorbar(
            x=plotData['Isobutane'], y=plotData['E Field'],
            xerr=None, yerr=np.ones(len(plotData['E Field'])),
            ls='', marker='x',
            label=label
        )
        gain.errorbar(
            x=plotData['Isobutane'], y=plotData['meanGain'],
            xerr=None, yerr=plotData['gainErr'],
            ls='', marker='x',
        )
        threshold.errorbar(
            x=plotData['Isobutane'], y=plotData['n_95'],
            xerr=None, yerr=plotData['n_95Err'],
            ls='', marker='x',
        )        
        variability.errorbar(
            x=plotData['Isobutane'], y=plotData['variability'],
            xerr=None, yerr=plotData['variabilityErr'],
            ls='', marker='x',
        )

        meanGain = plotData['meanGain'].mean()
        expectedVar = 1 - 10/meanGain

        gain.axhline(y=meanGain, c='r', ls=':', label=f'Mean Gain = {meanGain:.1f}')
        threshold.axhline(y=10, c='r', ls='--', label='10e Threshold')
        variability.axhline(y=expectedVar, c='r', ls=':', label=f'Expected Variability = {expectedVar:.4f}')
            
            
        field.set_ylabel('E Field (kV/cm)')
        gain.set_ylabel(r'$\bar{n}$')
        threshold.set_ylabel(r'$n_{95}$')
        variability.set_ylabel(r'$1 - n_{95}/\bar{n}$')

        gain.set_yscale('log')
        gain.set_ylim([1e1, 1e3])

        field.set_title('Required Field Strength')
        gain.set_title('Gas Gain')
        threshold.set_title('95% Threshold')
        variability.set_title('Gain Variability')

        for inax in [field, gain, threshold, variability]:
            inax.axvline(x=2, c='m', ls=':', label='T2K Isobutane %')
            inax.set_xlabel('Isobutane Concentration (%)')
            inax.grid()
            inax.legend()
        #field.legend()

        plt.tight_layout()
        return fig
    
#********************************************************************************#
    def variabilitySWtest(self):
        """
        Performs a Shapiro-Wilk test on the gain variability.

        The null hypothesis of this test is that the data is normally distributed.

        Returns:
            shapiroW (float): W value result of the SW test
            shapiroPval (flaot): P value result of the SW test
            meanVariability (float): Mean value of the gain variability
        """
        meanGain = self.gasEfficiencyData['meanGain'].mean()
        meanThreshold = self.gasEfficiencyData['n_95'].mean()

        expectedVariability = 1 - 10/meanGain
        dataVariability = 1 - meanThreshold/meanGain

        meanVariability = self.gasEfficiencyData['variability'].mean()

        shapiroW, shapiroPval = shapiro(self.gasEfficiencyData['variability'])

        return shapiroW, shapiroPval, meanVariability



#********************************************************************************#
#********************************************************************************#
#********************************************************************************#
#********************************************************************************#
#********************************************************************************#

class magboltzSimulation:
    """
    A class representing Magboltz simulation data for electron transport properties
    within a gas.

    Indlues the drift velocity, transverse diffusion, and longitudinal diffusion.

    Attributes:
        gasID (str): The unique identifier for the gas mixture being analyzed.
        fields (list[int]): A list of electric field strengths available for 
            the specified gasID.
        magboltzData (pandas.DataFrame): A collection of transport properties 
            including drift velocity and diffusion (longitudinal/transverse) 
            with associated error margins.
        optimalDriftField (float): The electric field value (kV/cm) where the 
            maximum drift velocity occurs within a defined range.
        plotParam (dict): Aesthetic settings (name and color) used for 
            generating consistent plots.
    """


#********************************************************************************#
    def __init__(self, gasID=None):
        """
        Initializes the Magboltz data handler for a specific gas mixture.

        Args:
            gasID (str): The identifier string for the gas mixture
        """

        if gasID is not None:
            self.gasID = gasID
            self.fields = self._checkGasID()

            self.magboltzData = self._getMagboltzData()
            self.optimalDriftField = self._getOptimalDriftField()

            self.plotParam = {}

        return



#********************************************************************************#
    def _checkGasID(self):
        """
        Validates that the gasID exists within the Magboltz data directory.

        Returns:
            list[int]: A sorted list of available electric field strengths for the gas.
        """

        gasOptions = self._getGasIDandFields()

        if self.gasID not in gasOptions:
            raise ValueError(f'Error: No files with {self.gasID} present.')
        
        return gasOptions[self.gasID]


#********************************************************************************#
    def _getGasIDandFields(self):
        """
        Scans the data directory to map available gas types to their field strengths.

        Expected filename format: 'magboltz.gasName.fieldStrength.dat'

        Returns:
            dict: A dictionary where keys are gas identifiers (str) and values 
                are lists of associated electric field strengths (int).
        """

        try:
            filePath = '../Data/Magboltz'
            allFiles = os.listdir(filePath)

        except FileNotFoundError:
            print(f'Error: Directory not found at {filePath}')
            return {}
        
        gasOptions = defaultdict(list)

        for inFile in allFiles:
            parts = inFile.split('.')

            if len(parts) >= 3 and parts[0] == 'magboltz':
                gasString = parts[1]
                fieldString = parts[2]

                gasOptions[gasString].append(int(fieldString)) 
        
        return gasOptions

        

#********************************************************************************#
    def _getMagboltzData(self):
        """
        Loads the results for electron drift and diffusion 
        for a given gas as computed by Magboltz.

        Assumes that the files have been generated, 
        and parses those of the form 'magboltz.<gasComp>*.dat'.

        Returns:
            pandas.DataFrame: A DataFrame containing the parsed diffusion data, sorted by 'eField'.
                            None if no matching files are found. 
                            The DataFrame includes:
                            - 'filename'
                            - 'gasComposition'
                            - 'eField'
                            - 'driftVelocity'
                            - 'driftVelocityErr'
                            - 'diffusionLongitudinal'
                            - 'diffusionLongitudinalErr'
                            - 'diffusionTransverse'
                            - 'diffusionTransverseErr'
        """
        
        gasFilenames = f'magboltz.{self.gasID}*.dat'
        dataPath = os.path.join('../Data/Magboltz', gasFilenames)
        fileList = glob.glob(dataPath)

        if not fileList:
            print(f"No files found matching pattern '{dataPath}'.")
            return None

        dataMagboltz = []

        for inFile in fileList:
            inData = {"filename": os.path.basename(inFile)}
            
            try:
                with open(inFile, 'r') as file:
                    lines = [line.strip() for line in file.readlines()]
                
                if len(lines) < 11:
                    print(f"Error: File {inFile} has unexpected format - Skipping.")
                    continue

                inData['gasComposition'] = lines[3]
                inData['eField'] = float(lines[5])
                inData['driftVelocity'] = float(lines[7])
                inData['driftVelocityErr'] = float(lines[8])
                inData['diffusionLongitudinal'] = float(lines[10])
                inData['diffusionLongitudinalErr'] = float(lines[11])
                inData['diffusionTransverse'] = float(lines[13])
                inData['diffusionTransverseErr'] = float(lines[14])

                dataMagboltz.append(inData)
                
            except IndexError:
                print(f"Parsing Error: Could not find data at expected line index in {inFile}.")
            except ValueError:
                print(f"Parsing Error: Could not convert value to float in {inFile}.")
            except IOError as e:
                print(f"Error opening or reading file {inFile}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {inFile}: {e}")

        rawData = pd.DataFrame(dataMagboltz)

        sortedData = rawData.sort_values(by='eField').reset_index(drop=True)

        return sortedData
        
#********************************************************************************#
    def _getOptimalDriftField(self, maxDriftField=10):
        """
        Calculates the electric field value that produces the maximum drift velocity.

        Args:
            maxDriftField (float): The upper bound of the electric field to search. (in kV/cm)

        Returns:
            float: The electric field strength corresponding to the maximum drift velocity.
        """
        
        maxDriftVelocity = self.magboltzData['driftVelocity'][self.magboltzData['eField'] <= maxDriftField].max()    
        optimalDriftField = self.magboltzData['eField'][self.magboltzData['driftVelocity']==maxDriftVelocity].iloc[0]

        return optimalDriftField

#********************************************************************************#
    def setPlotParams(self, gasName, color):
        """
        Sets parameters for data visualization. 

        Args:
            gasName (str): The label to be used in the plot legend.
            color (str): The color to be used for lines and error bands.
        """

        self.plotParam['gasName'] = gasName
        self.plotParam['c'] = color

        return

#********************************************************************************#
    def plotMagboltz(self, fig=None, axes=None):
        """
        Generates or updates a two-panel plot of Magboltz gas properties.

        Includes plots for the electron drift velocity and the diffusion coefficiencts.
        If fig and axes are provided, they are updated. 
        If not provided then a new plot is created. 

        Args:
            fig (matplotlib.figure.Figure): An existing figure object. 
            axes (list of matplotlib.axes.Axes): A list/array of two axes.

        Returns:
            tuple: (fig, axes) The Matplotlib figure and axes objects used.
        """

        if fig is None and axes is None:
            newPlot = True
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f'Magboltz Calculations')
        else:
            newPlot = False

        velocity = axes[0]
        diffusion = axes[1]

        if not self.plotParam:
            raise ValueError('Error - Plot parameters not set.')

        ##### Drift Velocity #####
        velocity.plot(
            self.magboltzData['eField'], self.magboltzData['driftVelocity'],
            c=self.plotParam['c'], label=self.plotParam['gasName']
        )
        velocity.fill_between(
            self.magboltzData['eField'], 
            self.magboltzData['driftVelocity']*(1-self.magboltzData['driftVelocityErr']/100),
            self.magboltzData['driftVelocity']*(1+self.magboltzData['driftVelocityErr']/100),
            facecolor=self.plotParam['c'], alpha=.25
        )

        ##### Transverse #####
        diffusion.plot(
            self.magboltzData['eField'], self.magboltzData['diffusionTransverse'],
            ls='-', c=self.plotParam['c'], label=self.plotParam['gasName']+' - Transverse'
        )
        diffusion.fill_between(
            self.magboltzData['eField'], 
            self.magboltzData['diffusionTransverse']*(1-self.magboltzData['diffusionTransverseErr']/100),
            self.magboltzData['diffusionTransverse']*(1+self.magboltzData['diffusionTransverseErr']/100),
            facecolor=self.plotParam['c'], alpha=.25,
        )

        ##### Longitudinal #####
        diffusion.plot(
            self.magboltzData['eField'], self.magboltzData['diffusionLongitudinal'],
            ls='--', c=self.plotParam['c'], label=self.plotParam['gasName']+' - Longitudinal'
        )
        diffusion.fill_between(
            self.magboltzData['eField'], 
            self.magboltzData['diffusionLongitudinal']*(1-self.magboltzData['diffusionLongitudinalErr']/100),
            self.magboltzData['diffusionLongitudinal']*(1+self.magboltzData['diffusionLongitudinalErr']/100),
            facecolor=self.plotParam['c'], alpha=.25
        )

        ## Drift field ##
        velocity.axvline(
            x=self.optimalDriftField, 
            c=self.plotParam['c'], ls=':', label=f'Optimal Drift Field = {self.optimalDriftField:.2f} kV/cm'
        )
        diffusion.axvline(
            x=self.optimalDriftField, 
            c=self.plotParam['c'], ls=':'
        )


        if newPlot:
            for inax in axes:
                inax.axvline(
                    x=50, 
                    c='m', ls='--', label='Minimum field for T2K gain of 10'
                )
                inax.set_xlabel('Electric Field Strength (kV/cm)')
                inax.set_xscale('log')
                inax.grid()

            velocity.set_title('Electron Drift Velocity')
            velocity.set_ylabel('Drift Velocity (um/ns)')
            
            diffusion.set_title('Diffusion Coefficients')
            diffusion.set_ylabel('Diffusion (um/sqrt(cm))')

        #Update legends
        for inax in axes:
            inax.legend()
        
        plt.tight_layout()

        return fig, axes
    

#********************************************************************************#
    def loadOptimalDrift(self, gasID=None):
        """
        Loads pre-computed optimal drift field data from a .pkl file.
        """

        if gasID is None:
            raise ValueError("Error - No gasID input.")
        
        self.gasID = gasID

        filePath = '../Data/Magboltz'
        fileName = f'OptimalDriftFields.{self.gasID}.pkl'
        filename = os.path.join(filePath, fileName)

        self.optimalFieldData = pd.read_pickle(filename)  

        return
    

#********************************************************************************#
    def plotOptimalParam(self):
        """
        Plots the optimal Magboltz parameters across CF4 and Isobutane concentrations.
        """
        
        if not hasattr(self, 'optimalFieldData'):
            raise ValueError("Error - No optimal field data loaded.")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
        fig.suptitle(f'Electron Dynamics in Ar/CF4/Isobutane Mixtures', fontweight='bold', fontsize=16)
        
        plotParams = {
            'optimalField': ('Electric Field', 'kV/cm'),
            'driftVelocity': ('Drift Velocity', 'um/ns'),
            'diffusionTransverse': ('Transverse Diffusion', 'um/sqrt(cm)'),
            'diffusionLongitudinal': ('Longitudinal Diffusion', 'um/sqrt(cm)')
        }

        for ax, (key, (title, unit)) in zip(axes.flat, plotParams.items()):
            plotData = self.optimalFieldData.pivot(index='CF4', columns='Isobutane', values=key)

            mesh = ax.pcolormesh(
                plotData.columns, 
                plotData.index, 
                plotData.values, 
                shading='nearest', 
                cmap='viridis',
                norm='log' if key == 'optimalField' else None
            )

            ax.scatter(2, 3, marker='$T2K$', color='r', s=1000, label='T2K Gas')
            ax.scatter(0, 0, marker='$Pure~Ar$', color='r', s=1000, label='Pure Ar')
            
            ax.set_title(f'{title} at Optimal Drift Field', fontweight='bold')
            ax.set_xlabel('Isobutane Concentration (%)')
            ax.set_ylabel('CF4 Concentration (%)')

            fig.colorbar(mesh, ax=ax, label=f'{title} ({unit})')

        return fig
