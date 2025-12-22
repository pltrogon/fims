######################################
# CLASS DEFINITION FOR MAGBOLTZ DATA #
######################################
from __future__ import annotations

import os
import pandas as pd
import awkward_pandas
import math
import matplotlib.pyplot as plt
import numpy as np
import glob



class magboltzData:
    """
    Docstring for magboltzData
    """


#********************************************************************************#
    # Initialize
    def __init__(self, gasComposition):
        """
        Initializes a data class containing the data from a specified gas composition
        """

        self.gasComp = gasComposition

        self.gasData = self._getMagboltzData()

        driftFieldData = self._getDriftFieldData()
        self.driftField = driftFieldData['fieldStrength']
        self.driftVelocity = driftFieldData['driftVeolcity']
        self.driftDiffusionT = driftFieldData['transverseDiffusion']
        self.driftDiffusionL = driftFieldData['longitudinalDiffusion']


#********************************************************************************#
    #String representation
    def __str__(self):
        """
        String representation of the magboltzData class.
        """
        return f'Gas composition: {self.gasComp}.'


#********************************************************************************#
    def _getMagboltzData(self):
        """
        Loads the results for electron drift and diffusion 
        for a given gas as computed by Magboltz.

        Assumes that the files have been generated, 
        and parses those of the form 'diffusion.<gasComp>*.dat'.


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
        
        gasFilenames = f'magboltz.{self.gasComp}*.dat'
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
def _getDriftFieldData(self, maxDriftField=10):
    """
    TODO
    """
    
    #Get index for maximum drift velocity in drift region
    driftFields = self.gasData['eField'] <= maxDriftField
    dataIndex = self.gasData['driftVelocity'][driftFields].idxmax()

    optimalDriftField = self.gasData['eField'].iloc[dataIndex]
    maxDriftVelocity = self.gasData['driftVelocity'].iloc[dataIndex]
    diffusionT = self.gasData['diffusionTransverse'].iloc[dataIndex]
    diffusionL = self.gasData['diffusionLongitudinal'].iloc[dataIndex]

    optimalField = {
        'fieldStrength': optimalDriftField,
        'driftVeolcity': maxDriftVelocity,
        'transverseDiffusion': diffusionT,
        'longitudinalDiffusion': diffusionL
    }
    return optimalField

#********************************************************************************#
def _getDriftDiffusion(self, maxDriftField=1):
    """
    TODO
    """
    
    maxDriftVelocity = self.gasData['driftVelocity'][self.gasData['eField'] <= maxDriftField].max()    
    optimalDriftField = self.gasData['eField'][self.gasData['driftVelocity']==maxDriftVelocity].iloc[0]

    return optimalDriftField

#********************************************************************************#
def plotDriftVelocity(self):
    """
    TODO
    """
    
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'Magboltz Calculations')
    
    ax1 = fig.add_subplot(111)
    
    ax1.plot(
        self.gasData['eField'], self.gasData['driftVelocity'],
    )
    ax1.fill_between(
        self.gasData['eField'], 
        self.gasData['driftVelocity']*(1-self.gasData['driftVelocityErr']/100),
        self.gasData['driftVelocity']*(1+self.gasData['driftVelocityErr']/100),
        alpha=.25
    )

    ax1.axvline(
        x=self.driftField, 
        ls=':', label=f'Drift Field = {self.driftField:.2f} kV/cm'
    )

    ax1.set_xlabel('Electric Field Strength (kV/cm)')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid()

    ax1.set_title(f'Drift Velocity {self.gasComp}')
    ax1.set_ylabel('Drift Velocity (um/ns)')

    plt.tight_layout()
    plt.show()

    return fig

#********************************************************************************#
def plotDiffusion(self):
    """
    TODO
    """
    
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'Magboltz Calculations')
    
    ax1 = fig.add_subplot(111)
    
    ##### Transverse #####
    ax1.plot(
        self.gasData['eField'], self.gasData['diffusionTransverse'],
        ls='-', label='Transverse'
    )
    ax1.fill_between(
        self.gasData['eField'], 
        self.gasData['diffusionTransverse']*(1-self.gasData['diffusionTransverseErr']/100),
        self.gasData['diffusionTransverse']*(1+self.gasData['diffusionTransverseErr']/100),
        alpha=.25
    )

    ##### Longitudinal #####
    ax1.plot(
        self.gasData['eField'], self.gasData['diffusionLongitudinal'],
        ls='--', label='Longitudinal'
    )
    ax1.fill_between(
        self.gasData['eField'], 
        self.gasData['diffusionLongitudinal']*(1-self.gasData['diffusionLongitudinalErr']/100),
        self.gasData['diffusionLongitudinal']*(1+self.gasData['diffusionLongitudinalErr']/100),
        alpha=.25
    )

    ax1.axvline(
        x=self.driftField, 
        ls=':', label=f'Drift Field = {self.driftField:.2f} kV/cm'
    )

    ax1.set_xlabel('Electric Field Strength (kV/cm)')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid()

    ax1.set_title(f'Diffusion Coefficients {self.gasComp}')
    ax1.set_ylabel('Diffusion (um/sqrt(cm))')

    plt.tight_layout()
    plt.show()

    return fig