import sys
import os
import uproot
import pandas as pd
import numpy as np
import awkward as ak
import math


CMTOMICRON = 1e4
VCMTOkVCM = 1e-3



# ==========================================
# DATA HANDLER CLASS
# ==========================================
class NIDData:
    DEFAULT_FILENAME = 'NIDData.root'

    def __init__(self, fileName: str = None):

        self.fileName = fileName if fileName else self.DEFAULT_FILENAME
        self.simData = None
        self.particleData = None
        self.eventData = None
        self.driftData = None
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
                        simDataDF[key] *= CMTOMICRON
                for key in fieldKeys:
                        simDataDF[key] *= VCMTOkVCM
                self.simData = simDataDF

            if 'particleDataTree' in file:
                particleDict = file['particleDataTree'].arrays(library="np")
                for key in ['x', 'y', 'z']:
                    particleDict[key] *= CMTOMICRON
                self.particleData = pd.DataFrame(particleDict)

                electronEvents = []
                for _, row in self.particleData.iterrows():
                    cp = row['changePoint']
                    mask = (cp == 1) | (cp == 2)
                    if np.any(mask):
                        electronEvents.append(pd.DataFrame({
                            'avalancheID': row['avalancheID'],
                            'particleID': row['particleID'],
                            'eventType': np.where(cp[mask] == 1, 'Detachment', 'Attachment'),
                            'x': row['x'][mask],
                            'y': row['y'][mask],
                            'z': row['z'][mask],
                        }))

                self.eventData = pd.concat(electronEvents, ignore_index=True) if electronEvents else pd.DataFrame()

            if 'driftDataTree' in file:
                driftDict = file['driftDataTree'].arrays(library="np")
                for key in ['x', 'y', 'z']:
                    driftDict[key] *= CMTOMICRON
                self.driftData = pd.DataFrame(driftDict)
                 

            if 'signalDataTree' in file:
                self.signalData = file['signalDataTree'].arrays(library='pd')

            return
