import numpy as np
import ROOT
import math
import matplotlib.pyplot as plt
import csv
import subprocess
import time

#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#This command updates the iterated variable in the runControl.txt
#file and the FIMS.sif file and updates the run number in runNo.txt
def updateParam(holeRadius, meshStandoff, cathodeHeight,
                meshVoltage, cathodeVoltage, numFieldLine, simNum, currentStep):

#---------------------------------
    with open('input_file/runControl.txt', 'r') as c:
        control = c.readlines()
    
    control[13] = f'meshStandoff = {meshStandoff};\n'
    control[15] = f'holeRadius = {holeRadius};\n'
    control[23] = f'meshVoltage = {meshVoltage};\n'
    control[24] = f'cathodeVoltage = {cathodeVoltage};\n'
    control[27] = f'numFieldLine = {numFieldLine};\n'
    
    with open('input_file/runControl.txt', 'w') as c:
        c.writelines(control)
#---------------------------------
    with open("input_file/FIMS.sif",'r') as f:
        FIMS = f.readlines()
    
    FIMS[149] = f'  Potential = {meshVoltage - (cathodeHeight/10)}\n'
    FIMS[155] = f'  Potential = {meshVoltage}\n'
    
    with open('input_file/FIMS.sif', 'w') as f:
        f.writelines(FIMS)
#---------------------------------
    with open("input_file/runNo.txt",'w') as n:
        print(currentStep, file = n)
#---------------------------------


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#This definition runs all of the terminal processes 
def Terminal_Commands():

    subprocess.run(['Programs/gmsh', 'input_file/FIMS.txt', '-3'])
    subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh',
                    '-out', 'input_file', '-autoclean'])
    subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
    subprocess.run(['build/fieldlines'])
    subprocess.run(['build/avalanche'])

#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#This definition outputs all of the data to a single csv file for easy inspection
def Output_Data(currentStep, width, confidence, simTime):
    cmToMicron = 10000
    
    dataTree = ROOT.TFile.Open(f'output_file/avalancheDatasim.{currentStep}.root')
    simData = dataTree['metaDataTree']
    for thing in simData:
        meshStandoff = thing.meshStandoff*cmToMicron
        holeRadius = thing.holeRadius*cmToMicron
        meshVoltage = thing.meshVoltage
        numFieldLine = thing.numFieldLine
        transparency = thing.fieldTransparency
    
    simNum = int(np.loadtxt('input_file/simNum.txt'))
    
#writing the data to the output file
    with open(f'output_file/simData{simNum}.csv', 'a') as f:
        print(f'{currentStep}, {holeRadius:0.2f}, {meshStandoff}, '
              f'{meshVoltage}, {numFieldLine}, {transparency:0.3f}, {width:0.3f}, '
              f' {confidence}, {simTime:0.3f}', file = f)
    
#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#This definition uses the parameters of the structure as inputs
#analyze the output files
def analyze(holeRadius, meshStandoff, meshVoltage, fieldTransLimit,
                numFieldLine, runstart, currentStep):
    cmToMicron = 10000            

#importing the data from Garfield++

    dataTree = ROOT.TFile.Open(f'output_file/avalancheDatasim.{currentStep}.root')
    simData = dataTree['metaDataTree']
    for thing in simData:
        transparency = thing.fieldTransparency
    
    driftlineData = np.loadtxt('output_file/driftlineDiag.csv',
                               delimiter = ',')
    electricField = driftlineData*cmToMicron

#Finding the width of the field bundle by first cutting the data
#to the region below a single hole
    cutEField = np.delete(electricField, np.where(electricField[:, 2] > 0), axis = 0)
    
#Finding the width of the field bundle by identifying the
#end points of the first and last field line
    endpx = np.array([])
    endpy = np.array([])
    endpz = np.array([])
    cutFieldNum = 0
    numFieldPoints = 1
    
    for point in cutEField[:, 2]:
        if cutEField[cutFieldNum, 2] - cutEField[cutFieldNum-1, 2] >= meshStandoff/2:
            endpx = np.append(endpx, cutEField[cutFieldNum
                     - int(numFieldPoints/2), 0])
            endpy = np.append(endpy, cutEField[cutFieldNum
                     - int(numFieldPoints/2), 1])
            endpz = np.append(endpz, cutEField[cutFieldNum
                     - int(numFieldPoints/2), 2])
            numFieldPoints = 1
        cutFieldNum += 1
        numFieldPoints += 1
    
    if not len(endpx):
        print('No field Lines Detected')
        width = 100
    else:
        width = (abs(math.sqrt(endpx[1]**2+endpy[1]**2))
                 + abs(math.sqrt(endpx[-1]**2+endpy[-1]**2)))
    
    
#Calculating the confidence of the transparency
    confidence = math.sqrt(transparency*(1-transparency)/(numFieldLine**2))

#Calculating the simulation run time
    runend = time.perf_counter()
    simTime = runend - runstart
    
#Outputting the data to a csv file    
    Output_Data(currentStep, width, confidence, simTime)

#Checking the field transparency to determine if the simulation
#should stop
    if transparency < fieldTransLimit:    
        print('Field line Transparency is too low. Ending Program')
        return True
        
    else:
        return False


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------


#iterate_variable is the command that takes a specified variable,
#runs the simulation, and then adjusts the value before restarting.
#Variable is the parameter that the user wishes to iterate, initial
#is the starting value of that parameter, final is the final value
#of that parameter, and steps is the number of tests that the user
#wishes to run.
def iterate_variable(variable, initial, final, steps=10):

#program initialization
#Noting the simulation number
    simNum = int(np.loadtxt('input_file/simNum.txt'))+1
    if simNum > 9999:
        print('sim number is very high. Consider resetting')

    with open('input_file/simNum.txt', 'w') as r:
            print(simNum, file = r)
            
#Setting initial variables
    var = str(variable).lower()
    currentStep = 0
    
#Creating the data csv    
    with open(f'output_file/simData{simNum}.csv', 'a') as sim:
        print('step, radius, standoff, meshVoltage, numFieldLine, '
                'transparency, width, confidence, time', file = sim)

#importing values from runControl.txt
    with open('input_file/runControl.txt', 'r') as c:
        control = c.readlines()
    remove = [item[:-2] for item in control]
    
    pixelWidth = float(remove[8].partition('=')[2])
    pitch = float(remove[10].partition('=')[2])
    meshStandoff = float(remove[13].partition('=')[2])
    meshThickness = float(remove[14].partition('=')[2])
    holeRadius = float(remove[15].partition('=')[2])
    cathodeHeight = float(remove[18].partition('=')[2])
    thicknessSiO2 = float(remove[19].partition('=')[2])
    numFieldLine = float(remove[27].partition('=')[2])
    fieldTransLimit = float(remove[28].partition('=')[2])
    
    fieldRatio = 80 #--------------This should be defined in runControl.txt
    meshVoltage = -fieldRatio*meshStandoff/10
    cathodeVoltage = meshVoltage - cathodeHeight/10

#This if tree determines which parameter is being iterated and then
#loops through each step of the iteration as determined by the "steps"
#term in the original definition    
    if var == 'r':
        while currentStep < steps:
            holeRadius = initial + currentStep*(final - initial)/(steps - 1)
            runstart = time.perf_counter()
            updateParam(holeRadius, meshStandoff, cathodeHeight, 
                        meshVoltage, cathodeVoltage, numFieldLine,
                        simNum, currentStep)     
            Terminal_Commands()
            if analyze(holeRadius, meshStandoff, meshVoltage, fieldTransLimit,
                            numFieldLine, runstart, currentStep):
                break
            
            currentStep += 1

    elif var == 'st':
        while currentStep < steps:
            runstart = time.perf_counter()
            meshStandoff = initial + currentStep*(final - initial)/(steps - 1)
            meshVoltage = -meshStandoff*fieldRatio/10
            
            updateParam(holeRadius, meshStandoff, cathodeHeight, 
                        meshVoltage, cathodeVoltage, numFieldLine,
                        simNum, currentStep)
            Terminal_Commands()
            if analyze(holeRadius, meshStandoff, meshVoltage, fieldTransLimit,
                        numFieldLine, runstart, currentStep):
                break
                
            currentStep += 1

    elif var == 'fl':
        updateParam(holeRadius, meshStandoff, cathodeHeight, 
                        meshVoltage, cathodeVoltage, numFieldLine,
                        simNum, currentStep)
        subprocess.run(['Programs/gmsh', 'input_file/FIMS.txt', '-3'])
        subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh',
                    '-out', 'input_file', '-autoclean'])
        while currentStep < steps:
            runstart = time.perf_counter()
            numFieldLine = initial + currentStep*(final - initial)/(steps - 1)
            
            updateParam(holeRadius, meshStandoff, cathodeHeight, 
                        meshVoltage, cathodeVoltage, numFieldLine,
                        simNum, currentStep)
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            subprocess.run(['build/avalanche'])
            analyze(holeRadius, meshStandoff, meshVoltage, fieldTransLimit,
                       numFieldLine, runstart, currentStep)
                
            currentStep += 1

    elif var == 'v':
        updateParam(holeRadius, meshStandoff, cathodeHeight, 
                        meshVoltage, cathodeVoltage, numFieldLine,
                        simNum, currentStep)
        subprocess.run(['Programs/gmsh', 'input_file/FIMS.txt', '-3'])
        subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh',
                        '-out', 'input_file', '-autoclean'])
        
        while currentStep < steps:
            runstart = time.perf_counter()
            meshVoltage = initial + currentStep*(final - initial)/(steps - 1)
            
            updateParam(holeRadius, meshStandoff, cathodeHeight, 
                        meshVoltage, cathodeVoltage, numFieldLine,
                        simNum, currentStep)
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            subprocess.run(['build/avalanche'])
            if analyze(holeRadius, meshStandoff, meshVoltage, fieldTransLimit,
                        numFieldLine, runstart, currentStep):
                break
                
            currentStep += 1

#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#This line runs the program and should be edited by the user to match
#their desired test conditions
iterate_variable('st',120,90,3)
