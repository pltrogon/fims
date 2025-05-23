import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import csv
import subprocess
import time


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------
#This definition clears all the files used in Run.py
def createFiles(fimsRunNum):
    with open(f'output_file/simData{fimsRunNum}.csv', 'w') as f:
        print('step, radius, meshTh, cathodeH, Stand, SiO2, ',
              'pitch, pixelW, voltage, transparency, width, ',
              'time, confidence', file = f)

#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#This command outputs the given parameters to the runControl.txt
#file and the FIMS.sif file
def Set_Param(holeRadius, meshThickness, cathodeHeight, 
              meshStandoff, thicknessSiO2, pitch, 
              pixelWidth, meshVoltage, fieldTransLimit=.999, 
              numFieldLine=100):
 
 
    with open('input_file/runControl.txt', 'r') as c:
        control = c.readlines()
    control[8] = f'pixelWidth = {pixelWidth};\n'
    control[10] = f'pitch = {pitch};\n'
    control[13] = f'meshStandoff = {meshStandoff};\n'
    control[14] = f'meshThickness = {meshThickness};\n'
    control[15] = f'holeRadius = {holeRadius};\n'
    control[18] = f'cathodeHeight = {cathodeHeight};\n'
    control[19] = f'thicknessSiO2 = {thicknessSiO2};\n'
    control[27] = f'numFieldLine = {numFieldLine};\n'
    control[28] = f'fieldTransLimit = {fieldTransLimit};\n'
    with open('input_file/runControl.txt', 'w') as c:
        c.writelines(control)
    
    with open("input_file/FIMS.sif",'r') as f:
        FIMS = f.readlines()
    FIMS[149] = f'  Potential = {meshVoltage - (cathodeHeight/10)}\n'
    FIMS[155] = f'  Potential = {meshVoltage}\n'
    with open('input_file/FIMS.sif', 'w') as f:
        f.writelines(FIMS)


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#This definition runs all of the commands from the terminal 
def Terminal_Commands():

    subprocess.run(['Programs/gmsh', 'input_file/FIMS.txt', '-3'])
    subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh',
                    '-out', 'input_file', '-autoclean'])
    subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
    subprocess.run(['build/fieldlines'])

#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#This definition uses the parameters of the structure as inputs
#analyze the output files
def analyze(holeRadius, meshThickness, cathodeHeight, 
            meshStandoff, thicknessSiO2, pitch, pixelWidth, 
            meshVoltage, fieldTransLimit, numFieldLine, runstart, 
            currentStep, fimsRunNum):
            

#importing the data from Garfield++ and storing it onto an array
#Data multiplied by 10000 so that 1 micron is displayed simply as 1
    driftlineData = np.loadtxt('output_file/driftlineDiag.csv',
                               delimiter = ',')
    electricField = driftlineData*10000

#Finding the electric field transparency
    eFieldNum = 0
    totFieldLines = 0
    numTransFieldLines = 0
    
    for point in electricField[:, 2]:
        if electricField[eFieldNum, 2] - electricField[eFieldNum-1, 2] > cathodeHeight/1.2:      
            totFieldLines += 1
        if electricField[eFieldNum, 2] - electricField[eFieldNum-1, 2] > (cathodeHeight + meshStandoff):       
            numTransFieldLines += 1
        eFieldNum += 1
    
    transparency = numTransFieldLines/totFieldLines

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
    
    runend = time.perf_counter()

#Calculating the confidence of the transparency
    confidence = math.sqrt(transparency*(1-transparency)/numFieldLine)

#writing the data to the output file
    with open(f'output_file/simData{fimsRunNum}.csv', 'a') as f:
        print(f'{currentStep}, {holeRadius}, {meshThickness}, '
              f'{cathodeHeight}, {meshStandoff}, {thicknessSiO2}, '
              f'{pitch}, {pixelWidth}, {meshVoltage}, '
              f'{transparency:0.3f}, {width:0.3f}, '
              f'{runend-runstart:0.4f}, {confidence}', file = f)
    
#Checking the field transparency to determine if the simulation
#should stop
    if transparency < fieldTransLimit:
        with open('output_file/fieldTooSmall.csv', 'a') as f:
            print(f'{holeRadius}, {meshStandoff}, '
                  f'{-meshVoltage/(meshStandoff/10)}, {width}',
                  f'{numFieldLine},{confidence}', file = f)      
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
def iterate_variable(variable='r', initial=20, final=19, steps=1):
    
    tstart = time.perf_counter()
    fimsRunNum= int(np.loadtxt('input_file/runNo.txt'))
    createFiles(fimsRunNum)

#list of default values to be used for each parameter. Each
#geometry value is in microns, and the voltage is in volts
    holeRadius = 16.
    meshStandoff = 100
    meshThickness = 4.
    thicknessSiO2 = 5.
    pitch = 27.5
    pixelWidth = 10.
    cathodeHeight = 400.
    fieldRatio = 80
    meshVoltage = -fieldRatio*meshStandoff/10
    
    var = str(variable).lower()
    currentStep = 0
    fieldTransLimit = .999
    numFieldLine=200

#This if tree determines which parameter is being iterated and then
#loops through each step of the iteration as determined by the "steps"
#term in the original definition    
    if var == 'r':
        while currentStep <= steps:
            holeRadius = initial+currentStep*(final-initial)/steps
            runstart = time.perf_counter()
            
            Set_Param(holeRadius, meshThickness, cathodeHeight,
                      meshStandoff, thicknessSiO2, pitch,
                      pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)     
            Terminal_Commands()
            if analyze(holeRadius, meshThickness, cathodeHeight,
                       meshStandoff, thicknessSiO2, pitch,
                       pixelWidth, meshVoltage, fieldTransLimit,
                       numFieldLine, runstart, currentStep, fimsRunNum):
                break
            
            currentStep += 1

    elif var == 'm':
        while currentStep <= steps:
            runstart = time.perf_counter()
            meshThickness = initial+currentStep*(final-initial)/steps
            
            Set_Param(holeRadius, meshThickness, cathodeHeight,
                      meshStandoff, thicknessSiO2, pitch,
                      pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)
            Terminal_Commands()
            if analyze(holeRadius, meshThickness, cathodeHeight,
                       meshStandoff, thicknessSiO2, pitch,
                       pixelWidth, meshVoltage, fieldTransLimit,
                       numFieldLine, runstart, currentStep, fimsRunNum):
                    break
                
            currentStep += 1
 
    elif var == 'c':
        while currentStep <= steps:
            runstart = time.perf_counter()
            cathodeHeight = initial+currentStep*(final-initial)/steps
            
            Set_Param(holeRadius, meshThickness, cathodeHeight, 
                      meshStandoff, thicknessSiO2, pitch, 
                      pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)
            Terminal_Commands()
            if analyze(holeRadius, meshThickness, cathodeHeight,
                       meshStandoff, thicknessSiO2, pitch,
                       pixelWidth, meshVoltage, fieldTransLimit,
                       numFieldLine, runstart, currentStep, fimsRunNum):       
                break
                
            currentStep += 1

    elif var == 'st':
        while currentStep<=steps:
            runstart = time.perf_counter()
            meshStandoff = initial+currentStep*(final-initial)/steps
            meshVoltage = -meshStandoff*fieldRatio/10
            
            Set_Param(holeRadius, meshThickness, cathodeHeight,
                      meshStandoff, thicknessSiO2, pitch,
                      pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)
            Terminal_Commands()
            if analyze(holeRadius, meshThickness, cathodeHeight,
                       meshStandoff, thicknessSiO2, pitch,
                       pixelWidth, meshVoltage, fieldTransLimit,
                       numFieldLine, runstart, currentStep, fimsRunNum):
                break
                
            currentStep += 1

    elif var == 'si':
        while currentStep <= steps:
            runstart = time.perf_counter()
            thicknessSiO2 = initial+currentStep*(final-initial)/steps
            
            Set_Param(holeRadius, meshThickness, cathodeHeight,
                      meshStandoff, thicknessSiO2, pitch,
                      pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)
            Terminal_Commands()
            if analyze(holeRadius, meshThickness, cathodeHeight,
                       meshStandoff, thicknessSiO2, pitch,
                       pixelWidth, meshVoltage, fieldTransLimit,
                       numFieldLine, runstart, currentStep, fimsRunNum):
                break
                
            currentStep += 1

    elif var == 'p':
        while currentStep <= steps:
            runstart = time.perf_counter()
            P = initial+currentStep*(final-initial)/steps
            
            Set_Param(holeRadius, meshThickness, cathodeHeight,
                      meshStandoff, thicknessSiO2, pitch,
                      pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)
            Terminal_Commands()
            if analyze(holeRadius, meshThickness, cathodeHeight,
                       meshStandoff, thicknessSiO2, pitch,
                       pixelWidth, meshVoltage, fieldTransLimit,
                       numFieldLine, runstart, currentStep, fimsRunNum):
                break
                
            currentStep += 1
            
    elif var == 'fl':
        Set_Param(holeRadius, meshThickness, cathodeHeight,
                    meshStandoff, thicknessSiO2, pitch,
                    pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)
        subprocess.run(['Programs/gmsh', 'input_file/FIMS.txt', '-3'])
        subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh',
                    '-out', 'input_file', '-autoclean'])
        while currentStep <= steps:
            runstart = time.perf_counter()
            numFieldLine = initial+currentStep*(final-initial)/steps
            
            Set_Param(holeRadius, meshThickness, cathodeHeight,
                      meshStandoff, thicknessSiO2, pitch,
                      pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            analyze(holeRadius, meshThickness, cathodeHeight,
                       meshStandoff, thicknessSiO2, pitch,
                       pixelWidth, meshVoltage, fieldTransLimit,
                       numFieldLine, runstart, currentStep, fimsRunNum)
                
            currentStep += 1


    elif var == 'v':
        Set_Param(holeRadius, meshThickness, cathodeHeight,
                  meshStandoff, thicknessSiO2, pitch,
                  pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)
        subprocess.run(['Programs/gmsh', 'input_file/FIMS.txt', '-3'])
        subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh',
                        '-out', 'input_file', '-autoclean'])
        
        while currentStep <= steps:
            runstart = time.perf_counter()
            meshVoltage = initial+currentStep*(final-initial)/steps
            
            Set_Param(holeRadius, meshThickness, cathodeHeight,
                      meshStandoff, thicknessSiO2, pitch,
                      pixelWidth, meshVoltage, fieldTransLimit, numFieldLine)
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            if analyze(holeRadius, meshThickness, cathodeHeight,
                       meshStandoff, thicknessSiO2, pitch,
                       pixelWidth, meshVoltage, fieldTransLimit,
                       numFieldLine, runstart, currentStep, fimsRunNum):
                break
                
            currentStep += 1
              
    tend = time.perf_counter()
    with open("output_file/simTime.txt",'a') as f:
        print(f'Total sim time is: {tend - tstart:0.4f} seconds',
              file=f)


#---------------------------------------------------------
#**********************************************************************
#---------------------------------------------------------

#This line runs the program and should be edited by the user to match
#their desired test conditions
iterate_variable('r',16,16,1)
