import numpy as np
import matplotlib.pyplot as plt
import csv
import subprocess
import time


#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#This definition uses the parameters of the structure as inputs to analyze the output files
def analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit, runstart, currentStep):
#importing the data from Garfield++ and storing it onto an array
    driftlineData = np.loadtxt('output_file/driftline.csv', delimiter = ',')
    electricField = driftlineData*10000

#Finding the electric field transparency
    eFieldNum = 0
    totFieldLines = 0
    numTransFieldLines = 0
    for point in electricField[:, 1]:
        if electricField[eFieldNum, 1] - electricField[eFieldNum-1, 1] > (cathodeHeight + meshStandoff)/4:
            totFieldLines += 1
        if electricField[eFieldNum, 1] - electricField[eFieldNum-1, 1] > (cathodeHeight + meshStandoff)-1:
            numTransFieldLines += 1
        eFieldNum += 1

#Finding the width of the field bundle by first cutting the data to the region below a single hole
    cutEField = np.delete(electricField, np.where((electricField[:, 0] <= -holeRadius) | (electricField[:, 0] >= holeRadius) | (electricField[:, 1] > 0)), axis = 0)
    
#Finding the width of the field bundle by identifying the end points of the first and last field line
    endp = np.array([])
    endpz = np.array([])
    cutFieldNum = 0
    numFieldPoints = 0
    for point in cutEField[:, 0]:
        numFieldPoints += 1
        if cutEField[cutFieldNum, 1] - cutEField[cutFieldNum-1, 1] >= meshStandoff/2:
            endp = np.append(endp, cutEField[cutFieldNum-int(numFieldPoints/2), 0])
            endpz = np.append(endpz, cutEField[cutFieldNum-int(numFieldPoints/2), 1])
            numFieldPoints = 0
        cutFieldNum += 1
    width = abs(endp[1]) + abs(endp[-1])
    
    runend = time.perf_counter()

#writing the data to the output file
    with open('output_file/simData.csv', 'a') as f:
        print(f'{currentStep-1}, {holeRadius}, {meshThickness}, {cathodeHeight}, {meshStandoff}, {thicknessSiO2}, {pitch}, {pixelWidth}, {meshVoltage}, {numTransFieldLines/totFieldLines:0.3f}, {width:0.3f}, {runend-runstart:0.4f}', file = f)
    
#Checking the field transparency to determine if the simulation should stop
    if numTransFieldLines/totFieldLines < fieldTransLimit:
        with open('output_file/fieldTooSmall.csv', 'a') as f:
            print(f'{holeRadius}, {meshStandoff}, {meshVoltage/(meshStandoff/10)}, {width}',  file = f)
        print('Field line Transparency is too low. Ending Program')
        return True
    else:
        return False            


#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#This command outputs the given parameters to the runControl.txt file and the FIMS.sif file
def Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit):
    #with open("input_file/params.txt", 'w') as f:
    #    print(f'holeRadius = {holeRadius};', file = f)
    #    print(f'meshThickness = {meshThickness};', file = f)
    #    print(f'cathodeHeight = {cathodeHeight};', file = f)
    #    print(f'meshStandoff = {meshStandoff};', file = f)
    #    print(f'thicknessSiO2 = {thicknessSiO2};', file = f)
    #    print(f'pitch = {pitch};', file = f)
    #    print(f'pixelWidth = {pixelWidth};', file = f)
    
    
    with open('input_file/runControl.txt', 'r') as c:
        control = c.readlines()
    control[8] = f'pixelWidth = {pixelWidth};\n'
    control[10] = f'pitch = {pitch};\n'
    control[13] = f'meshStandoff = {meshStandoff};\n'
    control[14] = f'meshThickness = {meshThickness};\n'
    control[15] = f'holeRadius = {holeRadius};\n'
    control[18] = f'cathodeHeight = {cathodeHeight};\n'
    control[19] = f'thicknessSiO2 = {thicknessSiO2};\n'
    control[28] = f'fieldTransLimit = {fieldTransLimit};\n'
    with open('input_file/runControl.txt', 'w') as c:
        c.writelines(control)
    
    
    with open("input_file/FIMS.sif",'r') as f:
        FIMS = f.readlines()
    FIMS[149] = f'  Potential = {meshVoltage - (cathodeHeight/10)}\n'
    FIMS[155] = f'  Potential = {meshVoltage}\n'
    with open('input_file/FIMS.sif', 'w') as f:
        f.writelines(FIMS)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#This definition runs all of the commands from the terminal such as gmsh and Elmer
def Terminal_Commands():
    subprocess.run(['Programs/gmsh', 'input_file/FIMS.txt', '-3'])
    subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
    subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
    subprocess.run(['build/fieldlines'])


#-------------------------------------------------------------------------------------------------------------------------------------------------------------


#In this initial definition, variable is the parameter that the user wishes to iterate, mini is the initial value of that parameter, maxi is the final value, and steps is the number of tests that the user wishes to run.
def iterate_variable(variable, mini, maxi, steps):
    tstart = time.perf_counter()
    with open('output_file/simData.csv', 'w') as f:
        print()

#list of default values to be used for each parameter. Each geometry value is in microns, and the voltage is in volts
    holeRadius = 15.
    meshStandoff = 100.
    meshThickness = 4.
    thicknessSiO2 = 5.
    pitch = 27.5
    pixelWidth = 10.
    cathodeHeight = 400.
    
    
    fieldRatio = 40
    meshVoltage = -fieldRatio*meshStandoff/10
    fieldTransLimit = .999

#This if tree determines which parameter is being iterated and then loops through each step of the iteration as determined by the "steps" term in the initial definition    
    var = str(variable).lower()
    currentStep = 0

    if var == 'r':
        while currentStep <= steps:
            holeRadius = mini+currentStep*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit)
            Terminal_Commands()
            currentStep += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit, runstart, currentStep):
                break


    elif var == 'm':
        while currentStep <= steps:
            runstart = time.perf_counter()
            meshThickness = mini+currentStep*(maxi-mini)/steps
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit)
            Terminal_Commands()
            currentStep += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit, runstart, currentStep):
                break
 
 
    elif var == 'c':
        while currentStep <= steps:
            cathodeHeight = mini+currentStep*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit)
            Terminal_Commands()
            currentStep += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit, runstart, currentStep):
                break


    elif var == 'st':
        while currentStep<=steps:
            meshStandoff = mini+currentStep*(maxi-mini)/steps
            meshVoltage = -meshStandoff*fieldRatio/10
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit)
            Terminal_Commands()
            currentStep += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit, runstart, currentStep):
                break


    elif var == 'si':
        while currentStep <= steps:
            thicknessSiO2 = mini+currentStep*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit)
            Terminal_Commands()
            currentStep += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit, runstart, currentStep):
                break

    elif var == 'p':
        while currentStep <= steps:
            P = mini+currentStep*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit)
            Terminal_Commands()
            currentStep += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit, runstart, currentStep):
                break


    elif var == 'v':
        Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit)
        subprocess.run(['Programs/gmsh', 'input_file/FIMS.txt', '-3'])
        subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
        while currentStep <= steps:
            meshVoltage = mini+currentStep*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit)    
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            currentStep += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, fieldTransLimit, runstart, currentStep):
                break

    else:
        print('Please indicate which variable you wish to iterate (holeRadius "R", mesh thickness "M", Cathode Height "C", stand-off height "ST", pitch "P", or meshVoltage "V"')    
    tend = time.perf_counter()
    with open("output_file/simTime.txt",'a') as f:
        print(f'Total sim time is: {tend - tstart:0.4f} seconds', file=f)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#This line runs the program and should be edited by the user to match their desired test conditions
iterate_variable('r', 12, 13, 1)
