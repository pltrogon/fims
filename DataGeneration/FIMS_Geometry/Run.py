import numpy as np
import matplotlib.pyplot as plt
import csv
import subprocess
import time


#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#This definition uses the parameters of the structure as inputs to analyze the output files
def analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, runstart, I):
#importing the data from Garfield++ and storing it onto an array
    Edata = np.loadtxt('output_file/driftline.csv', delimiter = ',')
    E_field = Edata*10000

#Finding the E_field_transparency
    m = 0
    field_tot = 0
    field_num = 0
    for point in E_field[:, 1]:
        if E_field[m, 1] - E_field[m-1, 1] > (cathodeHeight + meshStandoff)/4:
            field_tot += 1
        if E_field[m, 1] - E_field[m-1, 1] > (cathodeHeight + meshStandoff)-1:
            field_num += 1
        m += 1

#Finding the width of the field bundle by first cutting the data to the region below a single hole
    E_field_cut = np.delete(E_field, np.where((E_field[:, 0] <= -holeRadius) | (E_field[:, 0] >= holeRadius) | (E_field[:, 1] > 0)), axis = 0)
    
#Finding the width of the field bundle by identifying the end points of the first and last field line
    endp = np.array([])
    endpz = np.array([])
    i = 0
    j = 0
    for point in E_field_cut[:, 0]:
        j = j + 1
        if E_field_cut[i, 1] - E_field_cut[i-1, 1] >= 10:
            endp = np.append(endp, E_field_cut[i-int(j/2), 0])
            endpz = np.append(endpz, E_field_cut[i-int(j/2), 1])
            j = 0
        i = i + 1
    width = abs(endp[1]) + abs(endp[-1])
    
    runend = time.perf_counter()

#writing the data to the output file
    with open('output_file/Sim_Data.csv', 'a') as f:
        print(f'{I-1}, {holeRadius}, {meshThickness}, {cathodeHeight}, {meshStandoff}, {thicknessSiO2}, {pitch}, {pixelWidth}, {meshVoltage}, {field_num/field_tot:0.3f}, {width:0.3f}, {runend-runstart:0.4f}', file = f)
    
#Checking the field transparency to determine if the simulation should stop
    if field_num/field_tot < .999:
        with open('output_file/Field_Too_Small.csv', 'a') as f:
            print(f'{holeRadius}, {meshStandoff}, {meshVoltage/(meshStandoff/10)}, {width}',  file = f)
        print('Field line Transparency is too low. Ending Program')
        return True
    else:
        return False            


#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#This command outputs the given parameters to the run_control.txt file and the FIMS.sif file
def Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage):
    #with open("input_file/params.txt", 'w') as f:
    #    print(f'holeRadius = {holeRadius};', file = f)
    #    print(f'meshThickness = {meshThickness};', file = f)
    #    print(f'cathodeHeight = {cathodeHeight};', file = f)
    #    print(f'meshStandoff = {meshStandoff};', file = f)
    #    print(f'thicknessSiO2 = {thicknessSiO2};', file = f)
    #    print(f'pitch = {pitch};', file = f)
    #    print(f'pixelWidth = {pixelWidth};', file = f)
    
    
    with open('input_file/run_control.txt', 'r') as c:
        control = c.readlines()
    control[8] = f'pixelWidth = {pixelWidth};\n'
    control[10] = f'pitch = {pitch};\n'
    control[13] = f'meshStandoff = {meshStandoff};\n'
    control[14] = f'meshThickness = {meshThickness};\n'
    control[15] = f'holeRadius = {holeRadius};\n'
    control[18] = f'cathodeHeight = {cathodeHeight};\n'
    control[19] = f'thicknessSiO2 = {thicknessSiO2};\n'
    with open('input_file/run_control.txt', 'w') as c:
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
    with open('output_file/Sim_Data.csv', 'w') as f:
        print()

#list of default values to be used for each parameter. Each geometry value is in microns, and the voltage is in volts
    holeRadius = 15.
    meshStandoff = 100.
    pitch = 27.5
    meshThickness = 4.
    thicknessSiO2 = 5.
    cathodeHeight = 400.
    pixelWidth = 10.
    
    fieldRatio = 40
    meshVoltage = -fieldRatio*meshStandoff/10
        
#This if tree determines which parameter is being iterated and then loops through each step of the iteration as determined by the "steps" term in the initial definition    
    var = str(variable).lower()
    i = 0

    if var == 'r':
        while i <= steps:
            holeRadius = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage)
            Terminal_Commands()
            i += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, runstart, i):
                break


    elif var == 'm':
        while i <= steps:
            runstart = time.perf_counter()
            meshThickness = mini+i*(maxi-mini)/steps
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage)
            Terminal_Commands()
            i += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, runstart, i):
                break
 
 
    elif var == 'c':
        while i <= steps:
            cathodeHeight = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage)
            Terminal_Commands()
            i += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, runstart, i):
                break


    elif var == 'st':
        while i<=steps:
            meshStandoff = mini+i*(maxi-mini)/steps
            meshVoltage = -meshStandoff*fieldRatio/10
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage)
            Terminal_Commands()
            i += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, runstart, i):
                break


    elif var == 'si':
        while i <= steps:
            thicknessSiO2 = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage)
            Terminal_Commands()
            i += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, runstart, i):
                break

    elif var == 'p':
        while i <= steps:
            P = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage)
            Terminal_Commands()
            i += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, runstart, i):
                break


    elif var == 'v':
        Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage)
        subprocess.run(['Programs/gmsh', 'input_file/FIMS.txt', '-3'])
        subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
        while i <= steps:
            meshVoltage = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage)    
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i += 1
            if analyze(holeRadius, meshThickness, cathodeHeight, meshStandoff, thicknessSiO2, pitch, pixelWidth, meshVoltage, runstart, i):
                break

    else:
        print('Please indicate which variable you wish to iterate (holeRadius "R", mesh thickness "M", Cathode Height "C", stand-off height "ST", pitch "P", or meshVoltage "V"')    
    tend = time.perf_counter()
    with open("output_file/Sim_time.txt",'a') as f:
        print(f'Total sim time is: {tend - tstart:0.4f} seconds', file=f)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#This line runs the program and should be edited by the user to match their desired test conditions
iterate_variable('r', 18, 19, 1)
