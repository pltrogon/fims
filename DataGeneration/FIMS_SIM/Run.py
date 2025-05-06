import numpy as np
import matplotlib.pyplot as plt
import csv
import subprocess
import time

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#This definition uses the parameters of the structure as inputs to analyze the output files
def analyze(r0,tM,tA,tB,tC,P,L,V,runstart,I):
#importing the data from Garfield++ and storing it onto an array
    Edata= np.loadtxt('output_file/driftline.csv',delimiter=',')
    E_field = Edata*10000

#Finding the E_field_transparency
    m=0
    field_tot=0
    field_num=0
    for point in E_field[:,1]:
        if E_field[m,1]-E_field[m-1,1] > (tA+tB)/4:
            field_tot +=1
        if E_field[m,1]-E_field[m-1,1] > (tA+tB)-1:
            field_num +=1
        m+=1

#Finding the width of the field bundle by first cutting the data to the region below a single hole
    E_field_cut = np.delete(E_field, np.where((E_field[:,0] <= -r0) | (E_field[:,0] >= r0) | (E_field[:,1] > 10)), axis=0)
    
#Finding the width of the field bundle by identifying the end points of the first and last field line
    endp = np.array([])
    endpz = np.array([])
    i=0
    j=0
    for point in E_field_cut[:,0]:
        j=j+1
        if E_field_cut[i,1]-E_field_cut[i-1,1]>=10:
            endp = np.append(endp, E_field_cut[i-int(j/2),0])
            endpz = np.append(endpz, E_field_cut[i-int(j/2),1])
            j=0
        i = i+1
    width = abs(endp[1]) + abs(endp[-1])
    
    runend = time.perf_counter()
    
    with open('output_file/Sim_Data.csv','a') as f:
        print(f'{I-1}, {r0}, {tM}, {tA}, {tB}, {tC}, {P}, {L}, {V}, {field_num/field_tot:0.3f}, {width:0.3f}, {runend-runstart:0.4f}',file=f)
    
#Checking the field transparency to determine if the simulation should continue
    if field_num/field_tot < .999:
        with open('output_file/Field_Too_Small.csv', 'a') as f:
            print(f'{r0},{tB},{V/(tB/10)},{width}', file=f)
        print('Field line Transparency is too low. Ending Program')
        return True
    else:
        return False            

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#This command outputs the given parameters to the params.txt file
def Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V):
    with open("input_file/params.txt",'w') as f:
        print(f'r0={r0};', file=f)
        print(f'tM={tM};', file=f)
        print(f'tA={tA};', file=f)
        print(f'tB={tB};', file=f)
        print(f'tC={tC};', file=f)
        print(f'P={P};', file=f)
        print(f'L={L};', file=f)
        print(f'small={small};', file=f)
        print(f'med={med};', file=f)
        print(f'med2={med2};', file=f)
        print(f'large={large};', file=f)
    with open("input_file/FIMS.sif",'r') as f:
        FIMS = f.readlines()
    FIMS[149] = f'  Potential = {V-40}\n'
    FIMS[155] = f'  Potential = {V}\n'
    with open('input_file/FIMS.sif','w') as f:
        f.writelines(FIMS)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#In this initial definition, variable is the parameter that the user wishes to iterate, mini is the initial value of that parameter, maxi is the final value, and steps is the number of tests that the user wishes to run.
def iterate_variable(variable, mini, maxi, steps):
    tstart = time.perf_counter()
    with open('output_file/Sim_Data.csv', 'w') as f:
        print()
#list of default values to be used for each parameter
    r0 = 20
    tM = 4
    tA = 400
    tB = 100
    tC = 5
    P = 27.5
    L = 10
    small = .6
    med = 2.9
    med2 = 10
    large = 60
    i=0
    FR=40
    V=-FR*tB/10
    var=str(variable).lower()
    Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
    
#This if tree determines which parameter is being iterated and then loops through each step of the iteration as determined by the "steps" term in the initial definition    
    if var=='r':
        while i<=steps:
            r0 = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()#Adding a timer to note how long each run takes. The results of this timer will be saved to a Sim_time.txt file in the output_file folder
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)#Each loop is set to append the new parameters to the params.txt file, allowing for each step of the iteration to be individually checked and potentially used for calculations and analysis
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])#This line runs gmsh and creates a 3D finite element map using the given file paths
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])#These two lines run the Elmer software package. first, the ElmerGrid command converts the gmsh .msh file into an Elmer readable format and outputs to the input_file folder. Second, the ElmerSolver command runs the .sif command file and outputs the E-field into the input_file
            subprocess.run(['build/fieldlines'])#The final terminal runs the fieldlines garfield script and outputs to the output_file folder
            i+=1
            #The analyze function is defined above. 
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break
    elif var=='m':
        while i<=steps:
            runstart = time.perf_counter()
            tM = mini+i*(maxi-mini)/steps
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break
            
    elif var=='a':
        while i<=steps:
            tA = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break


    elif var=='b':
        while i<=steps:
            tB = mini+i*(maxi-mini)/steps
            V=-tB*FR/10
            runstart = time.perf_counter()
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break

    elif var=='c':
        while i<=steps:
            tC = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break

    elif var=='p':
        while i<=steps:
            P = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break

    elif var=='small':
        while i<=steps:
            small = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break

    elif var=='med' or var=='medium':
        while i<=steps:
            med = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break

    elif var=='med2' or var=='medium2':
        while i<=steps:
            med2 = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break

    elif var=='large':
        while i<=steps:
            large = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)
            subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
            subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break

    elif var=='v':
        subprocess.run(['Programs/gmsh-4.13.1-Linux64/bin/gmsh', 'input_file/FIMS.txt', '-3'])
        subprocess.run(['ElmerGrid', '14', '2', 'input_file/FIMS.msh', '-out', 'input_file', '-autoclean'])
        while i<=steps:
            V = mini+i*(maxi-mini)/steps
            runstart = time.perf_counter()
            Set_Param(r0,tM,tA,tB,tC,P,L,small,med,med2,large,V)    
            subprocess.run(['ElmerSolver', 'input_file/FIMS.sif'])
            subprocess.run(['build/fieldlines'])
            i+=1
            if analyze(r0,tM,tA,tB,tC,P,L,V,runstart,i):
                break

    else:
        print('Please indicate which variable you wish to iterate (radius "R", plate thickness "M", top air gap "A", stand-off height "B", SiO2 thickness "C", pitch "P", voltage ratio "V", or the mesh thicknesses "small, med, med2, or large"')    
    tend = time.perf_counter()
    with open("output_file/Sim_time.txt",'a') as f:
        print(f'Total sim time is: {tend - tstart:0.4f} seconds', file=f)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------


#This line runs the program and should be edited by the user to match their desired test conditions
iterate_variable('b',40,250,300)
