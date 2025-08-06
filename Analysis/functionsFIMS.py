import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gammaincc

from polyaClass import myPolya

#********************************************************************************#   
def getAnalysisNumbers():
    """
    """
    filename = 'analysisRunNumbers'

    if not os.path.exists(filename):
        with open(filename, "w") as file:
            file.write('-1')
            print(f"File '{filename}' created with default -1.")
            return [-1]

    allRunnos = []
    try:
        with open(filename, 'r') as file:
            for line in file:

                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    runNo = int(line.strip())
                    if runNo == -1:
                        continue
                    allRunnos.append(runNo)   
                    
                except ValueError:
                    print(f"Warning. Skipping non-integer line in '{filename}'.")
                    
    except Exception as e:
        print(f"An unexpected error occurred while reading '{filename}': {e}")
        return None

    return allRunnos
    
#********************************************************************************#   
def plotGeneralPolya(theta):
    """
    """
    n = np.linspace(0, 4, 101)
    plt.figure(figsize=(6, 4))
    
    for t in theta:
        generalPolya = myPolya(1, t)
        polyaProb = generalPolya.calcPolya(n)
        plt.plot(n, polyaProb,
                 label=r'$\theta$'+f' = {t:.2f}')
        
    plt.title(f"General Polya Distribution")
    plt.xlabel(r'Avalanche Size ($n/\bar{n}$)')
    plt.ylabel(r'$\bar{n}$ x Probability')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    return

#********************************************************************************#   
def plotPolya(theta):
    """
    """
    gain = [10, 25, 50, 75, 100]

    n = np.arange(0, 101, 1)
    
    numPlots = len(theta)
    numRows = int(np.ceil(numPlots/2))
    
    # Create the figure and add subplots
    fig, axes = plt.subplots(nrows=numRows, ncols=2, figsize=(12, 5*numRows))
    axesFlat = axes.flatten()

    fig.suptitle(f'Polya Avalanches')

    for i, t in enumerate(theta):
        for nBar in gain:
            plotPolya = myPolya(nBar, t)
            polyaProb = plotPolya.calcPolya(n)
            axesFlat[i].plot(n, polyaProb,
                             label=r'$\bar{n}$'+f' = {nBar:.0f}')

        axesFlat[i].set_title(r'$\theta$'+f' = {t:0.2f}')
        axesFlat[i].set_xlabel('Avalanche size')
        axesFlat[i].set_ylabel('Probability')
        axesFlat[i].legend()
        axesFlat[i].grid(True, alpha=0.5)

    for j in range(numPlots, len(axesFlat)):
        fig.delaxes(axesFlat[j])
    plt.show()
    return

#********************************************************************************#   
def plotPolyaEfficiency(theta):
    """
    """
    k = np.linspace(0, 1, 101) #Ratio: Threshold/Gain

    plt.figure(figsize=(12, 5))

    for t in theta:
        efficiency = gammaincc(t+1, k*(t+1))
        plt.plot(k, efficiency,
                 label=r"$\theta$"+f" = {t:0.2f}")

    targetEfficiency = 0.95
    plt.axhline(y=targetEfficiency,
                c='r', ls='--', label=f'{targetEfficiency*100:.0f}% Efficiency')
    plt.axvline(x=-np.log(targetEfficiency),
                c='r', ls=':', label=r'$\theta = 0$ Max: '+f'{-np.log(targetEfficiency):.3f}')

    plt.title(f"Efficiency of Polya: "
              +r"$\eta = \frac{\Gamma\left(\theta+1, (\theta+1)*n_{t}/\bar{n}\right)}{\Gamma\left(\theta+1\right)}$")
    plt.xlabel("Threshold / Gain Fraction: "
               +r"$n_{t} / \bar{n}$")
    plt.ylabel(f"Efficiency")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    return

#********************************************************************************#   
def plotThreshold():
    """
    """
    threshold = np.linspace(0, 16, 11)
    efficiency = [.95, .9]

    colors = ['b', 'r', 'g']

    plt.figure(figsize=(6, 4))
    
    for i, eff in enumerate(efficiency):
        gain = -threshold/np.log(eff)
        plt.plot(threshold, gain,
                 c=colors[i], label=f'Target Efficiency = {eff*100:.0f}%')

        polya5 = myPolya(1, 0.5)
        polya5.solveForGain(targetEff=eff, threshold=1)
        theta5 = threshold*polya5.gain
        plt.plot(threshold, theta5,
                 c=colors[i], ls=':', label=r'$\theta$ = 0.5')

        polya1 = myPolya(1, 1)
        polya1.solveForGain(targetEff=eff, threshold=1)
        theta1 = threshold*polya1.gain
        plt.plot(threshold, theta1,
                 c=colors[i], ls='--', label=r'$\theta$ = 1')

        polya2 = myPolya(1, 2)
        polya2.solveForGain(targetEff=eff, threshold=1)
        theta2 = threshold*polya2.gain
        plt.plot(threshold, theta2,
                 c=colors[i], ls='-.', label=r'$\theta$ = 2')
                 

    plt.title(f'Minimum Gain Required to Achieve Efficiency')
    plt.xlabel('Detector Threshold')
    plt.ylabel('Gain')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

#********************************************************************************#   
def plotPolyExamples(thetaStart=0, thetaEnd=5, numSteps=6):
    """
    """
    theta = np.linspace(thetaStart, thetaEnd, numSteps)

    plotGeneralPolya(theta)
    plotPolya(theta)
    plotPolyaEfficiency(theta)
    plotThreshold()
    


