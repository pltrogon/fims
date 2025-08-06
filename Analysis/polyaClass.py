#######################################
# CLASS DEFINITION FOR POLYA FUNCTION #
#######################################
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.special import gamma
from scipy.special import gammaincc
from scipy.optimize import curve_fit, fsolve


class myPolya:
    """
    Class representing a Polya distribution.
    Polya parameters are theta (shape) and gain (mean).

    Distribution:
    $
    P(n) = 
            \frac{1}{\bar{n} *
            \frac{(\theta+1)^{(\theta+1)}{\Gamma(\theta+1)} * 
            (\frac{n}{\bar{n}})^{\theta}$ *
            e^{-n/\bar{n}(\theta+1)}
    $

    This is representative of the size of avalanches in electron multiplication.
    Note that when theta=0, this reduces to an exponential.
    """


 #********************************************************************************#   
    def __init__(self, gain=None, theta=None):
        """
        Initializes a Polya with a given gain and theta.

        Args:
            gain (float): The mean value of the distribution.
                          Must be greater than 0.
            theta (float): The shape parameter of the distribution.
                           Must be greater than or equal to 0.
        """

        self.gain = gain
        self.theta = theta

        if gain is not None and theta is not None:
            try:
                self._checkSelf()
            except ValueError as e:
                print(f'Error with given parameters during initialization: {e}')
                self.gain = None
                self.theta = None
                

 #********************************************************************************#   
    def __call__(self, n, fGain=None, fTheta=None):
        """
        Allow for calling the Polya like a function.
        
        Calculates the probability distribution, 
        and can sets the gain/theta parameters.

        Args:
            n (float): 
        """

        if fGain is not None:
            self.gain = fGain

        if fTheta is not None:
            self.theta = fTheta
            
        try:
            self._checkSelf()
        except ValueError as e:
            print(f'Polya calculation error: {e}.')
            return np.zeros_like(n, dtype=float)

        result = self.calcPolya(n)
        return result        

    
 #********************************************************************************#   
    def _checkSelf(self):
        """
        """
        if self.theta is None or self.gain is None:
            raise ValueError('Polya parameters are none.')

        if self.gain <= 0:
            raise ValueError('Gain must be greater than 0.')     
        if self.theta < 0:
            raise ValueError('Theta must be >= 0.')

        return
    

 #********************************************************************************#   
    def calcPolya(self, n):
        """
        """
        try:
            self._checkSelf()
        except ValueError as e:
            raise ValueError(f'Invalid parameters: {e}')
        
        A = 1/self.gain
        B = np.power(self.theta+1, self.theta+1)
        C = 1/gamma(self.theta+1)
        D = np.power(n/self.gain, self.theta)
        E = np.exp(-n/self.gain*(self.theta+1))

        result = A*B*C*D*E
        return result

    
 #********************************************************************************#   
    def calcEfficiency(self, threshold=0):
        """
        """
        try:
            self._checkSelf()
        except ValueError as e:
            raise ValueError(f'Invalid parameters: {e}')

        if threshold < 0:
            raise ValueError('Threshold cannot be negative.')

        s = self.theta+1
        x = s*threshold/self.gain

        efficiency = gammaincc(s, x)
        return efficiency

    
 #********************************************************************************#   
    def _fsolveEfficiency(self, fsolveGain, threshold, targetEff):
        """
        """
        self.gain = fsolveGain
        try:
            self._checkSelf()
            calcEff = self.calcEfficiency(threshold)

        except ValueError:
            return 1e10
            
        return targetEff - calcEff

    
 #********************************************************************************#   
    def solveForGain(self, targetEff=1, threshold=0, initialGain=1):
        """
        """
        saveGain = self.gain
        self.gain = initialGain

        if threshold < 0:
            raise ValueError('Threshold cannot be negative.')
        if not (0 <= targetEff <= 1):
            raise ValueError('Invalid target efficiency.')
            
        try:
            self._checkSelf()
        except ValueError as e:
            self.gain = saveGain
            raise ValueError(f'Invalid parameters: {e}')

        try: 
            gainSolved, info, _, mesg = fsolve(
                self._fsolveEfficiency,
                initialGain, 
                args=(threshold, targetEff),
                full_output=True
            )
            
            if info['fvec'] is not None and np.isclose(info['fvec'][0], 0.0, atol=1e-6):
                # Update the gain with the solved value
                self.gain = gainSolved[0]
            else:
                self.gain = saveGain
                raise RuntimeError(f'fsolve did not converge. Message: {mesg['mesg']}')
        except Exception as e:
            raise RuntimeError(f'Error during fsolve for gain: {e}')
            
        return


 #********************************************************************************#   
    def fitPolya(self, n, data, gain, dataErr=None, expo=False):
        """
        """
        if expo: #Force theta to be 0 for exponential
            initial = [gain, 0]
            bounds = (
                [1, 0],
                [2*gain, 0.01]
            )
        else:
            initial = [gain, 0.5] #TODO - 0, 0.5, or 1 - whats best here
            bounds = (
                [1, 0],
                [2*gain, 1]
            )

        try:
            popt, pcov = curve_fit(
                self,
                n,
                data,
                p0=initial,
                sigma=dataErr,
                bounds=bounds,
                absolute_sigma=True
            )
            
            self.gain = popt[0]
            self.theta = popt[1]
            perr = np.sqrt(np.diag(pcov))
            print(f'Fit converged. Gain: {self.gain:.3f} (+/- {perr[0]:.3f}), Theta: {self.theta:.3f} (+/- {perr[1]:.3f})')

        except RuntimeError as e:
            raise RuntimeError(f'Error during fitPolya: {e}.')
        except ValueError as e:
            raise ValueError(f'Value error during fitPolya: {e}.')
        except Exception as e:
            raise RuntimeError(f'An unexpected error occurred in fitPolya: {e}')

        return

        
            







        
