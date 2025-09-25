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
    Polya parameters are theta (shape) and gain (mean or n-Bar).

    Distribution:
    
    P(n) = 
            1/nBar *
            (theta+1)^(theta+1) *
            1/Gamma(theta+1) *
            (n/nBar)^(theta) *
            e^(-(theta+1) * n/nBar)
    

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
        Allow for calling the Polya class like a function.
        
        Calculates the probability distribution, 
        and can set the gain/theta parameters.

        Args:
            n (float): Avalanche size for where to calculate probability.
            fGain (float): If provided, sets gain.
            fTheta (float): If provided, sets theta.

        Returns:
            float: Calculated Polya probability with the given parameters.
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
        Validation for the Polya parameters.

        Raises:
            ValueError: If gain or theta are None or not allowable values. 
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
        Calculates the probability density function of the Polya distribution
        for a given avalanche size.

        Args:
            n (float or np array): Avalanche sizes at which to calculate the probability.

        Returns:
            float or np array: Polya probabilities.
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
        Calculates the efficiency of the Polya distribution.

        This is the integral of the probabilities of avalanche sizes equal and
        greater than a given threshold in number of electrons.

        The fomula for this is the Regularized upper incomplete gamma function.
        
        Eff(n) = 
                Gamma(theta+1, (theta+1)*threshold/nBar) / Gamma(theta+1)
        
        where Gamma(a, x) is the upper incomplete gamma function.

        Args:
            threshold (float): The minimum detectable avalanche size.

        Returns:
            float: The calculated Polya efficiency.

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
    def calcEfficiencyErrs(self, threshold=0, gainErr=0, thetaErr=0):
        """
        Calculates the efficiency of the Polya distribution and its uncertainty.

        Args:
            threshold (float): The minimum detectable avalanche size.
            gainErr (float): The uncertainty in the gain.
            thetaErr (float): The uncertainty in theta.

        Returns:
            tuple: A tuple containing (efficiency, efficiency_error).
        """
        if gainErr < 0 or thetaErr < 0:
            raise ValueError('Errors cannot be negative') 
        
        #Save current params and calculate nominal efficiency
        gainSave = self.gain
        thetaSave = self.theta
        efficiency = self.calcEfficiency(threshold)

        self.gain = gainSave + gainErr
        self.theta = thetaSave + thetaErr
        effpp = self.calcEfficiency(threshold) - efficiency

        self.gain = gainSave + gainErr
        self.theta = thetaSave - thetaErr
        effpm = self.calcEfficiency(threshold) - efficiency

        self.gain = gainSave - gainErr
        self.theta = thetaSave + thetaErr
        effmp = self.calcEfficiency(threshold) - efficiency

        self.gain = gainSave - gainErr
        self.theta = thetaSave - thetaErr
        effmm = self.calcEfficiency(threshold) - efficiency


        maxErr = max(effpp, effpm, effmp, effmm)
        minErr = min(effpp, effpm, effmp, effmm)
        
        self.gain = gainSave
        self.theta = thetaSave
        return (efficiency, maxErr, minErr)

    
#********************************************************************************#   
    def _fsolveEfficiency(self, fsolveGain, threshold, targetEff):
        """
        Function to be used alongside 'fsolve' to determine the necessary gain 
        that yields a target efficiency with a specified threshold.

        Args:
            fsolveGain (float): The gain value to test.
            threshold (float): The threshold for efficiency calculation.
            targetEff (float): The target efficiency value.

        Returns:
            float: The difference between the caluclated and target efficiencies.
                   Is 1e10 if an error occurs.
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
        Solves for the gain necessary to achieve a specified efficiency with a
        a given threshold. Assumes theta is constant.

        Uses 'scipy.optimize.fsolve' to do so. Updates gain with the solution.

        Args:
            targetEff (float): The desired efficiency (between 0 and 1).
            threshold (float): The threshold for the efficiency calculation.
            initialGain (float): An initial guess for the gain.
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
                raise RuntimeError(f"fsolve did not converge. Message: {mesg['mesg']}")
        except Exception as e:
            raise RuntimeError(f'Error during fsolve for gain: {e}')
            
        return


#********************************************************************************#   
    def fitPolya(self, n, data, gain, dataErr=None, expo=False):
        """
        Fits a Polya distribution to an experimental dataset.

        Utilises 'scipy.optimize.curve_fit' to determine the gain and theta parameters
        that best represent the data. The solution is saved within the class.

        Optional parameter expo allows for flexibility to also fit to an exponential.
        This is achieved by limitting the theta parameter to be 0.

        Args:
            n (np.ndarray): Avalanche sizes.
            data (np.ndarray): Probability of avalanche size.
            gain (float): The raw gain or mean value of the data.
            dataErr (np.ndarray): The standard deviations of the probabilities.
            expo (bool): If True, reduces Polya to resemble exponential.
        """
        if expo: #Force theta to be 0 for exponential
            initial = [gain, 0]
            bounds = (
                [1, 0],
                [2*gain, 0.0001]
            )
        else:
            initial = [gain, 0.5] #TODO - 0, 0.5, or 1 - whats best here
            bounds = (
                [1, 0],
                [2*gain, 2]
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

        
            







        
