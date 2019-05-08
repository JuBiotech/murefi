import abc
import numpy
import scipy.optimize


class ErrorModel(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, independent:str, dependent:str, key:str):
        """ A parent class providing the general structure of an error model.

        Args:
            independent: independent variable of the error model
            dependent: dependent variable of the error model
            key: key found in the Timeseries objects of both the observed data and the prediction
        """
        self.independent = independent
        self.dependent = dependent
        self.key = key
        self.theta_fitted = None
        return super().__init__()
    
    @abc.abstractmethod
    def predict_dependent(self, y_hat, *, theta=None):
        """ Predicts the parameters of a probability distribution which characterises 
            the dependent variable given values of the independent variable.

        Args:
            y_hat (array): values of the independent variable
            theta: parameters of functions describing the mode and standard deviation of the PDF

        Returns:
            mu,sigma (array): values for mu and sigma characterising a PDF describing the dependent variable
        """
        raise NotImplementedError('The predict_dependent function should be implemented by the inheriting class.')
    
    @abc.abstractmethod
    def predict_independent(self, y_hat):
        """Predict the most likely value of the independent variable using the calibrated error model in inverse direction.

        Args:
            y_obs (array): observed measurements

        Returns:
            mu (array): predicted mode of the independent variable
        """
        raise NotImplementedError('The predict_independent function should be implemented by the inheriting class.')

    @abc.abstractmethod
    def infer_independent(self, y):
        """Infer the posterior distribution of the independent variable given the observations of one point of the dependent variable.
        Args:
            y_obs (array): observed measurements
        
        Returns:
            trace: trace of the posterior distribution of the inferred independent variable
        """  
        raise NotImplementedError('The infer_independent function should be implemented by the inheriting class.')
            
    def loglikelihood(self, *, y_obs,  y_hat, theta=None):
        """Loglikelihood of observations (dependent variable) given the independent variable

        Args:
            y_obs (array): observed backscatter measurements (dependent variable)
            y_hat (array): predicted values of independent variable
            theta: parameters describing the logistic function of mu and the polynomial function of sigma (to be fitted with data, otherwise theta=self.theta_fitted)
        
        Return:
            Sum of loglikelihoods

        """
        raise NotImplementedError('The loglikelihood function should be implemented by the inheriting class.')
        
    def fit(self, dependent, independent, *, theta_guessed, bounds=None):
        """Function to fit the error model with observed data. The attribute theta_fitted is overwritten after the fit.

        Args:
            dependent (array): observations of dependent variable
            independent (array): desired values of the independent variable or measured values of the same
            theta_guessed: initial guess for parameters describing the mode and standard deviation of a PDF of the dependent variable
            bounds: bounds to fit the parameters

        Returns:
            fit: Fitting result of scipy.optimize.minimize
        """
        raise NotImplementedError('The fitting function should be implemented by the inheriting class.')
  