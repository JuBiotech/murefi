import abc
import numpy
import scipy.optimize


class ErrorModel(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, independent:str, dependent:str, key:str):
        """ A parent class providing the general structure of an error model.
        independent: independent variable of the eroor model
        depenedent: dependent variable of the error model
        key: key found in the Timeseries objects of both the observed data and the prediction
        """
        self.independent = independent
        self.dependent = dependent
        self.key = key
        self.theta_fitted = None
        return super().__init__()
    
    @abc.abstractmethod
    def error_model(self, y_hat, theta):
        raise NotImplementedException('The error_model function should be implemented by the inheriting class.')
    
    @abc.abstractmethod
    def inverse(self, y_hat, theta=None):
        raise NotImplementedException('The inverse function should be implemented by the inheriting class.')
        
    def evaluate_loglikelihood(self, y, y_hat):
        return self.loglikelihood(y, y_hat, self.theta_fitted)
            
    def loglikelihood(self, y, y_hat, theta_opt):
        raise NotImplementedException('The loglikelihood function should be implemented by the inheriting class.')
        
    def fit():
        raise NotImplementedException('The fitting function should be implemented by the inheriting class.')
  