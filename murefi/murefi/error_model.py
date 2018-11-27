import abc
import numpy
import scipy.optimize
from . datatypes import Timeseries, Replicate, Dataset
from . parameter_mapping import ParameterMapping
from . ode_models import BaseODEModel, MonodModel

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
        
    def evaluate_loglikelihood(self, y, y_hat):
        return self.loglikelihood(y, y_hat, self.theta_fitted)
            
    def loglikelihood(self, y, y_hat, theta_opt):
        raise NotImplementedException('The loglikelihood function should be implemented by the inheriting class.')
        
    def fit():
        raise NotImplementedException('The fitting function should be implemented by the inheriting class.')

class BiomassErrorModel(ErrorModel):
    def logistic(self, y_hat, theta_log):
        """Log-log logistic model of the expected measurement outcomes, given a true independent variable.
        
        Arguments:
            y_hat (array): realizations of the independent variable
            theta_log (array): parameters of the log-log logistic model
                I_x: inflection point (ln(x))
                I_y: inflection point (ln(y))
                Lmax: maximum value in log sapce
                s: log-log slope
        """
        # IMPORTANT: Outside of this function, it is irrelevant that the correlation is modeled in log-log space.
   
        # Since the logistic function is assumed for logarithmic backscatter in dependency of logarithmic NTU, 
        # the interpretation of (I_x, I_y, Lmax and s) is in terms of log-space.
        
        I_x, I_y, Lmax = theta_log[:3]
        s = theta_log[3:]
        
        # For the same reason, y_hat (the x-axis) must be transformed into log-space.
        y_hat = numpy.log(y_hat)
        
        y_val = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-4*s * (y_hat - I_x)))
        
        # The logistic model predicts a log-transformed y_val, but outside of this
        # function, the non-log value is expected.        
        return numpy.exp(y_val)
    
    def polynomial(self, y_hat, theta_pol):
        # Numpy's polynomial function wants to get the highest degree first
        return numpy.polyval(theta_pol[::-1], y_hat)
    
    def error_model(self, y_hat, theta):
        mu = self.logistic(y_hat, theta[:4])
        sigma = self.polynomial(y_hat,theta[4:])
        return mu, sigma
        
    def loglikelihood(self, y, y_hat, theta):
        mu, sigma = self.error_model(y_hat, theta)
        # using t-distributed error in the non-transformed space
        likelihoods = scipy.stats.t.pdf(x=y, loc=mu, scale=sigma, df=1)
        loglikelihoods = numpy.log(likelihoods)
        ll = numpy.sum(loglikelihoods)
        return ll
    
    def fit(self, y, y_hat, theta_guessed, bounds):
        def sum_negative_loglikelihood(theta):
            return(-self.loglikelihood(y, y_hat, theta))
        fit = scipy.optimize.minimize(sum_negative_loglikelihood, theta_guessed, bounds=bounds)
        self.theta_fitted = fit.x
        return fit
    
    
class ObjectiveFactory:
    def create_objective(dataset: Dataset, model_template: MonodModel, par_map: ParameterMapping, error_models):
        """ dataset: Dataset object for which the parameters should be fitted.
        model_template (MonodModel): 
        par_map (ParameterMapping):
        error_models: list of ErrorModel objects
           """
        def negative_loglikelihood_dataset(theta_fit):
            L = 0
            prediction = model_template.predict_dataset(dataset, par_map, theta_fit)
            for replicate_key, replicates in dataset.items():
                data = replicates
                predicted_replicate = prediction[replicate_key]
                # iterate over timeseries/error_models
                for error_model in error_models:
                    key_pred_data = error_model.key
                    if key_pred_data in data.keys():
                        #loglikelihood with y_hat converted into y_hat-NTU
                        y_hat_NTU = (predicted_replicate[key_pred_data].y)/0.00885685
                        L += error_model.evaluate_loglikelihood(y=data[key_pred_data].y, y_hat=y_hat_NTU)
            if numpy.isnan(L):
                return numpy.inf
            return -L
        return negative_loglikelihood_dataset
        
        
        # iterate over all error models - without error models we have no way to calculate L
        # each error model describes the likelihood of a dependent (key_obs)
        # variable as a function of the independent (key_pred) variable
        
                
       