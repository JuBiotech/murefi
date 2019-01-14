import abc
import numpy
import scipy.optimize

from .. error_model import ErrorModel


class GlucoseErrorModel(ErrorModel):
    def linear(self, y_hat, theta_lin):
        """Linear model of the expected measurement outcomes, given a true independent variable.
        
        Arguments:
            y_hat (array): realizations of the independent variable
            theta_lin (array): parameters of the linear model
        """
        return theta_lin[0]+theta_lin[1]*y_hat
    
    def constant(self, y_hat, theta_con):
        """Constant model for the width of the error distribution
        
        Arguments:
            y_hat (array): realizations of the independent variable
            theta_con (array): parameters of the constant model
        """
        return theta_con + 0*y_hat

    def error_model(self, y_hat, theta=None):
        if theta is None:
            theta = self.theta_fitted
        mu = self.linear(y_hat, theta[:2])
        sigma = self.constant(y_hat,theta[2:])
        return mu, sigma

    def inverse(self, y_obs, theta=None):
        """Make a recalibration using the inverse error model.

        Args:
            y_obs (array): observed OD measurements
            theta (array): parameter vector of the error model. defaults to theta_fitted

        Returns:
            glucose (array): recalibrated glucose concentrations
        """
        if theta is None:
            theta = self.theta_fitted
        a, b, sigma = theta
        mu = (y_obs - b) / y_obs
        return mu
    
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