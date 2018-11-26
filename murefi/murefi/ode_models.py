import abc
import numpy
import scipy.integrate
from . datatypes import Replicate, Timeseries, Dataset
from . parameter_mapping import ParameterMapping


class BaseODEModel(object):
    """A dynamic model that uses ordinary differential equations."""
    def __init__(self, keys_y:tuple):
        """Create a dynamic model.
        Args:
            keys_y (iterable): Names of observables
        """
        self.keys_y:tuple = tuple(keys_y)
        self.n_y:int = len(self.keys_y)
        return super().__init__()
    
    @abc.abstractmethod
    def dydt(self, y, t, theta):
        """First derivative of the transient variables.
        Needs to be overridden by subclasses.
        
        Args:
            y (array): current state of the system
            t (float): time since intial state
            theta (array): system parameters
        Returns:
            array: change in y at time t
        """
        raise NotImplementedError()

    def solver(self, y0, x, theta) -> dict:   
        """Solves the dynamic system for all t in x.
        Uses scipy.integrate.odeint and self.dydt to solve the system.

        Args:
            y0 (array): initial state
            x (array): timepoints of the solution
            theta (array): system parameters
        Returns:
            dictionary with keys specified when object is created and values as numpy.ndarray for all t in [x]
        """
        # must force odeint to start simulation at t=0
        concat_zero = x[0] != 0
        if concat_zero:
            x = numpy.concatenate(([0], x))
        y = scipy.integrate.odeint(self.dydt, y0, x, (theta,)) 
        if concat_zero:
            y = y[1:]
        y_hat_dict = {
            key : y[:,i] 
            for i, key in enumerate(self.keys_y)
        }
        
        return y_hat_dict
    
    def predict_replicate(self, parameters, template:Replicate) -> Replicate:
        """Simulates an experiment that is comparable to the Replicate template.
        Args:
            parameters (array): concatenation of y0 and theta parameters
            template (Replicate): template that the prediction will be comparable with
        Returns:
            pred (Replicate): prediction result
        """
        assert not template is None, 'A template must be provided!'
        
        y0 = parameters[:self.n_y]
        theta = parameters[self.n_y:]
        x = template.x_any
        y_hat_all = self.solver(y0, x, theta)
        
        bmask = template.get_observation_booleans(list(template.keys()))
        
        #Get only those y_hat values for which data exist
        #All keys in bmask corresponds to available data for observables
        y_hat = {}
        for key in y_hat_all.keys():
             if key in list(bmask.keys()):
                    y_hat[key] = y_hat_all[key][bmask[key]]
             
        x_hat = {
            key : template[key].x 
            for key in y_hat_all.keys() 
            if key in list(bmask.keys())
        }
                    
        #Convert x_hat and y_hat entries in Timeseries objects which are fed to a new Replicate object pred
        
        pred = Replicate(template.iid )
        for key in y_hat.keys():
            pred[key] = Timeseries(key, x_hat[key], y_hat[key])
        
        return pred
    
    def predict_dataset(self, template:Dataset, par_map:ParameterMapping, theta_fit):
        """Simulates an experiment that is comparable to the Dataset template.
        Args:
            par_map (ParameterMapping): Object of the ParameterMapping class containing
                                        all parameters as dictionary, the fitting parameters 
                                        as array and their bounds as list of tuples
                                        
            template (Dataset):         template that the prediction will be comparable with
            
            theta_fit:                  array with fitting parameters
        
        Returns:
            Prediction (Dataset):       prediction result
        """
        assert not template is None, 'A template must be provided!'
        
        Prediction = Dataset()
        theta_fit_dic = {
            par_map.fitpars_array[i] : element 
            for i, element in enumerate(theta_fit)
        }
        
        for replicate_key, replicates in template.items():
            user_input = par_map.parameters_dic[replicate_key]
            theta = []
            for parameter in user_input:
                try:
                    float(parameter)
                    theta.append(float(parameter))
                except ValueError:
                    theta.append(theta_fit_dic[parameter])
            data = replicates
            Prediction[replicate_key] = self.predict_replicate(theta, data)
        return Prediction
    
    
class MonodModel(BaseODEModel):
    """ Class specifying the model for parameter fitting as Monod kinetics. """
    def dydt(self, y, t, theta):
        """First derivative of the transient variables.
        Args:
            y (array): array of observables
            t (float): time since intial state
            theta (array): Monod parameters
        Returns:
            array: change in y at time t
        """
        # NOTE: this method has significant performance impact!
        S, X = y
        mu_max, K_S, Y_XS = theta
        dXdt = mu_max * S * X / (K_S + S)
    
        yprime = [
            -1/Y_XS * dXdt,
            dXdt,
        ]
        return yprime