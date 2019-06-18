import abc
import numpy
import scipy.integrate
from . core import Replicate, Timeseries, Dataset, ParameterMapping


class BaseODEModel(object):
    """A dynamic model that uses ordinary differential equations."""
    def __init__(self, independent_keys:tuple):
        """Create a dynamic model.
        Args:
            independent_keys (iterable): formula symbols of observables
        """
        self.independent_keys:tuple = tuple(independent_keys)
        self.n_y:int = len(self.independent_keys)
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
            for i, key in enumerate(self.independent_keys)
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
        theta_dict = par_map.repmap(theta_fit)
        
        for replicate_key, replicates in template.items():
            data = replicates
            Prediction[replicate_key] = self.predict_replicate(theta_dict[replicate_key], data)
        return Prediction
        