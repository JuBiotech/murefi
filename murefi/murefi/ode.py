import abc
import numpy
import scipy.integrate

import calibr8
from . core import ParameterMapping
from . datastructures import Timeseries, Replicate, Dataset
from . import symbolic


class BaseODEModel(object):
    """A dynamic model that uses ordinary differential equations."""
    def __init__(self, theta_names:tuple, independent_keys:tuple):
        """Create a dynamic model.
        Args:
            independent_keys (iterable): formula symbols of observables
            theta_names (iterable): names of the model parameters in the correct order
        """
        self.theta_names = tuple(theta_names)
        self.independent_keys:tuple = tuple(independent_keys)
        self.n_y:int = len(self.independent_keys)
        super().__init__()
    
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

    def solver(self, y0, t, theta) -> dict:   
        """Solves the dynamic system for all timepoints in t.
        Uses scipy.integrate.odeint and self.dydt to solve the system.

        Args:
            y0 (array): initial state
            x (array): timepoints of the solution
            theta (array): system parameters
        Returns:
            dictionary with keys specified when object is created and values as numpy.ndarray for all timepoints in [t]
        """
        # must force odeint to start simulation at t=0
        concat_zero = t[0] != 0
        if concat_zero:
            t = numpy.concatenate(([0], t))
        y = scipy.integrate.odeint(self.dydt, y0, t, (theta,)) 
        if concat_zero:
            y = y[1:]
        y_hat_dict = {
            key : y[:,i] 
            for i, key in enumerate(self.independent_keys)
        }
        
        return y_hat_dict
    
    def predict_replicate(self, parameters, template:Replicate) -> Replicate:
        """Simulates an experiment that is comparable to the Replicate template with support for symbolically prediction.

        Args:
            parameters (array or tt.TensorVariable ): concatenation of y0 and theta parameters or 1D Tensor of the same
            template (Replicate): template that the prediction will be comparable with

        Returns:
            pred (Replicate): prediction result or symbolic predicted template (contains Timeseries with symbolic y-Tensors)
        """
        assert not template is None, 'A template must be provided!'
        
        y0 = parameters[:self.n_y]
        theta = parameters[self.n_y:]
        t = template.t_any

        if not calibr8.istensor([y0, theta]):
            y_hat_all = self.solver(y0, t, theta)

            # Get only those y_hat values for which data exist
            # All keys in masks corresponds to available data for observables
            masks = template.get_observation_booleans(list(template.keys()))
        
        else:
            # symbolically predict for all timepoints
            y_hat_all = symbolic.IntegrationOp(self.solver, self.independent_keys)(y0, t, theta)
            y_hat_all = {
                ikey : y_hat_all[i]
                for i, ikey in enumerate(self.independent_keys)
            }
        
            # mask the prediction
            masks = template.get_observation_indices(list(template.keys()))

        # Slice prediction into x_hat and y_hat 
        # Create Timeseries objects which are fed to a new Replicate object pred
        pred = Replicate(template.rid)
        for dependent_key, template_ts in template.items():
            independent_key = template_ts.independent_key
            mask = masks[dependent_key]
            t_hat = template_ts.t
            y_hat = y_hat_all[independent_key][mask]
            pred[dependent_key] = Timeseries(t_hat, y_hat, independent_key=independent_key, dependent_key=dependent_key)
        return pred
        
    def predict_dataset(self, template:Dataset, theta_mapping:ParameterMapping, theta_fit):
        """Simulates an experiment that is comparable to the Dataset template.
        Args:
            theta_mapping (ParameterMapping): Object of the ParameterMapping class containing
                                        all parameters as dictionary, the fitting parameters 
                                        as array and their bounds as list of tuples
                                        
            template (Dataset):         template that the prediction will be comparable with
            
            theta_fit:                  array with fitting parameters
        
        Returns:
            prediction (Dataset):       prediction result
        """
        assert not template is None, 'A template must be provided!'
        assert theta_mapping.order == self.theta_names, 'The parameter order must be compatible with the model!'
        
        prediction = Dataset()
        theta_dict = theta_mapping.repmap(theta_fit)

        for rid, replicate in template.items():
            prediction[rid] = self.predict_replicate(theta_dict[rid], replicate)
        return prediction
