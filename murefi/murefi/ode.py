import abc
import numpy
import scipy.integrate
from . core import Replicate, Timeseries, Dataset, ParameterMapping
from . import symbolic

try:
    import pymc3 as pm

except ModuleNotFoundError:  # pymc3 is optional, throw exception when used
    class _ImportWarnerPyMC3:
        __all__ = []

        def __init__(self, attr):
            self.attr = attr

        def __call__(self, *args, **kwargs):
            raise ImportError(
                "PyMC3 is not installed. In order to use this function:\npip install pymc3"
            )

    class _PyMC3:
        def __getattr__(self, attr):
            return _ImportWarnerPyMC3(attr)
    
    pm = _PyMC3()

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

        # Get only those y_hat values for which data exist
        # All keys in bmask corresponds to available data for observables
        bmask = template.get_observation_booleans(list(template.keys()))

        # Slice prediction into x_hat and y_hat 
        # Create Timeseries objects which are fed to a new Replicate object pred
        pred = Replicate(template.iid)
        for dependent_key, template_ts in template.items():
            independent_key = template_ts.independent_key
            mask = bmask[dependent_key]
            x_hat = template_ts.x
            y_hat = y_hat_all[independent_key][mask]
            pred[dependent_key] = Timeseries(x_hat, y_hat, independent_key=independent_key, dependent_key=dependent_key)
        return pred

    def symbolic_predict_replicate(self, parameters, template:Replicate):
        """Symbolically predict a replicate.

        Args:
            parameters (tt.TensorVariable): 1D Tensor of y0 and theta parameters
            template (Replicate): template that the prediction will be comparable with

        Returns:
            prediction (Replicate): symbolic predicted template (contains Timeseries with symbolic y-Tensors)
        """
        # TODO: merge this method with the non-symbolic predict_replicate

        assert not template is None, 'A template must be provided!'
        
        y0 = parameters[:self.n_y]
        theta = parameters[self.n_y:]
        x = template.x_any

        # symbolically predict for all timepoints
        y_hat_all = symbolic.IntegrationOp(self.solver, self.independent_keys)(y0, x, theta)
        y_hat_all = {
            ikey : y_hat_all[i]
            for i, ikey in enumerate(self.independent_keys)
        }
        
        # mask the prediction
        imask = template.get_observation_indices(list(template.keys()))
        
        # Slice prediction into x_hat and y_hat 
        # Create Timeseries objects which are fed to a new Replicate object pred
        pred = Replicate(template.iid)
        for dependent_key, template_ts in template.items():
            independent_key = template_ts.independent_key
            mask = imask[dependent_key]
            x_hat = template_ts.x
            y_hat = y_hat_all[independent_key][mask]
            pred[dependent_key] = Timeseries(x_hat, y_hat, independent_key=independent_key, dependent_key=dependent_key)
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
            prediction (Dataset):       prediction result
        """
        assert not template is None, 'A template must be provided!'
        
        prediction = Dataset()
        theta_dict = par_map.repmap(theta_fit)
        
        for element in theta_fit:
            if isinstance(element, pm.model.TransformedRV):
                for iid, replicate in template.items():
                    prediction[iid] = self.symbolic_predict_replicate(theta_dict[iid], replicate)
                return prediction
            
            elif isinstance(element, (int, float, str)):
                for iid, replicate in template.items():
                    prediction[iid] = self.predict_replicate(theta_dict[iid], replicate)
                return prediction
            
            else:
                raise Exception('Theta_fit contains unsupported data type')
                