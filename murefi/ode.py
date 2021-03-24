import abc
import numpy
import scipy.integrate
import typing

import calibr8
from . core import ParameterMapping
from . datastructures import Timeseries, Replicate, Dataset, DtypeError, ShapeError
from . import symbolic


class BaseODEModel(object):
    """A dynamic model that uses ordinary differential equations."""
    def __init__(self, parameter_names:tuple, independent_keys:tuple):
        """Create a dynamic model.

        Args:
            independent_keys (iterable): formula symbols of observables
            parameter_names (iterable): names of the model parameters in the correct order

        Attributes:
            parameter_names (tuple): the names of initial state and kinetic parameters
            independent_keys (tuple): independent keys of state variables
            n_parameters (int): number of initial state and kinetic parameters combined
            n_y0 (int): number of state variables
            n_ode_parameters (int): number of kinetic parameters
        """
        self.parameter_names:tuple = tuple(parameter_names)
        self.independent_keys:tuple = tuple(independent_keys)
        # derived from the inputs:
        self.n_parameters:int = len(self.parameter_names)
        self.n_y0:int = len(self.independent_keys)
        self.n_ode_parameters:int = self.n_parameters - self.n_y0
        super().__init__()

    @abc.abstractmethod
    def dydt(self, y, t, ode_parameters):
        """First derivative of the transient variables.
        Needs to be overridden by subclasses.
        
        Args:
            y (array): current state of the system (n_y0,)
            t (float): time since intial state
            ode_parameters (array): system parameters (n_ode_parameters,)
        Returns:
            array: change in y at time t
        """
        raise NotImplementedError()

    def solver(self, y0, t, ode_parameters) -> dict:   
        """Solves the dynamic system for all T timepoints in t.
        Uses scipy.integrate.odeint and self.dydt to solve the system.

        Args:
            y0 (array): initial states (n_y0,)
            t (array): timepoints of the solution
            ode_parameters (array): system parameters (n_ode_parameters,)
        Returns:
            y_hat (dict):
                keys are the independent keys of the model
                values are model states of shape (T,)
        """
        # must force odeint to start simulation at t=0
        concat_zero = t[0] != 0
        if concat_zero:
            t = numpy.concatenate(([0], t))
        y = scipy.integrate.odeint(self.dydt, y0, t, (ode_parameters,)) 
        # slicing and dict-conversion are dimensionality-agnostic
        if concat_zero:
            y = y[1:]
        y_hat_dict = {
            key : y[:,i]
            for i, key in enumerate(self.independent_keys)
        }
        return y_hat_dict

    def solver_vectorized(self, y0, t, ode_parameters) -> dict:   
        """Solves the dynamic system for all T timepoints in t with many parameter sets.
        Uses scipy.integrate.odeint and self.dydt to solve the system.

        Args:
            y0 (array): initial states (n_y0, n_sets)
            t (array): timepoints of the solution
            ode_parameters (array): system parameters (n_ode_parameters, n_sets)
        Returns:
            y_hat (dict):
                keys are the independent keys of the model
                values are model states of shape (T, n_sets)
        """
        # must force odeint to start simulation at t=0
        concat_zero = t[0] != 0
        if concat_zero:
            t = numpy.concatenate(([0], t))

        y0_shape = numpy.shape(y0)
        ode_parameters_shape = numpy.shape(ode_parameters)

        if y0_shape[0] != self.n_y0 or len(y0_shape) != 2:
            raise ShapeError('Invalid shape of initial states [y0].', actual=y0_shape, expected=f'({self.n_y0}, ?)')
        if ode_parameters_shape[0] != self.n_ode_parameters or len(ode_parameters_shape) != 2:
            raise ShapeError('Invalid shape of model parameters [ode_parameters].', actual=ode_parameters_shape, expected=f'({self.n_ode_parameters}, ?)')

        # there are many parametersets
        N_parametersets = y0_shape[1]
        y0 = numpy.atleast_2d(y0)
        ode_parameters = numpy.atleast_2d(ode_parameters)
        # reserve all memory for the results at once
        y = numpy.empty(shape=(len(t), self.n_y0, y0_shape[1]))
        # predict with each parameter set
        for s in range(N_parametersets):
            y[:,:,s] = scipy.integrate.odeint(self.dydt, y0[:,s], t, (ode_parameters[:,s],)) 

        # slice out augmented t0 timepoint and return as dict
        if concat_zero:
            y = y[1:]
        y_hat_dict = {
            key : y[:,i]
            for i, key in enumerate(self.independent_keys)
        }
        return y_hat_dict
    
    def predict_replicate(self, parameters:typing.Sequence, template:Replicate) -> Replicate:
        """Simulates an experiment that is comparable to the Replicate template.
        Supports symbolic prediction and vectorized prediction from a matrix of parameters.

        Args:
            parameters (array-like):
                The [parameters] sequence must be a tuple, list or numpy.ndarray, with the elements 
                being a concatenation of y0 and ode_parameters.

                Symbolic prediction requires a (n_parameters,) parameter vector of type {tuple, list, numpy.ndarray}.
                Elements may be a mix of scalars and Theano tensor variables.

                Prediction of distributions requires a (n_parameters,) parameter vector of type {tuple, list, numpy.ndarray}.
                Elements may be a mix of scalars and vectors, as long as all vectors have the same length.
            template (Replicate):
                template that the prediction will be comparable with

        Returns:
            pred (Replicate): prediction result (contains Timeseries with y being numpy.arrays or Tensors)
        """
        # check inputs (basic)
        if not isinstance(template, Replicate):
            raise ValueError('A template Replicate must be provided!')
        if not isinstance(parameters, (tuple, list, numpy.ndarray)):
            raise DtypeError('The provided [parameters] have the wrong type.', actual=type(parameters), expected='list, tuple or numpy.ndarray')
        P = len(parameters)
        if P != self.n_parameters:
            raise ShapeError('Invalid number of parameters.', actual=P, expected=self.n_parameters)

        # check that all parameters are either scalar, or (S,) vectors
        S = None
        symbolic_mode = calibr8.istensor(parameters)
        for pname, pval in zip(self.parameter_names, parameters):
            pdim = numpy.ndim(pval)
            pshape = numpy.shape(pval)
            if pdim == 1:
                if S is not None and pshape[0] != S:
                    raise ShapeError(f'Lenght of parameter values for {pname} is inconsistent with other parameters.')
                S = pshape[0]
            elif pdim > 1:
                raise ShapeError(
                    f'Entry for "{pname}" entry is more than 1-dimensional.',
                    actual=pshape,
                    expected='() or (?,)' if S is None else f'() or ({S},)'
                )
        if symbolic_mode and S is not None:
            raise DtypeError(
                'Symbolic prediction and numeric prediction of distributions are incompatible with each other. '
                'The [parameters] contained Tensors and vector-valued entries at the same time.'
            )

        # if any parameter entry is a vector, all must be vectors
        if S is not None:
            parameters = tuple(
                numpy.repeat(pars, S) if numpy.ndim(pars) == 0 else numpy.array(pars)
                for pars in parameters
            )

        # at this point, `parameters` is a tuple of either (symbolic) scalars, or (S,) vectors
        # and ready for prediction!

        # predictions are made for all timepoints and sliced to match the template
        t = template.t_any
        y0 = parameters[:self.n_y0]
        ode_parameters = parameters[self.n_y0:]
        y_hat_all = {}
        if symbolic_mode:
            masks = template.get_observation_indices(list(template.keys()))
            # symbolically predict for all timepoints
            if symbolic.HAVE_SUNODE:
                y_hat_all = symbolic.solve_sunode(
                    self.dydt,
                    self.independent_keys,
                    y0,
                    t,
                    ode_parameters,
                    self.parameter_names[self.n_y0:],
                )
            else:
                y_hat_tensor = symbolic.IntegrationOp(self.solver, self.independent_keys)(y0, t, ode_parameters)
                for i, ikey in enumerate(self.independent_keys):
                    y_hat_all[ikey] = y_hat_tensor[i]
            # y_hat_all is now a dictionary of symbolic predictions (full-length time)
        else:
            masks = template.get_observation_booleans(list(template.keys()))
            if S is None:
                # non-vectorized returns dict of (T,)
                y_hat_all = self.solver(y0, t, ode_parameters)
            else:
                # vectorized returns dict of (T, S)
                y_hat_all = self.solver_vectorized(y0, t, ode_parameters)

        # Create Timeseries objects from sliced 1D or 2D predictions
        pred = Replicate(template.rid)
        for dependent_key, template_ts in template.items():
            independent_key = template_ts.independent_key
            mask = masks[dependent_key]
            pred[dependent_key] = Timeseries(
                t=template_ts.t,
                y=y_hat_all[independent_key][mask,...].T,
                independent_key=independent_key,
                dependent_key=dependent_key
            )
        return pred
        
    def predict_dataset(self, template:Dataset, parameter_mapping:ParameterMapping, parameters:typing.Union[typing.Sequence, dict]):
        """Simulates an experiment that is comparable to the Dataset template.
        Args:
            parameter_mapping (ParameterMapping):
                maps elements in [parameters] to replicates in the [template]
            template (Dataset):
                template that the prediction will be comparable with
            parameters (array-like):
                prediction parameters, can be symbolic

        Returns:
            prediction (Dataset): prediction result
        """
        assert not template is None, 'A template must be provided!'
        if not parameter_mapping.order == self.parameter_names:
            raise ValueError('The parameter order must be compatible with the model!')
        
        prediction = Dataset()
        parameters_mapped = parameter_mapping.repmap(parameters)

        for rid, replicate in template.items():
            prediction[rid] = self.predict_replicate(parameters_mapped[rid], replicate)
        return prediction
