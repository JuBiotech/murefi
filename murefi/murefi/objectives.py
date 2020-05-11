import numpy
import typing

import calibr8
from . core import ParameterMapping
from . datastructures import Timeseries, Replicate, Dataset
from . ode import BaseODEModel


def for_dataset(dataset: Dataset, model: BaseODEModel, theta_mapping: ParameterMapping, error_models: typing.Iterable[calibr8.ErrorModel]):
    """Creates an objective function for fitting a Dataset
    
    Args:
        dataset: Dataset object for which the parameters should be fitted.
        model (BaseODEModel): ODE model
        theta_mapping (ParameterMapping): murefi.ParameterMapping object
        error_models: list of calibr8.ErrorModel objects

    Returns:
        objective: callable that takes a full parameter vector and returns the negative log-likelihood
    """
    if not theta_mapping.order == model.parameter_names:
        raise ValueError(f'The parameter order in the mapping does not match with the model! ({theta_mapping.order} != {model.parameter_names})')
    
    mappings = {
        rid : [
            # pairs of ErrorModel and observed Timeseries
            (em, rep_obs[em.dependent_key])
            for em in error_models
            if em.dependent_key in rep_obs
        ]
        for rid, rep_obs in dataset.items()
    }
    
    def negative_loglikelihood_dataset(theta_fit):
        is_symbolic = calibr8.istensor(theta_fit)
        L = [] if is_symbolic else 0
        prediction = model.predict_dataset(dataset, theta_mapping, theta_fit)

        for rid, em_ts_list in mappings.items():
            predicted_replicate = prediction[rid]
            for (em, observed_ts) in em_ts_list:
                predicted_ts = predicted_replicate[em.dependent_key]
                ll = em.loglikelihood(y=observed_ts.y, x=predicted_ts.y, replicate_id=rid, dependent_key=em.dependent_key)
                if is_symbolic:
                    L.append(ll)
                else:
                    L += ll
        
        if is_symbolic:
            return L
        if numpy.isnan(L):
            return numpy.inf
        return -L
    return negative_loglikelihood_dataset
