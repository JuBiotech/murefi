import numpy
import typing

import calibr8
from . core import ParameterMapping
from . datastructures import Timeseries, Replicate, Dataset
from . ode import BaseODEModel


def for_dataset(dataset: Dataset, model_template: BaseODEModel, theta_mapping: ParameterMapping, error_models: typing.Iterable[calibr8.ErrorModel]):
    """Creates an objective function for fitting a Dataset
    
    Args:
        dataset: Dataset object for which the parameters should be fitted.
        model_template (BaseODEModel): ODE model
        theta_mapping (ParameterMapping): murefi.ParameterMapping object
        error_models: list of calibr8.ErrorModel objects
    """
    assert theta_mapping.order == model_template.theta_names, 'The parameter order must be compatible with the model!'
    
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
        L = 0
        prediction = model_template.predict_dataset(dataset, theta_mapping, theta_fit)

        for rid, em_ts_list in mappings.items():
            predicted_replicate = prediction[rid]
            for (em, observed_ts) in em_ts_list:
                predicted_ts = predicted_replicate[em.dependent_key]
                L += em.loglikelihood(y=observed_ts.y, x=predicted_ts.y)
        
        if numpy.isnan(L):
            return numpy.inf
        return -L
    return negative_loglikelihood_dataset


def computation_graph_for_dataset(dataset: Dataset, model_template: BaseODEModel, theta_mapping: ParameterMapping, error_models: typing.Iterable[calibr8.ErrorModel], theta_fit):
    """Builds the computation graph for infering parameters of a Dataset with MCMC.
    
    Args:
        dataset: Dataset object for which the parameters should be fitted.
        model_template (BaseODEModel): ODE model
        theta_mapping (ParameterMapping): murefi.ParameterMapping object
        error_models: list of calibr8.ErrorModel objects
        theta_fit: symbolic parameter vector
    """
    assert theta_mapping.order == model_template.theta_names, 'The parameter order must be compatible with the model!'
    
    mappings = {
        rid : [
            # pairs of ErrorModel and observed Timeseries
            (em, rep_obs[em.dependent_key])
            for em in error_models
            if em.dependent_key in rep_obs
        ]
        for rid, rep_obs in dataset.items()
    }
    
    prediction = model_template.predict_dataset(dataset, theta_mapping, theta_fit)
    L = []
    for rid, em_ts_list in mappings.items():
        predicted_replicate = prediction[rid]
        for (em, observed_ts) in em_ts_list:
            predicted_ts = predicted_replicate[em.dependent_key]
            L.append(em.loglikelihood(y=observed_ts.y, x=predicted_ts.y, replicate_id=rid, dependent_key=em.dependent_key))
    return L

