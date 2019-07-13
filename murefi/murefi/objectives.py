import numpy

import calibr8
from . core import Timeseries, Replicate, Dataset, ParameterMapping
from . ode import BaseODEModel


def for_dataset(dataset: Dataset, model_template: BaseODEModel, par_map: ParameterMapping, error_models: calibr8.ErrorModel):
    """Creates an objective function for fitting a Dataset
    
    Args:
        dataset: Dataset object for which the parameters should be fitted.
        model_template (BaseODEModel): ODE model
        par_map (ParameterMapping): murefi.ParameterMapping object
        error_models: list of calibr8.ErrorModel objects
    """
    
    mappings = {
        iid : [
            # pairs of ErrorModel and observed Timeseries
            (em, rep_obs[em.dependent_key])
            for em in error_models
            if em.dependent_key in rep_obs
        ]
        for iid, rep_obs in dataset.items()
    }
    
    def negative_loglikelihood_dataset(theta_fit):
        L = 0
        prediction = model_template.predict_dataset(dataset, par_map, theta_fit)

        for iid, em_ts_list in mappings.items():
            predicted_replicate = prediction[iid]
            for (em, observed_ts) in em_ts_list:
                predicted_ts = predicted_replicate[em.dependent_key]
                L += em.loglikelihood(y=observed_ts.y, x=predicted_ts.y)
        
        if numpy.isnan(L):
            return numpy.inf
        return -L
    return negative_loglikelihood_dataset


def computation_graph_for_dataset(dataset: Dataset, model_template: BaseODEModel, par_map: ParameterMapping, error_models: calibr8.ErrorModel, theta_fit):
    """Builds the computation graph for infering parameters of a Dataset with MCMC.
    
    Args:
        dataset: Dataset object for which the parameters should be fitted.
        model_template (BaseODEModel): ODE model
        par_map (ParameterMapping): murefi.ParameterMapping object
        error_models: list of calibr8.ErrorModel objects
        theta_fit: symbolic parameter vector
    """
    
    mappings = {
        iid : [
            # pairs of ErrorModel and observed Timeseries
            (em, rep_obs[em.dependent_key])
            for em in error_models
            if em.dependent_key in rep_obs
        ]
        for iid, rep_obs in dataset.items()
    }
    
    prediction = model_template.predict_dataset(dataset, par_map, theta_fit)
    L = []
    for iid, em_ts_list in mappings.items():
        predicted_replicate = prediction[iid]
        for (em, observed_ts) in em_ts_list:
            predicted_ts = predicted_replicate[em.dependent_key]
            L.append = em.loglikelihood(y=observed_ts.y, x=predicted_ts.y, replicate_id=iid, dependent_key=em.dependent_key)
    return L

