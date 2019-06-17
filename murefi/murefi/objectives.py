import numpy

import calibr8
from . core import Timeseries, Replicate, Dataset, ParameterMapping
from . ode import BaseODEModel


def for_dataset(dataset: Dataset, model_template: BaseODEModel, par_map: ParameterMapping, error_models: calibr8.ErrorModel):
    """Creates an objective function for fitting a Dataset
    
    Args:
        dataset: Dataset object for which the parameters should be fitted.
        model_template (BaseODEModel): 
        par_map (ParameterMapping):
        error_models: list of calibr8.ErrorModel objects
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
                    L += error_model.loglikelihood(y_obs=data[key_pred_data].y, y_hat=predicted_replicate[key_pred_data].y)
        if numpy.isnan(L):
            return numpy.inf
        return -L
    return negative_loglikelihood_dataset
