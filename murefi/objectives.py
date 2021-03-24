import numpy
import typing

import calibr8
from . core import ParameterMapping
from . datastructures import Timeseries, Replicate, Dataset
from . ode import BaseODEModel
from . import symbolic


def for_dataset(dataset: Dataset, model: BaseODEModel, parameter_mapping: ParameterMapping, calibration_models: typing.Iterable[calibr8.CalibrationModel]):
    """Creates an objective function for fitting a Dataset
    
    Args:
        dataset: Dataset object for which the parameters should be fitted.
        model (BaseODEModel): ODE model
        parameter_mapping (ParameterMapping): murefi.ParameterMapping object
        calibration_models: list of calibr8.CalibrationModel objects

    Returns:
        objective: callable that takes a full parameter vector and returns the negative log-likelihood
    """
    if not parameter_mapping.order == model.parameter_names:
        raise ValueError(f'The parameter order in the mapping does not match with the model! ({parameter_mapping.order} != {model.parameter_names})')
    
    mappings = {
        rid : [
            # pairs of CalibrationModel and observed Timeseries
            (cm, rep_obs[cm.dependent_key])
            for cm in calibration_models
            if cm.dependent_key in rep_obs
        ]
        for rid, rep_obs in dataset.items()
    }
    
    def negative_loglikelihood_dataset(theta):
        is_symbolic = calibr8.istensor(theta)
        L = []
        prediction = model.predict_dataset(dataset, parameter_mapping, theta)

        for rid, em_ts_list in mappings.items():
            predicted_replicate = prediction[rid]
            for (cm, observed_ts) in em_ts_list:
                predicted_ts = predicted_replicate[cm.dependent_key]
                ll = cm.loglikelihood(
                    y=observed_ts.y, x=predicted_ts.y,
                    replicate_id=rid, dependent_key=cm.dependent_key
                ).sum()
                L.append(ll)

        if is_symbolic:
            return symbolic.theano.tensor.sum(L)
        else:
            L = numpy.sum(L)
            if numpy.isnan(L):
                return numpy.inf
            return -L
    return negative_loglikelihood_dataset
