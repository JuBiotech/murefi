import numpy
from . datatypes import Timeseries, Replicate, Dataset
from . ode_models import BaseODEModel, MonodModel
from . parameter_mapping import ParameterMapping

class ObjectiveFactory:
    def create_objective(dataset: Dataset, model_template: MonodModel, par_map: ParameterMapping, error_models):
        """ dataset: Dataset object for which the parameters should be fitted.
        model_template (MonodModel): 
        par_map (ParameterMapping):
        error_models: list of ErrorModel objects
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
                            L += error_model.evaluate_loglikelihood(y=data[key_pred_data].y, y_hat=predicted_replicate[key_pred_data].y)
            if numpy.isnan(L):
                return numpy.inf
            return -L
        return negative_loglikelihood_dataset 