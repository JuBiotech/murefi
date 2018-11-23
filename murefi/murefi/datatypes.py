import abc
import collections
import logging
import numpy
import scipy.stats
logger = logging.getLogger(__name__)
from . error_model import BiomassErrorModel 

class Timeseries(collections.Sized):
    """A timeseries represents observations of one transient variable at certain time points."""
    def __init__(self, ykey:str, x, y):
        """Bundles [x] and [y] into a timeseries.
        Args:
            ykey (str): symbol of the dependent variable (no . characters allowed)
            x (list or ndarray): timepoints
            y (list of ndarray): observations (same length as x)
        """
        assert isinstance(ykey, str)
        assert isinstance(x, (list, numpy.ndarray))
        assert isinstance(y, (list, numpy.ndarray))
        assert len(x) == len(y), 'x and y must have the same length.'
        assert numpy.array_equal(x, numpy.sort(x)), 'x must be monotonically increasing.'

        self.ykey = ykey
        self.x = numpy.array(x)
        self.y = numpy.array(y)
        return super().__init__()

    def __len__(self):
        return len(self.x)
    

class Replicate(collections.OrderedDict):
    """A replicate contains one or more timeseries."""
    def __init__(self, iid:str=None):
        """Create a data instance.
        Args:
            iid (str or None): the unique instance ID of the replicate
        """
        self.iid = iid
        if not hasattr(self, "default_x_any"):
            self.default_x_any = numpy.arange(0, 1, 0.1)
        return super().__init__()

    @property
    def x_any(self):
        """Array of x-values at which any variable was observed."""
        if len(self) > 0:
            return numpy.unique(numpy.hstack([
                ts.x
                for ykey,ts in self.items()
            ]))
        else:
            return self.default_x_any

    @property
    def x_max(self):
        """The value of the last observation timepoint."""
        return self.x_any[-1]

    def __setitem__(self, key:str, value:Timeseries):
        assert isinstance(value, Timeseries)
        assert key == value.ykey, f'The key in the Replicate ({key}) must be equal to the Timeseries.ykey ({value.ykey})'
        return super().__setitem__(key, value)
    
    
    def get_observation_booleans(self, keys_y:list) -> dict:
        """Gets the Boolean masks for observations of each y in [keys_y], relative to [x_any] and ts.x
        Args:
            keys_y (list): list of the observed variable names for which indices are desired
            x_any (array): array of timepoints that the indices shall be relative to
        Returns:
            dict: maps each ykey in keys_y to x_bmask (boolean mask with same size as x_any)
        """
        
        x_bmask = {}
        for yi,ykey in enumerate(keys_y):
            if ykey in self:
                ts = self[ykey]
                x_obs = ts.x
                assert numpy.all([x in self.x_any for x in x_obs]), 'Prediction timepoints [x_any] do ' \
                    f'not cover all observation timepoints of {ykey}.'
                x_bmask[ykey] = numpy.array([char in ts.x for i, char in enumerate(self.x_any)])
            else:
                x_bmask[ykey] = numpy.array([True for i in range(len(self.x_any))])
        return x_bmask

    
    def comparable_timeseries(self, x_hat, y_hat, y_hat_std, ts_obs):
        """ Grab comparable observations using the Boolean mask. """
        x_obs = ts_obs.x
        assert set(x_obs).issubset(set(x_hat)), 'Prediction timepoints [x_hat] do ' \
                    f'not cover all observation timepoints [x_obs] in {ts_obs.ykey}.'
        n_hat = len(x_hat)
        assert len(y_hat) == n_hat
        assert numpy.isscalar(y_hat_std) or len(y_hat_std) == n_hat
        y_obs = ts_obs.y
        # slice y_hat and y_hat_std to only those for which y_obs exists
        y_hat = y_hat[self.get_observation_booleans(ts_obs.ykey).get(ts_obs.ykey)]
        if not numpy.isscalar(y_hat_std):
            y_hat_std = y_hat_std[self.get_observation_booleans(ts_obs.ykey).get(ts_obs.ykey)]
        return y_hat, y_hat_std, y_obs

    @abc.abstractmethod
    def error_normal(self, y_hat, error_model: BiomassErrorModel):
        return error_model.error_model(y_hat, error_model.theta_fitted)
        

    def loglikelihood_ts(prediction, ts_obs:Timeseries, error_model: BiomassErrorModel):
        """Likelihood of the simulated timeseries y_hat taken from prediction(Replicate) given the observed timeseries y_obs."""
        assert ts_obs.ykey in prediction, f'{prediction.iid} did not contain a prediction of {ts_obs.ykey}'
        ts_hat = prediction[ts_obs.ykey]
        mu, sigma = prediction.error_normal(ts_hat.y, error_model)
        y_hat, y_hat_std, y_obs = prediction.comparable_timeseries(ts_hat.x, mu, sigma, ts_obs)
        # return the likelihood of the observations given the simulation
        return numpy.sum(numpy.log(scipy.stats.t.pdf(loc=y_hat, scale=y_hat_std, x=y_obs, df=1)))

    @staticmethod
    def loglikelihood(data, prediction, error_model: BiomassErrorModel) -> float:
        """Compute the log-likelihood of the Replicate object given the prediction.
        Args:
            data (Replicate): Measured or simulated data to be compared with the prediction
            prediction (Replicate): predicted data y_hat for all timepoints in self.x_any
            error_model (BiomassErrorModel): object were theta was already fitted with calibration data 
        Returns:
            float: mean log-likelihoods over observed timeseries
        """
        L = []
        for ykey,ts_obs in data.items():
            L.append(prediction.loglikelihood_ts(ts_obs, error_model))
        L = numpy.sum(L)
        if numpy.isnan(L):
            return -numpy.inf
        return numpy.sum(L)
    
class Dataset(collections.OrderedDict):
    """A dataset contains one or more Replicates."""
    __metaclass__ = abc.ABCMeta

    def __setitem__(self, key:str, value:Replicate):
        assert isinstance(value, Replicate)
        if not key == value.iid:
            logger.warn(f'The key "{key}" did not match value.iid "{value.iid}".' \
                         'Setting value.iid = key...')
            value.iid = key
        return super().__setitem__(key, value)
