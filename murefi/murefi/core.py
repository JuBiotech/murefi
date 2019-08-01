import abc
import collections
import logging
import numpy
import pandas
import scipy.stats

import calibr8


logger = logging.getLogger(__name__)


class Timeseries(collections.Sized):
    """A timeseries represents observations of one transient variable at certain time points."""
    def __init__(self, x, y, *, independent_key:str, dependent_key:str):
        """Bundles [x] and [y] into a timeseries.

        Args:
            x (list or ndarray): timepoints
            y (list of ndarray): observations (same length as x)
            independent_key (str): key of the independent variable (no . characters allowed)
            dependent_key (str): key of the observed timeseries (no . characters allowed)
        """
        assert isinstance(x, (list, numpy.ndarray))
        assert (isinstance(y, (list, numpy.ndarray)) or calibr8.istensor(y))
        assert isinstance(independent_key, str)
        assert isinstance(dependent_key, str)
        if not calibr8.istensor(y):
            assert len(x) == len(y), 'x and y must have the same length.'
        assert numpy.array_equal(x, numpy.sort(x)), 'x must be monotonically increasing.'

        self.x = numpy.array(x)
        self.y = numpy.array(y)  if not calibr8.istensor(y) else y   
        self.independent_key = independent_key
        self.dependent_key = dependent_key
        return super().__init__()

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f'{self.dependent_key}[:{len(self)}]'

    def __repr__(self):
        return self.__str__()
    

class Replicate(collections.OrderedDict):
    """A replicate contains one or more timeseries."""
    def __init__(self, iid:str=None):
        """Create a data instance.
        Args:
            iid (str or None): the unique instance ID of the replicate
        """
        self.iid = iid
        if not hasattr(self, 'default_x_any'):
            self.default_x_any = numpy.arange(0, 1, 0.1)
        return super().__init__()

    @property
    def x_any(self):
        """Array of x-values at which any variable was observed."""
        if len(self) > 0:
            return numpy.unique(numpy.hstack([
                ts.x
                for _, ts in self.items()
            ]))
        else:
            return self.default_x_any

    @property
    def x_max(self):
        """The value of the last observation timepoint."""
        return self.x_any[-1]

    def __setitem__(self, key:str, value:Timeseries):
        assert isinstance(value, Timeseries)
        assert key == value.dependent_key, f'The key in the Replicate ({key}) must be equal to the Timeseries.dependent_key ({value.dependent_key})'
        return super().__setitem__(key, value)
    
    def get_observation_booleans(self, keys_y:list) -> dict:
        """Gets the Boolean masks for observations of each y in [keys_y], relative to [x_any] and ts.x

        Args:
            keys_y (list): list of the timeseries keys for which indices are desired
            x_any (array): array of timepoints that the indices shall be relative to
        Returns:
            dict: maps each ykey in keys_y to x_bmask (boolean mask with same size as x_any)
        """
        x_bmask = {}
        x_any = self.x_any
        for yi, tskey in enumerate(keys_y):
            if tskey in self:
                x_bmask[tskey] = numpy.in1d(x_any, self[tskey].x)
            else:
                x_bmask[tskey] = numpy.repeat(False, len(x_any))
        return x_bmask

    def get_observation_indices(self, keys_y:list) -> dict:
        """Gets the index masks for observations of each y in [keys_y], relative to [x_any] and ts.x

        Args:
            keys_y (list): list of the timeseries keys for which indices are desired
            x_any (array): array of timepoints that the indices shall be relative to

        Returns:
            dict: maps each ykey in keys_y to x_imask (array of indices in x_any)
        """
        bmask = self.get_observation_booleans(keys_y)
        imask = {
            yk : numpy.arange(len(mask), dtype=int)[mask]
            for yk, mask in bmask.items()
        }
        return imask
    
    @staticmethod
    def make_template(tmin:float, tmax:float, independent_keys:list, iid:str=None, N:int=100):
        """Create a dense template Replicate for plotting-predictions.

        Args:
            tmin (float): first timepoint
            tmax (float): last timepoint
            independent_keys (list): list of independent variable keys to include in the template
            iid (str): optional replicate id
            N (int): total number of timepoints (default: 100)
        
        Returns:
            replicate (Replicate): replicate object containing dense timeseries with random y data
        """
        x = numpy.linspace(tmin, tmax, N)
        rep = Replicate(iid)
        for yk in independent_keys:
            rep[yk] = Timeseries(x, numpy.empty((N,)), independent_key=yk, dependent_key=yk)
        return rep
        
    def __str__(self):
        return f'Replicate({", ".join(map(str, self.values()))})'

    def __repr__(self):
        return self.__str__()
    
    
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

    @staticmethod
    def make_template(tmin:float, tmax:float, independent_keys:list, rids:list, N:int=100):
        """Create a dense template Dataset for plotting-predictions.

        Args:
            tmin (float): first timepoint
            tmax (float): last timepoint
            independent_keys (list): list of independent variable keys to include in the template
            rids (list): replicates ids that shall be present in the Dataset
            N (int): total number of timepoints (default: 100)
        
        Returns:
            dataset (Dataset): dataset object containing Replicates with dense timeseries of random y data
        """
        return {
            rid : Replicate.make_template(tmin, tmax, independent_keys, iid=rid, N=N)
            for rid in rids
        }


class ParameterMapping(object):
    @property
    def order(self) -> tuple:
        """Names of the model parameters"""
        return self._order
        
    @property
    def parameters(self) -> collections.OrderedDict:
        """Names of unique parameters in the mapping"""
        return self._parameters

    @property
    def ndim(self) -> int:
        """Dimensionality of the parameterization"""
        return len(self.parameters)
        
    @property
    def bounds(self) -> tuple:
        """(lower, upper) tuples for all parameters"""
        return self._bounds

    @property
    def guesses(self) -> tuple:
        """Initial guesses for all parameters"""
        return self._guesses

    @property
    def mapping(self) -> dict:
        """Dictionary of parameter names or values (floats and strings) for each replicate"""
        return self._mapping

    def __init__(self, mapping:pandas.DataFrame, *, bounds:dict, guesses:dict):
        """Helper object for mapping a global parameter vector to replicate-wise model parameters.
        
        The order of parameter columns will be preserved.
        
        Args:
            mapping (pandas.DataFrame): dataframe of parameter settings (replicate ids in rows, parameters in columns)
            bounds (dict): dictionary of customized bounds (unique parameter names > dimension names > None)
            guesses (dict): dictionary of customized initial guesses (unique parameter names > dimension names > None)
        """
        if bounds is None:
            bounds = dict()
        if guesses is None:
            guesses = dict()
        mapping = mapping.set_index(mapping.columns[0])

        self._order = tuple(mapping.columns)

        _parameters = {}
        for r, id in enumerate(mapping.index):
            for c, pname in enumerate(mapping.columns):
                v = mapping.loc[id, pname]
                if not str(v).replace('.', '', 1).isdigit():
                    if v in _parameters and _parameters[v] != pname:
                        raise ValueError(f'Unique parameter {v} is used in different model parameters.')
                    _parameters[v] = pname
        self._parameters = collections.OrderedDict([
            (key, _parameters[key])
            for key in sorted(_parameters.keys())
        ])
        
        self._bounds = tuple([
            bounds[pkey] if pkey in bounds else
            bounds[pdim] if pdim in bounds else
            (None, None)
            for pkey, pdim in self.parameters.items()
        ])

        self._guesses = tuple([
            guesses[pkey] if pkey in guesses else
            guesses[pdim] if pdim in guesses else
            None
            for pkey, pdim in self.parameters.items()
        ])

        self._mapping = {
            key : tuple(
                float(v) if str(v).replace('.', '', 1).isdigit() else str(v)
                for v in [mapping.loc[key, pname] for pname in self.order]
            )
            for key in mapping.index
        }
        return


    def repmap(self, theta_full):
        """Remaps a full parameter vector to a dictionary of replicate-wise parameter vectors.

        Args:
            theta_full (array): full parameter vector

        Returns:
            theta_dict (dict): dictionary of replicate-wise parameter vectors
        """
        pname_to_pvalue = {
            pname : theta_full[p]
            for p, pname in enumerate(self.parameters)
        }
        theta_dict = {
            rkey : tuple([
                pname_to_pvalue[pname] if isinstance(pname, str) else pname
                for pname in pnames
            ])
            for rkey, pnames in self.mapping.items()
        }
        return theta_dict
