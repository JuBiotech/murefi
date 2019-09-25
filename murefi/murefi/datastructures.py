import abc
import collections
import h5py
import logging
import numpy

import calibr8


logger = logging.getLogger(__name__)


class Timeseries(collections.Sized):
    """A timeseries represents observations of one transient variable at certain time points."""
    def __init__(self, x, y, *, independent_key:str, dependent_key:str):
        """Bundles [x] and [y] into a timeseries.

        Args:
            x (list or ndarray): timepoints
            y (list of ndarray): observations (same length as x)
            independent_key (str): key of the independent variable (no . or / characters allowed)
            dependent_key (str): key of the observed timeseries (no . or / characters allowed)
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

    def _to_dataset(self, grep:h5py.Group):
        """Store the Timeseries to a h5py.Dataset within the provided group.
        
        Args:
            grep (h5py.Group): parent group
        """
        ds = grep.create_dataset(
            self.dependent_key, (2,len(self)), dtype=float,
            data=(self.x, self.y)
        )
        ds.attrs['independent_key'] = self.independent_key
        ds.attrs['dependent_key'] = self.dependent_key
        return

    @staticmethod
    def _from_dataset(tsds:h5py.Dataset):
        """Read a Timeseries from a h5py.Dataset.
        
        Args:
            gts (h5py.Dataset): dataset of the timeseries
        """
        ts = Timeseries(
            x=tsds[0,:],
            y=tsds[1,:],
            independent_key=tsds.attrs['independent_key'],
            dependent_key=tsds.attrs['dependent_key']
        )
        return ts

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f'{self.dependent_key}[:{len(self)}]'

    def __repr__(self):
        return self.__str__()
    

class Replicate(collections.OrderedDict):
    """A replicate contains one or more timeseries."""
    def __init__(self, rid:str=None):
        """Create a data instance.
        Args:
            rid (str or None): the unique instance ID of the replicate
        """
        self.rid = rid
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
    def make_template(tmin:float, tmax:float, independent_keys:list, rid:str=None, N:int=100):
        """Create a dense template Replicate for plotting-predictions.

        Args:
            tmin (float): first timepoint
            tmax (float): last timepoint
            independent_keys (list): list of independent variable keys to include in the template
            rid (str): optional replicate id
            N (int): total number of timepoints (default: 100)
        
        Returns:
            replicate (Replicate): replicate object containing dense timeseries with random y data
        """
        x = numpy.linspace(tmin, tmax, N)
        rep = Replicate(rid)
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
        if not key == value.rid:
            logger.warn(f'The key "{key}" did not match value.rid "{value.rid}".' \
                         'Setting value.rid = key...')
            value.rid = key
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
        ds = Dataset()
        for rid in rids:
            ds[rid] = Replicate.make_template(tmin, tmax, independent_keys, rid=rid, N=N)
        return ds

    def save(self, filepath:str):
        """Saves the Dataset to a HDF5 file.

        Can be loaded with `murefi.load_dataset`.

        Args:
            filepath (str): file path or name to save
        """
        with h5py.File(filepath, 'w') as hfile:
            for rid, rep in self.items():
                grep = hfile.create_group(rid)
                for dkey, ts in rep.items():
                    ts._to_dataset(grep)
        return

    @staticmethod
    def load(filepath:str):
        """Load a Dataset from a HDF5 file.

        Args:
            filepath (str): path to the file containing the data

        Returns:
            dataset (Dataset)
        """
        ds = Dataset()
        with h5py.File(filepath, 'r') as hfile:
            for rid, grep in hfile.items():
                rep = Replicate(rid)
                for dkey, tsds in grep.items():
                    rep[dkey] = Timeseries._from_dataset(tsds)
                ds[rid] = rep
        return ds


def save_dataset(dataset:Dataset, filepath:str):
    """Saves a Dataset to a HDF5 file.

    Can be loaded with `murefi.load_dataset`.

    Args:
        filepath (str): file path or name to save
    """
    return dataset.save(filepath)


def load_dataset(filepath:str) -> Dataset:
    """Load a Dataset from a HDF5 file.

    Args:
        filepath (str): path to the file containing the data

    Returns:
        dataset (Dataset)
    """
    return Dataset.load(filepath)
    