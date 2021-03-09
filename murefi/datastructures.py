import collections
import h5py
import logging
import numpy
import typing
import warnings

import calibr8


logger = logging.getLogger(__name__)


class ShapeError(Exception):
    """Error that the shape of a variable is incorrect."""
    def __init__(self, message, actual=None, expected=None):
        if expected and actual:
            super().__init__('{} (actual {} != expected {})'.format(message, actual, expected))
        else:
            super().__init__(message)


class DtypeError(TypeError):
    """Error that the dtype of a variable is incorrect."""
    def __init__(self, message, actual=None, expected=None):
        if expected and actual:
            super().__init__('{} (actual {} != expected {})'.format(message, actual, expected))
        else:
            super().__init__(message)


class Timeseries(collections.abc.Sized):
    """A timeseries represents observations of one transient variable at certain time points."""
    def __init__(self, t, y, *, independent_key:str, dependent_key:str):
        """Bundles [t] and [y] into a timeseries.

        Args:
            t (list or ndarray): timepoints (T,)
            y (list of ndarray): values with shape (T,) or distribution of values with shape (?, T)
            independent_key (str): key of the independent variable (no . or / characters allowed)
            dependent_key (str): key of the observed timeseries (no . or / characters allowed)
        """
        if not isinstance(t, (tuple, list, numpy.ndarray)):
            raise DtypeError(f'Argument [t] had the wrong type.', actual=type(t), expected='tuple, list or numpy.ndarray')
        if not (isinstance(y, (tuple, list, numpy.ndarray)) or calibr8.istensor(y)):
            raise DtypeError(f'Argument [y] had the wrong type.', actual=type(y), expected='tuple, list, numpy.ndarray or TensorVariable')
        if not isinstance(independent_key, str):
            raise DtypeError(independent_key, actual=type(independent_key), expected=str)
        if not isinstance(dependent_key, str):
            raise DtypeError(dependent_key, actual=type(dependent_key), expected=str)
        
        if not numpy.array_equal(t, numpy.sort(t)):
            raise ValueError('t must be monotonically increasing.')

        T = len(t)
        if not calibr8.istensor(y):
            y = numpy.atleast_1d(y)
            if (y.ndim == 1 and not y.shape == (T,)) \
                or (y.ndim == 2 and y.shape[1] != T):
                raise ShapeError(f'Argument [y] had the wrong shape.', actual=y.shape, expected=f'({T},) or (?, {T})')


        self.t = numpy.array(t)
        self.y = numpy.array(y)  if not calibr8.istensor(y) else y   
        self.independent_key = independent_key
        self.dependent_key = dependent_key
        super().__init__()

    @property
    def is_distribution(self) -> bool:
        """Indicates if the observations are available as a distribution or not."""
        return self.y.ndim > 1

    def _to_dataset(self, grep:h5py.Group):
        """Store the Timeseries to a h5py.Dataset within the provided group.
        
        Args:
            grep (h5py.Group): parent group
        """
        y_2d = numpy.atleast_2d(self.y)
        data = numpy.insert(y_2d, 0, values=self.t, axis=0)
        assert data.shape[0] == y_2d.shape[0] + 1
        assert data.shape[1] == len(self.t)
        # the data is saved as ONE matrix of shape (1 + N_y, N_t):
        # --> vector of times is the first row
        ds = grep.create_dataset(
            self.dependent_key, data.shape, dtype=float,
            data=data
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
            t=tsds[0,:],
            # load y as a vector, unless there's more than one row for it
            y=tsds[1,:] if tsds.shape[0] == 2 else tsds[1:,:],
            independent_key=tsds.attrs['independent_key'],
            dependent_key=tsds.attrs['dependent_key']
        )
        return ts

    def __len__(self):
        return len(self.t)

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
        if not hasattr(self, 'default_t_any'):
            self.default_t_any = numpy.arange(0, 1, 0.1)
        super().__init__()

    @property
    def t_any(self) -> typing.Optional[numpy.ndarray]:
        """Array of time values at which any variable was observed."""
        if len(self) > 0:
            return numpy.unique(numpy.hstack([
                ts.t
                for _, ts in self.items()
            ]))
        else:
            return None

    @property
    def t_max(self) -> float:
        """The value of the last observation timepoint."""
        return self.t_any[-1]

    def __setitem__(self, key:str, value:Timeseries):
        assert isinstance(value, Timeseries)
        assert key == value.dependent_key, f'The key in the Replicate ({key}) must be equal to the Timeseries.dependent_key ({value.dependent_key})'
        return super().__setitem__(key, value)
    
    def get_observation_booleans(self, keys_y:list) -> dict:
        """Gets the Boolean masks for observations of each y in [keys_y], relative to [t_any] and ts.x

        Args:
            keys_y (list): list of the timeseries keys for which indices are desired
            t_any (array): array of timepoints that the indices shall be relative to
        Returns:
            dict: maps each ykey in keys_y to t_bmask (boolean mask with same size as t_any)
        """
        t_bmask = {}
        t_any = self.t_any
        for yi, tskey in enumerate(keys_y):
            if tskey in self:
                t_bmask[tskey] = numpy.in1d(t_any, self[tskey].t)
            else:
                t_bmask[tskey] = numpy.repeat(False, len(t_any))
        return t_bmask

    def get_observation_indices(self, keys_y:list) -> dict:
        """Gets the index masks for observations of each y in [keys_y], relative to [t_any] and ts.x

        Args:
            keys_y (list): list of the timeseries keys for which indices are desired
            t_any (array): array of timepoints that the indices shall be relative to

        Returns:
            dict: maps each ykey in keys_y to x_imask (array of indices in t_any)
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
        if tmin == tmax:
            N = 1
        t = numpy.linspace(tmin, tmax, N)
        rep = Replicate(rid)
        for yk in independent_keys:
            rep[yk] = Timeseries(t, numpy.empty((N,)), independent_key=yk, dependent_key=yk)
        return rep
        
    def __str__(self):
        return f'Replicate({", ".join(map(str, self.values()))})'

    def __repr__(self):
        return self.__str__()


class Dataset(collections.OrderedDict):
    """A dataset contains one or more Replicates."""

    def __setitem__(self, key:str, value:Replicate):
        assert isinstance(value, Replicate)
        if not key == value.rid:
            raise KeyError(f'The key "{key}" did not match Replicate.rid "{value.rid}".')
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

    @staticmethod
    def make_template_like(dataset, independent_keys:typing.Iterable[str], *, N:int=200, tmin:typing.Optional[float]=None):
        """Create a dense template Dataset that has the same start and end times as another Dataset.

        Args:
            dataset (murefi.Dataset): a template dataset (typically with real observations)
            independent_keys (list): list of independent variable keys to include in the template
            N (int): total number of timepoints (default: 200)
            tmin (float, optional): override for the start time (when tmin=None, the first timepoint of the template replicate is used)
        
        Returns:
            dataset (Dataset): dataset object containing Replicates with dense timeseries of random y data
        """
        ds = Dataset()
        for rid, rep in dataset.items():
            ds[rid] = Replicate.make_template(
                tmin=rep.t_any[0] if tmin is None else tmin,
                tmax=rep.t_max,
                independent_keys=independent_keys,
                rid=rid, N=N
            )
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
    