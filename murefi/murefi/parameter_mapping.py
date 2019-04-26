import collections
import numpy
import pandas


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
            pname : pvalue
            for pname, pvalue in zip(self.parameters, theta_full)
        }
        theta_dict = {
            rkey : numpy.array([
                pname_to_pvalue[pname] if isinstance(pname, str) else pname
                for pname in pnames
            ], dtype=float)
            for rkey, pnames in self.mapping.items()
        }
        return theta_dict
