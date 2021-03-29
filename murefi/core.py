import collections
import logging
import numpy
import pandas
import typing
import warnings

import calibr8

from . datastructures import ShapeError


logger = logging.getLogger(__name__)


class ParameterMapping(object):
    @property
    def order(self) -> tuple:
        """Names of the model parameters"""
        return self._order
        
    @property
    def parameters(self) -> collections.OrderedDict:
        """Maps unique parameters to the names of the corresponding model parameters."""
        return self._parameters

    @property
    def theta_names(self) -> typing.Tuple[str]:
        """Names of unique parameters in the mapping"""
        return tuple(self.parameters.keys())

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

    @property
    def coords(self) -> typing.Dict[str, typing.Tuple[str]]:
        """ Groups the unique parameter ids by the kind of parameter.

        Keys are in the form f"{pkind}_dim" to avoid conflicting with
        random variable names (see https://github.com/arviz-devs/arviz/issues/1642).

        This dictionary can be used with pymc3.Model(coords=coords) to ease creation
        of vector-shaped priors.
        """
        raw_coords = {
            pkind : []
            for pkind in self.order
        }
        for pname, pkind in self.parameters.items():
            raw_coords[pkind].append(pname)
        coords = {
            f"{pkind}_dim" : tuple(pnames)
            for pkind, pnames in raw_coords.items()
            if len(pnames) > 0
        }
        return coords

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
        if not mapping.index.name == "rid":
            warnings.warn(
                f"The index of the mapping DataFrame should be named 'rid' but was '{mapping.index.name}'.",
                UserWarning,
            )

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
        super().__init__()

    def merge_vectors(self, parameter_vectors: typing.Dict[str, typing.Sequence]) -> tuple:
        """ Creates a full length (ndim,) parameter vector from a dictionary of
        parameter vectors.

        Arguments
        ---------
        parameter_vectors : dict
            dictionary that maps parameter categories (keys in .coords) to a sequence of
            elements corresponding to the unique parameters in the same order as the values
            in the .coords property

        Returns
        -------
        full_vec : tuple
            the elements from the input parameter vectors in the order
            specified by .parameters
        """
        coords = self.coords
        full_vec = tuple(
            (
                parameter_vectors[pkind][coords[f"{pkind}_dim"].index(pname)]
                if pkind in parameter_vectors else
                parameter_vectors[f"{pkind}_dim"][coords[f"{pkind}_dim"].index(pname)]
            )
            for pname, pkind in self.parameters.items()
        )
        return full_vec

    def as_dataframe(self) -> pandas.DataFrame:
        """ Re-creates the DataFrame representation of this parameter mapping.

        It is NOT the identical DataFrame object it was initialized from!
        """
        df_mapping = pandas.DataFrame.from_dict(self.mapping, orient="index")
        df_mapping.index.name = "rid"
        df_mapping.columns = self.order
        return df_mapping

    def repmap(self, theta_full:typing.Union[typing.Sequence, dict]) -> typing.Dict[str, typing.Sequence]:
        """Remaps a full parameter vector to a dictionary of replicate-wise parameters.

        Args:
            theta_full (array-like, dict): full parameter dict, vector or matrix
                when dict:
                    keys are the unique parameter names (see ParameterMapping.parameters)
                    values are float or Tensors, or numpy.ndarray (N_parametersets,)
                when vector:
                    (N_parameters,) tuple, list, Tensor or numpy.ndarray with elements being scalar (float/Tensor)
                when matrix:
                    (N_parameters, N_parametersets) numpy.ndarray

        Returns:
            theta_dict (dict): dictionary of replicate-wise parameter vectors/matrices
        """
        # prepare a dictionary that maps parameter names to values
        pname_to_pvalue = {}
        if isinstance(theta_full, dict):
            missing_parameters = set(self.parameters.keys()).difference(set(theta_full.keys()))
            if missing_parameters:
                raise KeyError(f'Parameters {missing_parameters} are missing from [theta_full].')
            pname_to_pvalue = theta_full
        else:
            if not len(theta_full) == len(self.parameters):
                raise ShapeError('[theta_full] does not match with the parameter mapping.', actual=(len(theta_full),), expected=(len(self.parameters),))
            for p, pname in enumerate(self.parameters):
                pname_to_pvalue[pname] = theta_full[p]

        # iterate over the mapping to collect replicate-wise parameters
        theta_dict = {
            rkey : tuple([
                pname_to_pvalue[pname] if isinstance(pname, str) else pname
                for pname in pnames
            ])
            for rkey, pnames in self.mapping.items()
        }
        return theta_dict

    def __repr__(self) -> str:
        return f"ParameterMapping({len(self.mapping)} replicates, {len(self.order)} inputs, {self.ndim} free parameters)"
