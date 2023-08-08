import base64
import hashlib
import logging
from typing import Any, Dict, Sequence, Tuple, Union

import calibr8
import numpy
from typing_extensions import TypeAlias

try:
    # PyTensor
    import pytensor as _backend
    import pytensor.tensor as pt
    from pytensor.graph.basic import Apply, Variable
    from pytensor.graph.op import Op
except ModuleNotFoundError:
    # PyTensor is not available
    _backend = calibr8.utils.ImportWarner("pytensor")
    pt = calibr8.utils.ImportWarner("pytensor")
    Op = object
    Apply: TypeAlias = Any
    Variable: TypeAlias = Any


try:
    import pymc as pm
except ModuleNotFoundError:
    pm = calibr8.utils.ImportWarner("pymc")


try:
    import sunode
    from sunode.wrappers import as_pytensor

    HAS_SUNODE = True
except ModuleNotFoundError:
    sunode = calibr8.utils.ImportWarner("sunode")
    HAS_SUNODE = False


logger = logging.getLogger(__name__)


def make_hash_sha256(obj):
    """Computes a sha256 hash for the object."""
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(obj)).encode())
    return base64.b64encode(hasher.digest()).decode()


def make_hashable(obj):
    """Makes tuples, lists, dicts, sets and frozensets hashable."""
    if isinstance(obj, (tuple, list)):
        return tuple((make_hashable(e) for e in obj))
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in obj))
    return obj


class IntegrationOp(Op):
    """This is a PyTensor Op that becomes a node in the computation graph.
    It is not differentiable, because it uses a 'solver' function that is provided by the user.
    """

    __props__ = ("solver", "keys_y")

    def __init__(self, solver, keys_y):
        self.solver = solver
        self.keys_y = keys_y
        super().__init__()

    def __hash__(self):
        subhashes = (hash(type(self)), make_hash_sha256(self.solver), make_hash_sha256(self.keys_y))
        return hash(subhashes)

    def make_node(self, y0: list, t, theta: list):
        # NOTE: PyTensor does not allow a list of tensors to be one of the inputs
        #       that's why they have to be pt.stack()ed which also merges them into one dtype!
        # TODO: check dtypes and raise warnings
        y0 = pt.stack([pt.as_tensor_variable(y) for y in y0])
        theta = pt.stack([pt.as_tensor_variable(var) for var in theta])
        t = pt.as_tensor_variable(t)
        apply_node = Apply(
            self, [y0, t, theta], [pt.dmatrix()]  # symbolic inputs: y0 and theta
        )  # symbolic outputs: Y_hat
        # NOTE: to support multiple different dtypes as transient variables, the
        #       output type would have to be a list of dvector/svectors.
        return apply_node

    def perform(self, node, inputs, output_storage):
        # this performs the actual simulation using the provided solver
        # which takes actual y0/t/theta values and returns a matrix
        y0, t, theta = inputs
        Y_hat = self.solver(y0, t, theta)  # solve for all x
        output_storage[0][0] = numpy.stack([Y_hat[ykey] for iy, ykey in enumerate(self.keys_y)])
        return

    def grad(self, inputs, outputs):
        return [
            _backend.gradient.grad_undefined(
                self, k, inp, "No gradient defined through Python-wrapping IntegrationOp."
            )
            for k, inp in enumerate(inputs)
        ]

    def infer_shape(self, fgraph, node, input_shapes):
        s_y0, s_x, s_theta = input_shapes
        output_shapes = [(s_y0[0], s_x[0])]
        return output_shapes


def named_with_shapes_dict(
    vars: Dict[str, Union[numpy.ndarray, Variable]], names: Sequence[str]
) -> Dict[str, Tuple[Union[numpy.ndarray, Variable], Tuple[int, ...]]]:
    d = {}
    for n, name in enumerate(names):
        v = vars[n]
        if calibr8.istensor(v):
            d[name] = (v, ())
        else:
            v = numpy.array(v).astype(_backend.config.floatX)
            d[name] = (v, numpy.shape(v))
    return d


def solve_sunode(
    dydt,
    independent_keys,
    y0,
    t,
    ode_parameters,
    parameter_names,
) -> dict:
    def dydt_dict(t, y, params):
        dy = dydt(
            [getattr(y, ikey) for ikey in independent_keys],
            t,
            [getattr(params, pkey) for pkey in parameter_names],
        )
        return {ikey: dy[i] for i, ikey in enumerate(independent_keys)}

    y0 = named_with_shapes_dict(y0, independent_keys)
    params = named_with_shapes_dict(ode_parameters, parameter_names)
    params["extra"] = numpy.zeros(1)
    solution, *_ = as_pytensor.solve_ivp(
        y0=y0,
        params=params,
        rhs=dydt_dict,
        tvals=t,
        t0=t[0],
        derivatives="forward",
        solver_kwargs=dict(sens_mode="simultaneous"),
    )
    return solution
