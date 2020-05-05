import logging
import numpy
import pandas
import base64
import hashlib


HAVE_PYMC3 = False

try:
    import pymc3
    HAVE_PYMC3 = True
except ModuleNotFoundError:  # pymc3 is optional, throw exception when used
    class _ImportWarnerPyMC3:
        __all__ = []

        def __init__(self, attr):
            self.attr = attr

        def __call__(self, *args, **kwargs):
            raise ImportError(
                "PyMC3 is not installed. In order to use this function:\npip install pymc3"
            )

    class _PyMC3:
        def __getattr__(self, attr):
            return _ImportWarnerPyMC3(attr)
    
    pm = _PyMC3()

try:
    import theano
except ModuleNotFoundError:  # theano is optional, throw exception when used

    class _ImportWarnerTheano:
        __all__ = []

        def __init__(self, attr):
            self.attr = attr

        def __call__(self, *args, **kwargs):
            raise ImportError(
                "Theano is not installed. In order to use this function:\npip install theano"
            )

    class _Theano:
        def __getattr__(self, attr):
            return _ImportWarnerTheano(attr)
    
    theano = _Theano()

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


class IntegrationOp(theano.Op if HAVE_PYMC3 else object):
    """This is a theano Op that becomes a node in the computation graph.
    It is not differentiable, because it uses a 'solver' function that is provided by the user.
    """
    __props__ = ('solver', 'keys_y')

    def __init__(self, solver, keys_y):
        self.solver = solver
        self.keys_y = keys_y
        super().__init__()

    def __hash__(self):
        subhashes = (
            hash(type(self)),
            make_hash_sha256(self.solver),
            make_hash_sha256(self.keys_y)
        )
        return hash(subhashes)

    def make_node(self, y0:list, t, theta:list):
        # NOTE: theano does not allow a list of tensors to be one of the inputs
        #       that's why they have to be theano.tensor.stack()ed which also merges them into one dtype!
        # TODO: check dtypes and raise warnings
        y0 = theano.tensor.stack([theano.tensor.as_tensor_variable(y) for y in y0])
        theta = theano.tensor.stack([theano.tensor.as_tensor_variable(var) for var in theta])
        t = theano.tensor.as_tensor_variable(t)
        apply_node = theano.Apply(self,
                            [y0, t, theta],     # symbolic inputs: y0 and theta
                            [theano.tensor.dmatrix()])     # symbolic outputs: Y_hat
        # NOTE: to support multiple different dtypes as transient variables, the
        #       output type would have to be a list of dvector/svectors.
        return apply_node

    def perform(self, node, inputs, output_storage):
        # this performs the actual simulation using the provided solver
        # which takes actual y0/t/theta values and returns a matrix
        y0, t, theta = inputs
        Y_hat = self.solver(y0, t, theta)       # solve for all x
        output_storage[0][0] = numpy.stack([
            Y_hat[ykey]
            for iy, ykey in enumerate(self.keys_y)
        ])
        return

    def grad(self, inputs, outputs):
        return [theano.gradient.grad_undefined(self, k, inp,
                        'No gradient defined through Python-wrapping IntegrationOp.')
                for k, inp in enumerate(inputs)]

    def infer_shape(self, node, input_shapes):
        s_y0, s_x, s_theta = input_shapes
        output_shapes = [
            (s_y0[0],s_x[0])
        ]
        return output_shapes