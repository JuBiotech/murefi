import numpy

def get_tau_sd(tau=None, sd=None):
    """Find precision and standard deviation (taken from pymc3)
    .. math::
        \tau = \frac{1}{\sigma^2}
    Parameters
    ----------
    tau : array-like, optional
    sd : array-like, optional
    Results
    -------
    Returns tuple (tau, sd)
    Notes
    -----
    If neither tau nor sd is provided, returns (1., 1.)
    """
    if tau is None:
        if sd is None:
            sd = 1.
            tau = 1.
        else:
            tau = sd**-2.

    else:
        if sd is not None:
            raise ValueError("Can't pass both tau and sd")
        else:
            sd = tau**-.5

    # cast tau and sd to float in a way that works for both np.arrays
    # and pure python
    tau = 1. * tau
    sd = 1. * sd

    return (tau, sd)


def logp_normal(mu, sd, x):
    """The logp of [x] in a normal distribution of [mu] and [sd]."""
    tau, sd = get_tau_sd(tau=None, sd=sd)
    return (-tau * (x - mu)**2 + numpy.log(tau / numpy.pi / 2.)) / 2.