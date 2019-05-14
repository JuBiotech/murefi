
from .. ode import BaseODEModel

class MonodModel(BaseODEModel):
    """ Class specifying the model for parameter fitting as Monod kinetics. """
    def dydt(self, y, t, theta):
        """First derivative of the transient variables.
        Args:
            y (array): array of observables
            t (float): time since intial state
            theta (array): Monod parameters
        Returns:
            array: change in y at time t
        """
        # NOTE: this method has significant performance impact!
        S, X = y
        mu_max, K_S, Y_XS = theta
        dXdt = mu_max * S * X / (K_S + S)
    
        yprime = [
            -1/Y_XS * dXdt,
            dXdt,
        ]
        return yprime