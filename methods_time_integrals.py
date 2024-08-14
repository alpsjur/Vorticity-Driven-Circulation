import xarray as xr
import numpy as np

def quadratic_friction(C, Cd=0.003):
    """
    Computes quadratic friction for a given value or array `C`.

    Parameters:
    C (float or array-like): The input value(s) for which to calculate friction.
    Cd (float, optional): Drag coefficient. Default is 0.003.

    Returns:
    float or numpy.ndarray: The computed quadratic friction.
    """
    return Cd * C * np.abs(C)  # Quadratic friction: Cd * C * |C|

def linear_friction(C, R=5e-4):
    """
    Computes linear friction for a given value or array `C`.

    Parameters:
    C (float or array-like): The input value(s) for which to calculate friction.
    R (float, optional): Resistance coefficient. Default is 5e-4.

    Returns:
    float or numpy.ndarray: The computed linear friction.
    """
    return R * C  # Linear friction: R * C

def integrate(c, dt, C0, friction=None):
    """
    Integrates the given input series `c` over time step `dt`, starting from initial condition `C0`.
    
    Parameters:
    c (array-like): Input series to be integrated.
    dt (float): Time step for integration.
    C0 (float): Initial condition for the integration.
    friction (str, optional): Type of friction to apply. Can be 'linear' or 'quadratic'.
                              If None, no friction is applied.

    Returns:
    numpy.ndarray: The integrated series, with optional friction applied.
    """
    # No friction case: simple cumulative sum with initial condition
    if friction is None:
        C = np.cumsum(c * dt) + C0
        return C
    
    # Initialize the result array with NaNs
    C = np.zeros_like(c) * np.nan
    C[0] = C0  # Set the initial condition

    # Select the appropriate friction function based on the input
    if friction == "linear":
        friction_function = linear_friction
    elif friction == "quadratic":
        friction_function = quadratic_friction
    else:
        raise ValueError("Unsupported friction type. Use 'linear' or 'quadratic'.")

    # Perform the integration with the chosen friction model
    for i, ci in enumerate(c[:-1]):
        fi = ci - friction_function(C[i])  # Apply friction to the current step
        C[i + 1] = C[i] + dt * fi  # Update the next value based on the friction-adjusted input
        
    return C