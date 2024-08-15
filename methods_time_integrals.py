import xarray as xr
import numpy as np

def quadratic_friction(C, H=3114, Cd=0.003, R=None):
    """
    Computes quadratic friction for a given value or array `C`.

    Parameters:
    C (float or array-like): The input value(s) for which to calculate friction.
    Cd (float, optional): Drag coefficient. Default is 0.003.

    Returns:
    float or numpy.ndarray: The computed quadratic friction.
    """
    return Cd * C * np.abs(C) / H 

def linear_friction(C, H=3114, Cd = None, R=5e-4):
    """
    Computes linear friction for a given value or array `C`.

    Parameters:
    C (float or array-like): The input value(s) for which to calculate friction.
    R (float, optional): Resistance coefficient. Default is 5e-4.

    Returns:
    float or numpy.ndarray: The computed linear friction.
    """
    return R * C / H 


def integrate(c, dt, C0, friction=None, Cd=None, R=None):
    """
    Integrates the given input series `c` over time step `dt`, starting from the initial condition `C0`.
    
    Parameters:
    c (array-like): Input series to be integrated.
    dt (float): Time step for integration.
    C0 (float): Initial condition for the integration.
    friction (str, optional): Type of friction to apply. Can be 'linear' or 'quadratic'.
                              If None, no friction is applied.
    **kwargs: Additional arguments to pass to the friction function.

    Returns:
    numpy.ndarray: The integrated series, with optional friction applied.

    Raises:
    ValueError: If an unsupported friction type is provided.
    """
    if Cd is not None and np.isscalar(Cd):
        Cd = np.full_like(c, Cd)
    
    # No friction case: simple cumulative sum with initial condition
    if friction is None:
        C = np.cumsum(c * dt) + C0
        return C
    
    # Initialize the result array with the same shape as input and fill with NaNs
    C = np.zeros_like(c) 
    C[0] = C0  # Set the initial condition

    # Select the appropriate friction function based on the input
    if friction == "linear":
        friction_function = linear_friction
    elif friction == "quadratic":
        friction_function = quadratic_friction
    else:
        raise ValueError("Unsupported friction type. Use 'linear' or 'quadratic'.")
   

    # Perform the integration with the chosen friction model
    for i in range(len(c) - 1):
        ci = c[i]
        Cdi = Cd[i] if Cd is not None else None
        
        # Apply friction to the current step, using the corresponding value of Cd if available
        if Cdi is not None:
            fi = ci - friction_function(C[i], Cd=Cdi, R=R)
        else:
            fi = ci - friction_function(C[i], R=R)
        
        # Update the next value based on the friction-adjusted input
        C[i + 1] = C[i] + dt * fi
        
    return C