import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

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

def create_Ab(c, C0, R, H, dt):
    """
    Creates matrices A and b for a system based on the time derivative of circulation data,
    a linear friction parameter, and time intervals.

    Parameters:
    c (array-like): Input array containing the time derivative of circulation (may contain NaNs).
    C0 (float): Initial circulation value.
    R (float): Linear friction parameter.
    H (float): Depth parameter.
    dt (float): Time interval between measurements.

    Returns:
    tuple: A tuple containing:
        - A (ndarray): Matrix representing the system's filtering model affected by linear friction.
        - b (ndarray): Vector representing the decaying contribution of the initial circulaiton.
        - valid_indices (ndarray): Boolean array indicating valid (non-NaN) entries in c.
    """
    # Identify indices where c is not NaN
    valid_indices = ~np.isnan(c)
    c_valid = c[valid_indices]  # Time derivative of circulation excluding NaNs
    n = len(c_valid)  # Number of valid entries

    # Adjust dt based on the number of consecutive NaNs
    adjusted_dt = np.zeros_like(c)
    dt_counter = 0

    for i in range(len(c)):
        if np.isnan(c[i]):
            dt_counter += 1  # Count consecutive NaNs
        else:
            adjusted_dt[i] = dt * (dt_counter + 1)  # Adjust dt for valid entries
            dt_counter = 0  # Reset counter after encountering a valid entry

    # Remove NaNs from adjusted_dt
    adjusted_dt = adjusted_dt[valid_indices]

    # Construct vector b 
    decay_factors = np.exp(-R * adjusted_dt / H)
    cumulative_decay = np.cumprod(np.ones(n) * decay_factors) / decay_factors
    b = cumulative_decay * C0

    # Construct matrix A 
    A = np.ones((n, n)) * decay_factors
    A = np.triu(A)  # Upper triangular part filled with decay factors
    np.fill_diagonal(A, 1)  # Set diagonal to 1

    # Update matrix A to reflect filtering effect
    for i in range(n):
        A[i, i:] = np.cumprod(A[i, i:]) * adjusted_dt[i]

    # Correct the first entry of A to ensure it is initialized properly
    A[0, 0] *= 0 

    return A, b, valid_indices

def integrating_factor(c, dt, C0, R, H):
    """
    Computes the circulation values using the integrating factor method.

    Parameters:
    c (array-like): Input array containing the time derivative of circulation (may contain NaNs).
    dt (float): Time interval between measurements.
    C0 (float): Initial circulation value.
    R (float): Linear friction parameter.
    H (float): A parameter related to the system (e.g., volume or mass).

    Returns:
    ndarray: An array representing the computed circulation values, with NaNs preserved
             where the input array `c` had NaNs.
    """
    # Generate the matrix A and vector b using the create_Ab function
    A, b, valid_indices = create_Ab(c, C0, R, H, dt)
    
    # Initialize the output array C with NaN values
    C = np.full_like(c, np.nan)  

    # Extract valid entries of c (where c is not NaN)
    c_valid = c[valid_indices]

    # Compute circulation values for valid entries using matrix multiplication
    C[valid_indices] = np.matmul(c_valid, A) + b

    return C


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
        
        # Apply friction to the current step, using the corresponding value of Cd if available
        fi = ci - friction_function(C[i], Cd=Cd, R=R)
        
        # Update the next value based on the friction-adjusted input
        C[i + 1] = C[i] + dt * fi
        
    return C


def find_Cdnod(ts, Cd):
    """
    Computes the non-dimensional drag coefficient for quadratic friction.

    Parameters:
    ts (xarray.Dataset): Dataset containing circulation data.
    Cd (float): Drag coefficient.

    Returns:
    numpy.ndarray: Computed non-dimensional drag coefficient.
    """
    L = ts.L_line
    ub2 = ts.ub2circ_area.values / L
    u = ts.ucirc_area.values / L 
    
    return Cd * ub2 / (u * np.abs(u))

def find_Rnod(ts, Cd):
    """
    Computes the friction coefficient for linear friction.

    Parameters:
    ts (xarray.Dataset): Dataset containing circulation data.
    Cd (float): Drag coefficient.

    Returns:
    numpy.ndarray: Computed friction coefficient.
    """
    ub2 = ts.ub2circ_area.values
    u = ts.ucirc_area.values
    
    return Cd * ub2 / u
