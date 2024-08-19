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


def find_Cdnod(ts, Cd):
    L = ts.L.values
    ub2 = ts.ub2circ.values/L
    u = ts.ucirc.values/L 
    
    return Cd*ub2/(u*np.abs(u))


def find_Rnod(ts, Cd):
    L = ts.L.values
    ub2 = ts.ub2circ.values/L
    u = ts.ucirc.values/L
    
    return Cd*ub2/u



def plot_integrals(ts, dt=60*60*24, friction="quadratic", adjustDc = True, staticDc = True, dynamicDc = False, adjustR = False, R=5e-4, Cd = 0.003):
    color_wind = "cornflowerblue"
    color_nonlin = "darkorange"
    
    L = ts.L.values
    t = ts.ocean_time

    tau = ts.taucirc.values/L#(L*H*rho)
    nonlin = tau + ts.zflux.values/L
    u = ts.ucirc.values/L
    u0 = u[0]
    
    info = {}

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(t, u, color="black", label="simulations")
    
    ax.plot((np.nan), (np.nan), color=color_wind, label = "surface forcing")
    ax.plot((np.nan), (np.nan),color=color_nonlin, label = "surface forcing + vorticity flux")
    
    ax.set_xlabel("Time [year]")
    ax.set_ylabel("Normalized circulation [m s-1]")
    
    if friction == "linear" and not adjustR:
        uwind = integrate(tau, dt, u0, friction=friction, R=R)
        unonlin = integrate(nonlin, dt, u0, friction=friction, R=R)
        
        ax.plot(t, uwind, color=color_wind, lw=1)
        ax.plot(t, unonlin, color=color_nonlin, lw=1)
        
        ax.legend()
        
        info["r"] = np.corrcoef(unonlin, u)[0,1]
        
        return fig, ax, info
    
    if friction == "linear":
        Rnod = find_Rnod(ts, Cd)
    
        uwind = integrate(tau, dt, u0, friction=friction, R=np.nanmean(Rnod))
        unonlin = integrate(nonlin, dt, u0, friction=friction, R=np.nanmean(Rnod))
        
        ax.plot(t, uwind, color=color_wind, lw=1)
        ax.plot(t, unonlin, color=color_nonlin, lw=1)
        
        ax.legend()
        
        info["Rnod"] = np.nanmean(Rnod)
        info["r"] = np.corrcoef(unonlin, u)[0,1]
        
        return fig, ax, info
    
    
    if not adjustDc: 
        uwind = integrate(tau, dt, u0, friction=friction)
        unonlin = integrate(nonlin, dt, u0, friction=friction)
        
        ax.plot(t, uwind, color=color_wind, lw=1)
        ax.plot(t, unonlin, color=color_nonlin, lw=1)
        
        ax.legend()
        
        info["r"] = np.corrcoef(unonlin, u)[0,1]
        
        return fig, ax, info
    
    
    Cdnod = find_Cdnod(ts, Cd)
     
    ls_dyn = "solid"

    if staticDc:
        uwind_statCd = integrate(tau, dt, u0, friction=friction, Cd=np.nanmean(Cdnod))
        unonlin_statCd = integrate(nonlin, dt, u0, friction=friction, Cd=np.nanmean(Cdnod))
        
        ax.plot(t, uwind_statCd, color=color_wind, lw=1)
        ax.plot(t, unonlin_statCd, color=color_nonlin, lw=1)
        
        ls_dyn = "dashed"

    if dynamicDc:
        uwind_dynCd = integrate(tau, dt, u0, friction=friction, Cd=Cdnod)
        unonlin_dynCd = integrate(nonlin, dt, u0, friction=friction, Cd=Cdnod)
        
        ax.plot(t, uwind_dynCd, color=color_wind, lw=1, ls=ls_dyn)
        ax.plot(t, unonlin_dynCd, color=color_nonlin, lw=1, ls=ls_dyn)
        
    info["Cdnod"] = np.nanmean(Cdnod)
    info["r"] = np.corrcoef(unonlin_statCd, u)[0,1]
    
    ax.legend()
    
    return fig, ax, info