import xarray as xr
import numpy as np

from methods_time_integrals import find_Rnod, integrate

# Define the data path where datasets are stored
datapath = "path/to/data/"

# Time step for integration in seconds (1 day)
dt = 60 * 60 * 24

# Constants for oceanographic calculations
H = 3114       # Water depth in meters
rho = 1025     # Seawater density in kg/m^3
Cd = 0.003     # Drag coefficient

# File containing timeseries of circulations and fluxes
L800_file = "lofoten800_lofoten_h_50km_timeseries.nc" 

# Load the datasets using xarray
ts_L800 = xr.open_dataset(datapath + L800_file).sel(ocean_time=slice("01.01.1997", None))

# Adjusting surface stress by dividing by water depth and density
ts_L800["taucirc_area"] = ts_L800.taucirc_area / (H * rho)

# Extract and normalize diagnosed velocity circulation
L = ts_L800.L_line
C = ts_L800.ucirc_area / L
C0 = C.isel(ocean_time=0)  # Initial circulation value

# Extract and normalize forcing terms
f_wind = ts_L800.taucirc_area.values / L
f_zbarflux = -ts_L800.zbarflux_area.values / L
f_znodflux = -ts_L800.znodflux_area.values / L

# Initialize the forcing for the initial condition 
f_ini = np.zeros_like(f_wind)

# Calculate the R value
R = np.nanmean(find_Rnod(ts_L800, Cd))
s = np.nanstd(find_Rnod(ts_L800, Cd))

# Integrate each forcing term over time 
C_ini = integrate(f_ini, dt, C0, friction="linear", R=R)
C_wind = integrate(f_wind, dt, 0, friction="linear", R=R)
C_zbarflux = integrate(f_zbarflux, dt, 0, friction="linear", R=R)
C_znodflux = integrate(f_znodflux, dt, 0, friction="linear", R=R)

# Combine the results into an array for saving
terms = np.array([C_ini, C_wind, C_zbarflux, C_znodflux])

# Define the names of the estimated circulation terms 
Fnames = ["initial state", "surface stress", "barotropic vorticity", "baroclinic vorticity"]

# Create a new xarray Dataset to save the results
ds_out = xr.Dataset(
    data_vars=dict(
        C=("ocean_time", C.values),               # Diagnosed circulaiton
        estimates=(["term", "ocean_time"], terms),  # Estimated circulation
    ),
    coords=dict(
        ocean_time=C.ocean_time,                  # Time coordinates
        term=Fnames                               # Forcing term labels
    ),
    attrs=dict(
        r=R,  # Friction parameter
        L=L   # Length scale
    )
)

# Save the resulting dataset to a NetCDF file
ds_out.to_netcdf(datapath + "lofoten800_estimates.nc")