import xarray as xr
import numpy as np

from methods_time_integrals import find_Rnod, integrate

# Define the data path where datasets are stored
datapath = "data/"

# Time step for integration in seconds (1 day)
dt = 60 * 60 * 24

# Constants for oceanographic calculations
H = 3114       # Water depth in meters
rho = 1025     # Seawater density in kg/m^3
Cd = 0.003     # Drag coefficient

# Define the filenames for the datasets
A4_file = "A4/A4_lofoten_h_50km_timeseries.nc" 
L800_file = "lofoten800/lofoten800_lofoten_h_50km_timeseries.nc" 

# Load the datasets using xarray
ts_A4 = xr.open_dataset(datapath + A4_file)
ts_L800 = xr.open_dataset(datapath + L800_file).sel(ocean_time=slice("01.01.1997", None))

# Adjusting surface stress by dividing by water depth and density
ts_A4["taucirc_area"] = ts_A4.taucirc_area / (H * rho)
ts_L800["taucirc_area"] = ts_L800.taucirc_area / (H * rho)

# Combine the barotropic and baroclinic vorticity flux terms into a single term
ts_L800["zflux_area"] = ts_L800.zbarflux_area + ts_L800.znodflux_area

# Interpolate missing values in the circulation area velocity
# The method "linear" fills in missing data linearly
ts_L800["ucirc_area"] = ts_L800["ucirc_area"].interpolate_na(dim="ocean_time", method="linear")

# Calculate the length scale and circulation
L = ts_L800.L_line
C = ts_L800.ucirc_area / L
C0 = C.isel(ocean_time=0)  # Initial circulation value

# Extract forcing terms
f_wind = ts_L800.taucirc_area.values / L
f_zbarflux = ts_L800.zbarflux_area.values / L
f_znodflux = ts_L800.znodflux_area.values / L

# Initialize the forcing for the initial condition to zero
f_ini = np.zeros_like(f_wind)

# Calculate the Rnod value using a custom function
R = np.nanmean(find_Rnod(ts_L800, Cd))
s = np.nanstd(find_Rnod(ts_L800, Cd))

#R = R-s

# Integrate each forcing term over time
C_ini = integrate(f_ini, dt, C0, friction="linear", R=R)
C_wind = integrate(f_wind, dt, 0, friction="linear", R=R)
C_zbarflux = integrate(f_zbarflux, dt, 0, friction="linear", R=R)
C_znodflux = integrate(f_znodflux, dt, 0, friction="linear", R=R)

# Combine the results into an array for saving
terms = np.array([C_ini, C_wind, C_zbarflux, C_znodflux])

# Define the names of the terms for clarity
Fnames = ["initial state", "surface stress", "barotropic vorticity", "baroclinic vorticity"]

# Create a new xarray Dataset to save the results
ds_out = xr.Dataset(
    data_vars=dict(
        C=("ocean_time", C.values),  # Main circulation result
        forcing=(["term", "ocean_time"], terms),  # Forcing terms
    ),
    coords=dict(
        ocean_time=C.ocean_time,  # Time coordinates
        term=Fnames  # Forcing term labels
    ),
    attrs=dict(
        r=R,  # Friction parameter
        L=L   # Length scale
    )
)

# Save the resulting dataset to a NetCDF file
ds_out.to_netcdf(datapath + "lofoten800/lofoten800_lofoten_h_50km_time_integrals.nc")