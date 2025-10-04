import xarray as xr
import numpy as np
import glob

# Define the data path and parameters for contour processing
datapath = "path/to/data/"
contour = "lofoten"
ctype = "h"
scale = "50km"

def get_files(file_pattern):
    """
    Retrieve files matching the specified pattern and load them as xarray datasets.

    Parameters:
    file_pattern (str): The pattern to match files.

    Returns:
    list: A list of xarray.Dataset objects loaded from the matching files.
    """
    files = sorted(glob.glob(file_pattern))
    datasets = []
    for file in files:
        try:
            datasets.append(xr.open_dataset(file))
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    return datasets

# Retrieve the files for contour variables and surface stress circulation
files = get_files(f"{datapath}processed/contour_variables_{contour}_{ctype}_{scale}_ocean_avg_*.nc")
taufiles = get_files(f"{datapath}processed/surface_stress_circulation_{contour}_{ctype}_{scale}_ocean_avg_*.nc")

# Concatenate the datasets along the ocean_time dimension
ts = xr.concat(files, dim="ocean_time")
tau_ts = xr.concat(taufiles, dim="ocean_time")

# Create a new dataset to hold the combined data
ds = xr.Dataset()

# Combine variables from both datasets into a single dataset
for dss in [ts, tau_ts]:
    for var_name, values in dss.items():
        ds[var_name] = dss[var_name]

# Copy over attributes related to contour length and area
ds.attrs["L_area"] = ts.attrs["L_area"]
ds.attrs["L_line"] = ts.attrs["L_line"]

# Save the combined dataset to a NetCDF file
output_path = f"{datapath}lofoten800_{contour}_{ctype}_{scale}_timeseries.nc"
ds.to_netcdf(output_path)

print(f"Combined dataset saved to {output_path}")
