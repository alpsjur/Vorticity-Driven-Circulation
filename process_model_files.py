import xarray as xr
import xgcm
import numpy as np
import os
import sys

# Option to save grid metrics
save_grid_metrics = False

# Define the path to the dataset (update with actual path)
# data available from https://thredds.met.no/thredds/catalog/romshindcast/lofoten800_2017/catalog.html
datapath = "path/to/data/"

if not os.path.exists(datapath+"extracted_fields/"):
    os.makedirs(datapath+"extracted_fields/")
    
if not os.path.exists(datapath+"temp/"):
    os.makedirs(datapath+"temp/")

# Chunk sizes for parallel processing (using Dask)
xchunk = -1
ychunk = -1
schunk = -1
tchunk = 1

# Dictionary defining the chunks for different dimensions
chunks = {
    "ocean_time": tchunk, 
    "xi_rho": xchunk, "xi_u": xchunk, 
    "eta_rho": ychunk, "eta_v": ychunk, 
    "s_rho": schunk, "s_w": schunk,
}

# Get the filename from the command line argument
filename = sys.argv[1]

print(f"Extracting fields from {filename}")

# Open the dataset using xarray with the specified chunks
ds = xr.open_dataset(datapath + filename)
ds = ds.chunk({"ocean_time": 1})  # Chunk only the time dimension initially

# Rename dimensions to match expected naming conventions for xgcm
ds = ds.rename({'eta_u': 'eta_rho', 'xi_v': 'xi_rho', 'xi_psi': 'xi_u', 'eta_psi': 'eta_v'})

# Define the coordinates dictionary for xgcm grid object
coords = {
    'X': {'center': 'xi_rho', 'inner': 'xi_u'}, 
    'Y': {'center': 'eta_rho', 'inner': 'eta_v'}, 
    'Z': {'center': 's_rho', 'outer': 's_w'}
}

# Create the grid object using xgcm
grid = xgcm.Grid(ds, coords=coords, periodic=[])

# Calculate the vertical coordinates (z-levels) at rho and w points
Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
z_rho = Zo_rho * (ds.zeta + ds.h) + ds.zeta

Zo_w = (ds.hc * ds.s_w + ds.Cs_w * ds.h) / (ds.hc + ds.h)
z_w = Zo_w * (ds.zeta + ds.h) + ds.zeta

# Add calculated z-coordinates to dataset
ds.coords['z_w'] = z_w.where(ds.mask_rho, 0).transpose('ocean_time', 's_w', 'eta_rho', 'xi_rho')
ds.coords['z_rho'] = z_rho.where(ds.mask_rho, 0).transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')

# Interpolate grid metrics to u, v, and psi points
ds['pm_v'] = grid.interp(ds.pm, 'Y')
ds['pn_u'] = grid.interp(ds.pn, 'X')
ds['pm_u'] = grid.interp(ds.pm, 'X')
ds['pn_v'] = grid.interp(ds.pn, 'Y')
ds['pm_psi'] = grid.interp(grid.interp(ds.pm, 'Y'), 'X')  # Interpolated to psi points
ds['pn_psi'] = grid.interp(grid.interp(ds.pn, 'X'), 'Y')  # Interpolated to psi points

# Calculate grid spacings (dx, dy) at various grid points
ds['dx'] = 1 / ds.pm
ds['dx_u'] = 1 / ds.pm_u
ds['dx_v'] = 1 / ds.pm_v
ds['dx_psi'] = 1 / ds.pm_psi

ds['dy'] = 1 / ds.pn
ds['dy_u'] = 1 / ds.pn_u
ds['dy_v'] = 1 / ds.pn_v
ds['dy_psi'] = 1 / ds.pn_psi

# Calculate vertical grid spacing differences
ds['dz'] = grid.diff(ds.z_w, 'Z', boundary='fill')
ds['dz_w'] = grid.diff(ds.z_rho, 'Z', boundary='fill')
ds['dz_u'] = grid.interp(ds.dz, 'X')
ds['dz_w_u'] = grid.interp(ds.dz_w, 'X')
ds['dz_v'] = grid.interp(ds.dz, 'Y')
ds['dz_w_v'] = grid.interp(ds.dz_w, 'Y')

# Calculate grid cell areas
ds['dA'] = ds.dx * ds.dy

# Define metrics for xgcm grid object
metrics = {
    ('X',): ['dx', 'dx_u', 'dx_v', 'dx_psi'],  # X distances
    ('Y',): ['dy', 'dy_u', 'dy_v', 'dy_psi'],  # Y distances
    ('Z',): ['dz', 'dz_u', 'dz_v', 'dz_w', 'dz_w_u', 'dz_w_v'],  # Z distances
    ('X', 'Y'): ['dA']  # Areas
}

# Re-create the grid object with the new metrics
grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=[])

# Save grid metrics to file if enabled
if save_grid_metrics:
    grid_vars = xr.Dataset()
    vars = ["dx", "dx_u", "dx_v", "dx_psi", "dy", "dy_u", "dy_v", "dy_psi", "dA", "h"]
    for var in vars:
        grid_vars[var] = ds[var].load()
    grid_vars.to_netcdf("extracted_fields/grid_metrics.nc")

# Re-chunk the dataset for efficient parallel processing
ds = ds.chunk(chunks)

# Extract velocity fields
u = ds.u
v = ds.v

# Select bottom velocity
ub = u.isel(s_rho=0)
vb = v.isel(s_rho=0)

# Calculate vertically averaged velocities
ubar = grid.average(u.fillna(0), ["Z"])
vbar = grid.average(v.fillna(0), ["Z"])

# Calculate horizontal gradients of velocity components
dudy = grid.derivative(u, "Y")
dvdx = grid.derivative(v, "X")

# Calculate relative vorticity (zeta)
zeta = -dudy + dvdx  

# Calculate vertically averaged vorticity
zetabar = grid.average(zeta.fillna(0), ["Z"])

# Calculate crossterm
zunod = grid.interp(zeta - zetabar, "Y") * (u - ubar)
zvnod = grid.interp(zeta - zetabar, "X") * (v - vbar)

# Vertically average the crossterm
zunod = grid.average(zunod.fillna(0), ["Z"])
zvnod = grid.average(zvnod.fillna(0), ["Z"])

# Convert to single-precision float to save memory
ub = ub.astype(np.float32)
vb = vb.astype(np.float32)
ubar = ubar.astype(np.float32)
vbar = vbar.astype(np.float32)
zetabar = zetabar.astype(np.float32)
zunod = zunod.astype(np.float32)
zvnod = zvnod.astype(np.float32)

# Prepare to save data for each day
n = len(ds.ocean_time) // 4
nstart = 0
timesteps = np.array_split(vbar.ocean_time.values, n)
timesteps = timesteps[nstart:]

# Loop through each time segment and save the processed data to temporary files
for i, t in enumerate(timesteps):
    i += nstart
    print(f" {i+1:>10}/{n}")
    _ub = ub.sel(ocean_time=t)
    _vb = vb.sel(ocean_time=t)
    _ubar = ubar.sel(ocean_time=t)
    _vbar = vbar.sel(ocean_time=t)
    _z = zetabar.sel(ocean_time=t)
    _zu = zunod.sel(ocean_time=t)
    _zv = zvnod.sel(ocean_time=t)

    # Define output file paths
    uboutname = datapath + f"temp/ub_{filename[:-4]}_temp_{i:03}.nc"
    vboutname = datapath + f"temp/vb_{filename[:-4]}_temp_{i:03}.nc"
    uoutname = datapath + f"temp/ubar_{filename[:-4]}_temp_{i:03}.nc"
    voutname = datapath + f"temp/vbar_{filename[:-4]}_temp_{i:03}.nc"
    zoutname = datapath + f"temp/zbar_{filename[:-4]}_temp_{i:03}.nc"
    zuoutname = datapath + f"temp/zunod_{filename[:-4]}_temp_{i:03}.nc"
    zvoutname = datapath + f"temp/zvnod_{filename[:-4]}_temp_{i:03}.nc"

    # Remove existing files to avoid conflicts
    for fname in [uboutname, vboutname, uoutname, voutname, zoutname, zuoutname, zvoutname]:
        if os.path.exists(fname):
            os.remove(fname)

    # Save data to NetCDF files
    _ub.to_netcdf(uboutname)
    _vb.to_netcdf(vboutname)
    _ubar.to_netcdf(uoutname)
    _vbar.to_netcdf(voutname)
    _z.to_netcdf(zoutname)
    _zu.to_netcdf(zuoutname)
    _zv.to_netcdf(zvoutname)

# Reload the saved files and merge them
ub = xr.open_mfdataset([datapath + f"temp/ub_{filename[:-4]}_temp_{i:03}.nc" for i in range(n)])
vb = xr.open_mfdataset([datapath + f"temp/vb_{filename[:-4]}_temp_{i:03}.nc" for i in range(n)])
ubar = xr.open_mfdataset([datapath + f"temp/ubar_{filename[:-4]}_temp_{i:03}.nc" for i in range(n)])
vbar = xr.open_mfdataset([datapath + f"temp/vbar_{filename[:-4]}_temp_{i:03}.nc" for i in range(n)])
zbar = xr.open_mfdataset([datapath + f"temp/zbar_{filename[:-4]}_temp_{i:03}.nc" for i in range(n)])
zunod = xr.open_mfdataset([datapath + f"temp/zunod_{filename[:-4]}_temp_{i:03}.nc" for i in range(n)])
zvnod = xr.open_mfdataset([datapath + f"temp/zvnod_{filename[:-4]}_temp_{i:03}.nc" for i in range(n)])

# Define final output file paths
uboutname = datapath + "extracted_fields/ub_" + filename[:-1]
vboutname = datapath + "extracted_fields/vb_" + filename[:-1]
uoutname = datapath + "extracted_fields/ubar_" + filename[:-1]
voutname = datapath + "extracted_fields/vbar_" + filename[:-1]
zoutname = datapath + "extracted_fields/zbar_" + filename[:-1]
zuoutname = datapath + "extracted_fields/zunod_" + filename[:-1]
zvoutname = datapath + "extracted_fields/zvnod_" + filename[:-1]

# Remove existing files to avoid conflicts
for fname in [uboutname, vboutname, uoutname, voutname, zoutname, zuoutname, zvoutname]:
    if os.path.exists(fname):
        os.remove(fname)

# Save the final merged data to NetCDF files
ub.to_netcdf(uboutname)
vb.to_netcdf(vboutname)
ubar.to_netcdf(uoutname)
vbar.to_netcdf(voutname)
zbar.to_netcdf(zoutname)
zunod.to_netcdf(zuoutname)
zvnod.to_netcdf(zvoutname)

# Close the opened datasets
ub.close()
vb.close()
ubar.close()
vbar.close()
zbar.close()
zunod.close()
zvnod.close()
ds.close()

# Remove temporary files to clean up
for i in range(n):
    os.remove(datapath + f"temp/ub_{filename[:-4]}_temp_{i:03}.nc")
    os.remove(datapath + f"temp/vb_{filename[:-4]}_temp_{i:03}.nc")
    os.remove(datapath + f"temp/ubar_{filename[:-4]}_temp_{i:03}.nc")
    os.remove(datapath + f"temp/vbar_{filename[:-4]}_temp_{i:03}.nc")
    os.remove(datapath + f"temp/zbar_{filename[:-4]}_temp_{i:03}.nc")
    os.remove(datapath + f"temp/zunod_{filename[:-4]}_temp_{i:03}.nc")
    os.remove(datapath + f"temp/zvnod_{filename[:-4]}_temp_{i:03}.nc")
