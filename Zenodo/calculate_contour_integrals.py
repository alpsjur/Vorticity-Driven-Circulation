import xgcm
import xarray as xr
import numpy as np
import sys
import os

# Define the data path (modify according to your actual path)
datapath = "path/to/data/"

if not os.path.exists(datapath+"processed/"):
    os.makedirs(datapath+"processed/")
    
if not os.path.exists(datapath+"temp/"):
    os.makedirs(datapath+"temp/")

# Option to resample data to daily means
daily_mean = True

# Set chunk sizes for Dask parallel processing
xchunk, ychunk, tchunk = -1, -1, 1
chunks = {
    "ocean_time": tchunk, 
    "xi_rho": xchunk, "xi_u": xchunk, 
    "eta_rho": ychunk, "eta_v": ychunk,
}

# Get the filename and contour name from command line arguments
filename = sys.argv[1]
contour = sys.argv[2]

# Function to calculate flux over an area
def calc_flux_area(u, v, grid, mask):
    # Compute derivatives of u and v
    dudx, dvdy = grid.derivative(u, 'X'), grid.derivative(v, 'Y')
    
    # Calculate divergence and apply mask
    udiv = (dudx + dvdy).where(mask)
    
    # Integrate over the masked area
    flux_area = grid.integrate(udiv, ['X', 'Y'])
    return flux_area

# Function to calculate flux along a line
def calc_flux_line(u, v, grid, C):
    # Interpolate u and v to rho points
    u_rho, v_rho = grid.interp(u, "X"), grid.interp(v, "Y")
    uc, vc = u_rho.interp(xi_rho=C.xi_rho, eta_rho=C.eta_rho), v_rho.interp(xi_rho=C.xi_rho, eta_rho=C.eta_rho)

    # Calculate line integrals
    flux_line = (uc * C.ty - vc * C.tx) * C.distance
    flux_line = flux_line.sum("point")
    return flux_line

# Function to calculate circulation over an area
def calc_circ_area(u, v, grid, mask):
    # Compute derivatives of u and v
    dudy, dvdx = grid.derivative(u, 'Y'), grid.derivative(v, 'X')
    
    # Calculate curl and apply mask
    ucurl = (grid.interp(dvdx, ['X', 'Y']) - grid.interp(dudy, ['X', 'Y'])).where(mask)
    
    # Integrate over the masked area
    circ_area = grid.integrate(ucurl, ['X', 'Y'])
    return circ_area

# Function to calculate circulation along a line
def calc_circ_line(u, v, grid, C):
    # Interpolate u and v to rho points
    u_rho, v_rho = grid.interp(u, "X"), grid.interp(v, "Y")
    uc, vc = u_rho.interp(xi_rho=C.xi_rho, eta_rho=C.eta_rho), v_rho.interp(xi_rho=C.xi_rho, eta_rho=C.eta_rho)
    
    # Calculate line integrals
    ucirc_line = (uc * C.tx + vc * C.ty) * C.distance
    ucirc_line = ucirc_line.sum("point")
    return ucirc_line

# Define paths to the various data files
ubfile = datapath + "extracted_fields/ub_" + filename[:-1]
vbfile = datapath + "extracted_fields/vb_" + filename[:-1]
ubarfile = datapath + "extracted_fields/ubar_" + filename[:-1]
vbarfile = datapath + "extracted_fields/vbar_" + filename[:-1]
zbarfile = datapath + "extracted_fields/zbar_" + filename[:-1]
zunodfile = datapath + "extracted_fields/zunod_" + filename[:-1]
zvnodfile = datapath + "extracted_fields/zvnod_" + filename[:-1]

# Load the grid metrics and extracted fields datasets
ds = xr.open_dataset(datapath + "extracted_fields/grid_metrics.nc")
ds["ub"] = xr.open_dataarray(ubfile)
ds["vb"] = xr.open_dataarray(vbfile)
ds["ubar"] = xr.open_dataarray(ubarfile)
ds["vbar"] = xr.open_dataarray(vbarfile)
ds["zbar"] = xr.open_dataarray(zbarfile)
ds["zunod"] = xr.open_dataarray(zunodfile)
ds["zvnod"] = xr.open_dataarray(zvnodfile)

# Resample data to daily means to remove tidal fluctuations, if specified
if daily_mean:
    ds = ds.resample(ocean_time="1d").mean()

# Chunk the dataset for efficient parallel processing
ds = ds.chunk(chunks)

# Define grid coordinates and metrics for xgcm
coords = {'X': {'center': 'xi_rho', 'inner': 'xi_u'}, 'Y': {'center': 'eta_rho', 'inner': 'eta_v'}}
metrics = {
    ('X',): ['dx', 'dx_u', 'dx_v', 'dx_psi'],  # X distances
    ('Y',): ['dy', 'dy_u', 'dy_v', 'dy_psi'],  # Y distances
    ('X', 'Y'): ['dA']  # Areas
}

# Create the xgcm grid object
grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=[])

# Load contour and mask datasets
C = xr.open_dataset(datapath+f"contours/{contour}_contour.nc")
mask = xr.open_dataarray(datapath+f"contours/{contour}_mask.nc")

# Calculate velocity magnitudes and squared velocities
speed = np.sqrt(ds.ub**2 + ds.vb**2)
ub2 = speed * ds.ub
vb2 = speed * ds.vb

# Extract mean velocities and vorticity
ubar = ds.ubar
vbar = ds.vbar
zeta = ds.zbar

# Calculate products of velocity and vorticity components
zunod = ds.zunod
zvnod = ds.zvnod
zubar = grid.interp(zeta, "Y") * ubar
zvbar = grid.interp(zeta, "X") * vbar

# Calculate circulation and fluxes over areas and along lines
ub2circ_area = calc_circ_area(ub2, vb2, grid, mask)
ub2circ_line = calc_circ_line(ub2, vb2, grid, C)
ucirc_area = calc_circ_area(ubar, vbar, grid, mask)
ucirc_line = calc_circ_line(ubar, vbar, grid, C)
zbarflux_area = calc_circ_area(zubar, zvbar, grid, mask)
zbarflux_line = calc_circ_line(zubar, zvbar, grid, C)
znodflux_area = calc_circ_area(zunod, zvnod, grid, mask)
znodflux_line = calc_circ_line(zunod, zvnod, grid, C)

# Divide data into 4 time segments and save each segment
n = len(ds.ocean_time) // 4
timesteps = np.array_split(ds.ocean_time.values, n)
for i, t in enumerate(timesteps):
    ds_segment = xr.Dataset()
    
    # Store calculated data for each time segment
    ds_segment["ub2circ_area"] = ub2circ_area.sel(ocean_time=t)
    ds_segment["ub2circ_line"] = ub2circ_line.sel(ocean_time=t)
    ds_segment["ucirc_area"] = ucirc_area.sel(ocean_time=t)
    ds_segment["ucirc_line"] = ucirc_line.sel(ocean_time=t)
    ds_segment["zbarflux_area"] = zbarflux_area.sel(ocean_time=t)
    ds_segment["zbarflux_line"] = zbarflux_line.sel(ocean_time=t)
    ds_segment["znodflux_area"] = znodflux_area.sel(ocean_time=t)
    ds_segment["znodflux_line"] = znodflux_line.sel(ocean_time=t)
    
    # Define output filename and save the segment to a NetCDF file
    outname = datapath + f"temp/contour_variables_{contour}_{filename[:-4]}_temp_{i:03}.nc"
    if os.path.exists(outname):
        os.remove(outname)
    ds_segment.to_netcdf(outname)
    ds_segment.close()

# Reopen all saved segments and merge into a single dataset
ds = xr.open_mfdataset([datapath + f"temp/contour_variables_{contour}_{filename[:-4]}_temp_{i:03}.nc" for i in range(n)])
ds.attrs["L_area"] = mask.L.values
ds.attrs["L_line"] = C.distance.sum("point").values

# Define the final output filename and save the merged dataset
if daily_mean:
    outname = datapath + f"processed/daily_contour_variables_{contour}_" + filename[:-1]
else:
    outname = datapath + f"processed/contour_variables_{contour}_" + filename[:-1]

if os.path.exists(outname):
    os.remove(outname)
ds.to_netcdf(outname)

# Close all datasets
ds.close()
C.close()
mask.close()

# Clean up temporary files
for i in range(n):
    os.remove(datapath + f"temp/contour_variables_{contour}_{filename[:-4]}_temp_{i:03}.nc")
