import xgcm
import xarray as xr
import numpy as np
import sys
import os
import warnings

# Suppress warnings to avoid cluttering the output
warnings.filterwarnings("ignore")


# Define local data path
datapath = "path/to/data/"

if not os.path.exists(datapath+"processed/"):
    os.makedirs(datapath+"processed/")
    
if not os.path.exists(datapath+"temp/"):
    os.makedirs(datapath+"temp/")


# Define constant for water density
rho0 = 1025  # kg/m³

# Define file path to the remote dataset 
# Surface stress can be read in using OPeNDAP
filepath = "https://thredds.met.no/thredds/dodsC/romshindcast/lofoten800_2016/"

# Get the filename and contour name from command line arguments
filename = sys.argv[1]
contour = sys.argv[2]

# Create a copy of the filename for future use
filename_copy = filename

# Dictionary to map specific file numbers to dates
dated = {
    "031": 19950706, "030": 19950606, "029": 19950507, "028": 19950407,
    "027": 19950308, "026": 19950206, "025": 19950107, "024": 19941208,
    "022": 19941009, "021": 19940909, "020": 19940810, "019": 19940711,
    "018": 19940611, "017": 19940512, "016": 19940412, "015": 19940313,
    "014": 19940211, "013": 19940112, "012": 19931213, "011": 19931113,
    "010": 19931014, "009": 19930914, "008": 19930815, "007": 19930716,
    "006": 19930616, "005": 19930517, "004": 19930417, "003": 19930318
}

# Extract the file number from the filename and get the corresponding date
nr = filename[-7:-4]
if nr in dated:
    filename = f"ocean_avg_0{nr}_{dated[nr]}.nc4"


# Set chunk sizes for Dask parallel processing
xchunk, ychunk, tchunk = -1, -1, 1
chunks = {"ocean_time": tchunk}

# Load grid metrics dataset and surface stress components
ds = xr.open_dataset(datapath + "extracted_fields/grid_metrics.nc")
ds["sustr"] = xr.open_dataset(filepath + filename).sustr
ds["svstr"] = xr.open_dataset(filepath + filename).svstr

# Rename dataset dimensions for compatibility with xgcm
ds = ds.rename({'eta_u': 'eta_rho', 'xi_v': 'xi_rho'})
ds = ds.chunk(chunks)

# Define coordinates and metrics for xgcm grid object
coords = {'X': {'center': 'xi_rho', 'inner': 'xi_u'}, 'Y': {'center': 'eta_rho', 'inner': 'eta_v'}}
metrics = {
    ('X',): ['dx', 'dx_u', 'dx_v', 'dx_psi'],  # X distances
    ('Y',): ['dy', 'dy_u', 'dy_v', 'dy_psi'],  # Y distances
    ('X', 'Y'): ['dA']  # Areas
}

# Create xgcm grid object
grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=[])

# Load contour and mask datasets
C = xr.open_dataset(datapath+f"contours/{contour}_contour.nc")
mask = xr.open_dataarray(datapath+f"contours/{contour}_mask.nc")

# Function to calculate circulation over an area
def circ_area(ds, grid, mask):
    # Extract surface stress components
    u, v = ds.sustr, ds.svstr

    # Compute derivatives for curl calculation
    dudy, dvdx = grid.derivative(u, 'Y'), grid.derivative(v, 'X')

    # Calculate curl and apply mask
    ucurl = (grid.interp(dvdx, ['X', 'Y']) - grid.interp(dudy, ['X', 'Y'])).where(mask)

    # Integrate over the area to compute total circulation
    taucirc_area = grid.integrate(ucurl, ['X', 'Y'])
    
    return taucirc_area

# Function to calculate circulation along a line
def circ_line(ds, grid, C):
    # Extract horizontal velocity and surface stress components
    taux, tauy = ds.sustr, ds.svstr
    
    # Calculate effective stress components normalized by density
    u = taux / rho0
    v = tauy / rho0

    # Convert to rho points and interpolate
    u_rho = grid.interp(u, "X") / ds.h
    v_rho = grid.interp(v, "Y") / ds.h

    # Interpolate stress components to contour points
    uc = u_rho.interp(xi_rho=C.xi_rho, eta_rho=C.eta_rho)
    vc = v_rho.interp(xi_rho=C.xi_rho, eta_rho=C.eta_rho)

    # Calculate line integrals for circulation
    taucirc_line = (uc * C.tx + vc * C.ty) * C.distance
    taucirc_line = taucirc_line.sum("point")
    
    return taucirc_line

# Calculate circulation over area and along line
ucirc_area = circ_area(ds, grid, mask)
ucirc_line = circ_line(ds, grid, C)

# Save results for every 4 time steps
n = len(ds.ocean_time) // 4
nstart = 0
timesteps = np.array_split(ds.ocean_time.values, n)

for i, t in enumerate(timesteps[nstart:]):
    i += nstart
    
    # Select data for current time segment
    ua = ucirc_area.sel(ocean_time=t)
    ul = ucirc_line.sel(ocean_time=t)
    
    # Prepare dataset for saving
    ds_u = xr.Dataset(
        data_vars=dict(
            taucirc_area=("ocean_time", ua.values),
            taucirc_line=("ocean_time", ul.values),
        ),
        coords=dict(ocean_time=ua.ocean_time.values),
        attrs=dict(
            long_name="surface stress circulation around contour",
            contour=f"{contour}"
        )
    )
    
    # Save current dataset to a NetCDF file
    ds_u.to_netcdf(datapath + f"temp/tau_circulation_{contour}_{filename[:-4]}_temp_{i:03}.nc")
    ds_u.close()

# Merge all saved datasets into one
ucirc = xr.open_mfdataset([datapath + f"temp/tau_circulation_{contour}_{filename[:-4]}_temp_{i:03}.nc" for i in range(n)])
ucirc.attrs["L_area"] = mask.L.values
ucirc.attrs["L_line"] = C.distance.sum("point").values

# Save the merged dataset to a final NetCDF file
ucirc.to_netcdf(datapath + f"processed/surface_stress_circulation_{contour}_" + filename_copy[:-1])

# Close all datasets
ucirc.close()
ds.close()
C.close()
mask.close()

# Remove temporary files to clean up
for i in range(n):
    os.remove(datapath + f"temp/tau_circulation_{contour}_{filename[:-4]}_temp_{i:03}.nc")
