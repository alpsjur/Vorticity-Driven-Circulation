import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt


# Define the data path where datasets are stored
datapath = "data/"

# Define the drag coefficient
Cd = 0.003

# Define the filenames for the datasets (these should be updated with the correct file paths)
A4_file = "A4/dummy"  # TODO: Update with the actual file name
lofoten800_file = "lofoten800/dummy"  # TODO: Update with the actual file name

# Load the datasets using xarray
ts_A4 = xr.open_dataset(datapath + A4_file)
ts_lofoten800 = xr.open_dataset(datapath + lofoten800_file)

# Calculate the corrected drag coefficient (Cdnod) for each dataset
Cdnod_A4 = Cd * ts_A4.ub2circ_area.values / (ts_A4.ucirc_area.values * np.abs(ts_A4.ucirc_area.values))
Cdnod_lofoten800 = Cd * ts_lofoten800.ub2circ_area.values / (ts_lofoten800.ucirc_area.values * np.abs(ts_lofoten800.ucirc_area.values))

# Create a plot to compare the two Cdnod values
fig, ax = plt.subplots()

# Plot the Cdnod for both datasets
ax.plot(Cdnod_A4, label='A4')
ax.plot(Cdnod_lofoten800, label='Lofoten 800')

# Add a horizontal line to represent the original drag coefficient Cd
ax.axhline(Cd, color='red', linestyle='--', label='Cd')

# Add labels and a legend for better clarity
ax.set_xlabel('Index')
ax.set_ylabel('Adjusted Cd')
ax.legend()
