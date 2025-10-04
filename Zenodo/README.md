# Code for Estimating Wind- and Vorticity-Driven Circulation in Lofoten800

This directory contains scripts for estimating wind- and vorticity-driven circulation around depth contours in numerical simulations of the Lofoten Basin (Lofoten800). The simulations are based on the Regional Ocean Modeling System (ROMS), with 62 vertical levels and a horizontal resolution of approximately 800 meters. These simulations are provided by MET Norway.

## Pipeline
1. **Download model files**: Lofoten800 model files are available at [this link](https://thredds.met.no/thredds/catalog/romshindcast/lofoten800_2017/catalog.html).
2. **Process model files**: Run `process_model_files.py` to calculate and extract relevant variables from the model fields.
3. **Create contour files**: Run `create_contours.ipynb` to generate files containing contour masks and contour vectors. These are required to extract variables from a contour.
4. **Calculate contour integrals**: Run `calculate_contour_integrals.py` and `calculate_surface_stress_circulation.py` to compute time series of velocity circulations, surface stress circulation, and vorticity fluxes for a given contour.
5. **Combine time series**: Run `combine_time_series.py` to merge all the contour integrals into a single file for easier handling.
6. **Estimate circulation**: Run `estimate_circulation.py` to perform the final time integration and estimate the vorticity- and wind-driven circulation. This step uses methods from `methods_time_integrals.py`.

The file `lofoten800_lofoten_h_50km_timeseries.nc` is the output of steps 1-5 for a depth contour in the Lofoten Basin, smoothed with a 50 km filter. The file `lofoten800_estimates.nc` is the output of step 6 for the same contour.

## Content
- **calculate_contour_integrals.py**
- **calculate_surface_stress_circulation.py**
- **combine_time_series.py**
- **create_contours.ipynb**
- **estimate_circulation.py**
- **lofoten800_estimates.nc**: NetCDF file containing the estimated and diagnosed velocity circulation. The estimated circulation is divided into contributions from the initial state, surface stress circulation, barotropic vorticity flux, and baroclinic vorticity flux.
- **lofoten800_lofoten_h_50km_timeseries.nc**: NetCDF file containing time series of surface stress circulation, vorticity fluxes, and diagnosed velocity circulations.
- **methods_time_integrals.py**
- **process_model_files.py**
  

Enjoy! 🌊