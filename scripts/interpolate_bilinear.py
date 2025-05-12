import xarray as xr
import xesmf as xe
import numpy as np
import logging

# Load both ERA5 (coarse) and CERRA (fine) test sets
ds_era5 = xr.open_dataset("../data/test_era5.nc")
ds_cerra = xr.open_dataset("../data/test_cerra.nc")

# Ensure they use the same time
common_times = np.intersect1d(ds_era5.time.values, ds_cerra.time.values)
ds_era5 = ds_era5.sel(time=common_times)
ds_cerra = ds_cerra.sel(time=common_times)

# Create regridder from ERA5 to CERRA grid using bicubic interpolation
regridder = xe.Regridder(ds_era5, ds_cerra, method='bilinear')

# Perform regridding on ERA5 t2m
ds_era5_interpolated = regridder(ds_era5['t2m'])

# Save to disk
ds_era5_interpolated.to_dataset(name="t2m").to_netcdf(
    f"../data/test_bilinear-interpolation.nc"
)
logging.info("Bicubic interpolation saved to data/test_era5_bilinear-interpolation.nc")