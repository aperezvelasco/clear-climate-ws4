# preprocess_subset_netcdf.py
# Generates train/val/test NetCDF subsets for WS4 demo

import os
import glob
import xarray as xr
import numpy as np
from tqdm import tqdm

np.float = float


def load_datasets(data_dir, prefix="t2m_None", resolution="era5", year_filter=None):
    pattern = os.path.join(data_dir, f"{prefix}_{resolution}_*_0p*.nc")
    files = sorted(glob.glob(pattern))
    if "=" in year_filter:
        year = str(year_filter.split("=")[1])
        files = [f for f in files if year in f]
    elif "<" in year_filter:
        year = str(year_filter.split("<")[1])
        files = [f for f in files if int(f.split("_")[3][:4]) < int(year)]

    print(f"Loading {len(files)} files from {resolution.upper()}...")
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds


def save_dataset(ds, path):
    print(f"Saving to {path}")
    ds.to_netcdf(path)


def main():
    base_dir = "~/data/clear-climate/serbia"
    out_dir = os.path.join(base_dir, "subset")
    os.makedirs(out_dir, exist_ok=True)

    era5_dir = os.path.join(base_dir, "era5/t2m/None")
    cerra_dir = os.path.join(base_dir, "cerra/t2m/None")

    # Load pre-2019 datasets
    era5_pre = load_datasets(era5_dir, resolution="era5", year_filter="<2019")
    cerra_pre = load_datasets(cerra_dir, resolution="cerra", year_filter="<2019")
    era5_pre = era5_pre.sel(time=cerra_pre.time)

    total_time = era5_pre.time.size
    indices = np.random.permutation(total_time)
    train_indices = sorted(indices[:4000])
    val_indices = sorted(indices[4000:5000])

    # Sample indices
    train_era5 = era5_pre.isel(time=train_indices)
    val_era5 = era5_pre.isel(time=val_indices)
    train_cerra = cerra_pre.sel(time=train_era5.time)
    val_cerra = cerra_pre.sel(time=val_era5.time)

    # Load and save test set (2019 only)
    era5_test = load_datasets(era5_dir, resolution="era5", year_filter="=2019")
    cerra_test = load_datasets(cerra_dir, resolution="cerra", year_filter="=2019")
    era5_test = era5_test.sel(time=cerra_test.time)

    save_dataset(train_era5, os.path.join(out_dir, "train_era5.nc"))
    save_dataset(train_cerra, os.path.join(out_dir, "train_cerra.nc"))
    save_dataset(val_era5, os.path.join(out_dir, "val_era5.nc"))
    save_dataset(val_cerra, os.path.join(out_dir, "val_cerra.nc"))
    save_dataset(era5_test, os.path.join(out_dir, "test_era5_2019.nc"))
    save_dataset(cerra_test, os.path.join(out_dir, "test_cerra_2019.nc"))


if __name__ == "__main__":
    main()
