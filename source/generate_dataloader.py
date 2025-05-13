import torch
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from xbatcher import BatchGenerator


def load_netcdf_pair(x_path, y_path, batch_size=16, variable_name="t2m", shuffle=True):
    ds_x = xr.open_dataset(x_path)
    ds_y = xr.open_dataset(y_path)
    ds_x = ds_x.transpose("time", "lat", "lon")
    ds_y = ds_y.transpose("time", "lat", "lon")

    # Ensure they share the same time steps (intersection only)
    common_times = np.intersect1d(ds_x["time"].values, ds_y["time"].values)
    ds_x = ds_x.sel(time=common_times)
    ds_y = ds_y.sel(time=common_times)

    if len(common_times) == 0:
        raise ValueError(f"No overlapping timestamps between {x_path} and {y_path}")

    x_gen = BatchGenerator(
        ds_x[[variable_name]],
        input_dims={
            "time": batch_size,
            "lat": len(ds_x.lat.values),
            "lon": len(ds_x.lon.values),
        },
        preload_batch=False,
    )

    y_gen = BatchGenerator(
        ds_y[[variable_name]],
        input_dims={
            "time": batch_size,
            "lat": len(ds_y.lat.values),
            "lon": len(ds_y.lon.values),
        },
        preload_batch=False,
    )

    def batch_to_tensor(x_batch, y_batch, variable_name="t2m"):
        x_arr = x_batch[variable_name].values
        y_arr = y_batch[variable_name].values

        # Ensure 3D shape before adding channel dim
        if x_arr.ndim == 2:
            x_arr = x_arr[None, :, :]  # Add time dim
        if y_arr.ndim == 2:
            y_arr = y_arr[None, :, :]

        x = torch.tensor(x_arr[:, None, :, :], dtype=torch.float32)
        y = torch.tensor(y_arr[:, None, :, :], dtype=torch.float32)
        return x, y

    data = [batch_to_tensor(xb, yb) for xb, yb in zip(x_gen, y_gen)]

    return DataLoader(data, batch_size=None, shuffle=shuffle)
