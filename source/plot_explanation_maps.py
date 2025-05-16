import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np


def plot_explanation_map(
    attr,
    title,
    input_lats=None,
    input_lons=None,
    pixel_coords=None,
    cmap="Oranges",
    filename=None,
):
    # Convert numpy to xarray.DataArray
    data = attr[0, 0, :, :]
    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": input_lats, "lon": input_lons},
        name="attribution",
    )

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot attribution
    im = da.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=True,
        cbar_kwargs={"orientation": "horizontal", "label": "Attribution Value"},
    )

    ax.coastlines(resolution="10m", linestyle="-", linewidths=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.5)

    # Optional pixel highlight
    if pixel_coords:
        lat_idx, lon_idx = pixel_coords
        lat_val = input_lats[lat_idx]
        lon_val = input_lons[lon_idx]
        ax.plot(
            lon_val,
            lat_val,
            marker="o",
            color="black",
            markersize=6,
            transform=ccrs.PlateCarree(),
        )

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return filename
