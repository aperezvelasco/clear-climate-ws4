import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr


def plot_explanation_map(
    attr,
    title,
    input_lats=None,
    input_lons=None,
    pixel_coords=None,
    cmap="Oranges",
    filename=None,
    vmin=None,
    vmax=None,
    cbar_label="Attribution Value",
):

    # Convert numpy to xarray.DataArray
    data = attr[0, 0, :, :]
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": input_lats, "lon": input_lons},
        name="attribution",
    )

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot attribution
    mesh = da.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False  # Disable automatic colorbar
    )

    # Add coastlines and borders
    ax.coastlines(resolution="10m", linestyle="-", linewidths=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.5)

    # Optional pixel marker
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

    # Add title
    ax.set_title(title, fontsize=15)

    # Add manual colorbar
    cbar_position = [0.15, 0.05, 0.7, 0.05]
    cbar_ax = fig.add_axes(cbar_position)
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=12)

    # Save or show
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return filename

