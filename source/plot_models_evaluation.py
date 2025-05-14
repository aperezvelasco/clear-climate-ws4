import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr


def plot_comparison(metric_1, metric_2, titles, lats, lons, title, cmap, output_path):
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    data = [metric_1, metric_2]
    vmin = min(metric_1.min(), metric_2.min())
    vmax = max(metric_1.max(), metric_2.max())
    if vmin < 0 and vmax >= 0:
        if abs(vmin) > vmax:
            vmax = abs(vmin)
        else:
            vmin = -vmax

    meshes = []

    for i, ax in enumerate(axs):
        da = xr.DataArray(
            data[i], coords={"lat": lats, "lon": lons}, dims=("lat", "lon")
        )
        mesh = ax.pcolormesh(
            da.lon, da.lat, da, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
        )
        meshes.append(mesh)
        ax.coastlines(linewidths=0.5, linestyle="-")
        ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.5)
        ax.set_title(f"{titles[i]} (Mean: {round(float(da.mean()), 2)})", fontsize=12)

    # Colorbar that spans both plots
    cbar_ax = fig.add_axes([0.1, 0.15, 0.8, 0.05])  # [left, bottom, width, height]
    cbar = fig.colorbar(meshes[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label(title, fontsize=12)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path
