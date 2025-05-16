import matplotlib.pyplot as plt


def plot_explanation_map(
    attr, title, cmap="Oranges", filename=None, pixel_coords=None
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=False)

    data = attr[0, 0, :, :]
    im = ax.imshow(data, cmap=cmap, vmin=data.min(), vmax=data.max())
    ax.set_title(title)
    ax.axis("off")

    # Add black dot at pixel
    if pixel_coords:
        y, x = pixel_coords
        ax.plot(x, y, marker="o", color="black", markersize=5)

    # Colorbar below
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.05])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Attribution Value")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return filename
