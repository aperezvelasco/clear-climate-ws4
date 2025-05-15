import matplotlib.pyplot as plt


def plot_explanation_map(
    attr1, attr2, titles, suptitle="", cmap="viridis", filename=None, pixel_coords=None
):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=False)

    # Normalize color scale
    vmin = min(attr1.min(), attr2.min())
    vmax = max(attr1.max(), attr2.max())

    ims = []
    for i, (attr, title) in enumerate(zip([attr1, attr2], titles)):
        data = attr[0, 0, :, :]
        im = axs[i].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i].set_title(title)
        axs[i].axis("off")

        # Add black dot at pixel
        if pixel_coords:
            y, x = pixel_coords
            axs[i].plot(x, y, marker="o", color="black", markersize=5)

        ims.append(im)

    # Reserve space for colorbar
    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])  # [left, bottom, width, height]
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Attribution Value")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return filename
