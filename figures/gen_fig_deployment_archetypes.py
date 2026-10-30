"""Draw the frozen deployment prototypes and spatial partition, without team profiles."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_support as plots


HERE = Path(__file__).resolve().parent


def main():
    prototypes = plots.table("deployment_prototypes.csv").set_index("pattern")
    centroids = plots.table("region_centroids.csv").set_index("region")

    # Values are the unchanged training-fitted prototype shares expressed as counts.
    fig = plt.figure(figsize=(3.5, 3.2), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[0.65, 2.0])
    upper = fig.add_subplot(grid[0])
    image = plots.heat(upper, prototypes.to_numpy() * 5, vmin=0, vmax=5)
    upper.set(
        xticks=range(12),
        xticklabels=range(12),
        yticks=range(len(prototypes)),
        yticklabels=[f"P{i}" for i in prototypes.index],
        xlabel="Region",
        ylabel="Prototype",
    )
    upper.tick_params(labelsize=7)
    colorbar = fig.colorbar(image, ax=upper, fraction=0.045, pad=0.025, ticks=[0, 2.5, 5])
    colorbar.set_label("CT count", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    upper.text(-0.14, 1.13, "(a)", transform=upper.transAxes, fontsize=8, va="bottom")

    lower = fig.add_subplot(grid[1])
    plots.bg(lower)
    lower.scatter(
        centroids.x,
        centroids.y,
        s=33,
        c=np.arange(12),
        cmap="tab20",
        edgecolors="white",
        linewidths=0.5,
    )
    for region, row in centroids.iterrows():
        lower.annotate(f"R{region}", (row.x, row.y), xytext=(3, 3),
                       textcoords="offset points", fontsize=7)
    lower.set(xlabel="Game X (units)", ylabel="Game Y (units)",
              xticks=[-1500, 0, 1500], yticks=[-1000, 1000, 3000])
    lower.tick_params(labelsize=7)
    lower.text(-0.23, 1.01, "(b)", transform=lower.transAxes, fontsize=8, va="bottom")

    for extension in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"fig_deployment_archetypes.{extension}", dpi=350,
                    bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)


if __name__ == "__main__":
    main()
