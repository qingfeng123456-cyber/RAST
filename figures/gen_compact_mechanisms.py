"""Compact publication layouts with unchanged evidence and readable native fonts."""
from pathlib import Path
import hashlib
import json
import matplotlib.pyplot as plt
import plot_support as plots

HERE = Path(__file__).resolve().parent


def main():
    table = plots.ANALYSIS / "joint_count_law_source.csv"
    original = table.read_bytes()
    manifest = {}
    for name, scale in [("fig_readout", .86), ("fig_joint_law", .82)]:
        fig, caption, sources = getattr(plots, name)()
        width, height = fig.get_size_inches()
        fig.set_size_inches(width * scale, height * scale)
        # Native font sizes stay unchanged while the canvas and printed width
        # shrink together. Numeric cells, case selection, and color limits match.
        stem = name + "_compact"
        for extension in ("pdf", "svg", "png"):
            fig.savefig(HERE / f"{stem}.{extension}", dpi=350,
                        bbox_inches="tight", pad_inches=.045)
        plt.close(fig)
        manifest[stem] = dict(source_figure=name, canvas_scale=scale,
            sources=sources, caption=caption, retraining=False,
            source_sha256={s:hashlib.sha256((plots.ANALYSIS/s).read_bytes()).hexdigest()
                           for s in sources})
    if table.read_bytes() != original:
        table.write_bytes(original)
        raise RuntimeError("Numeric table serialization changed; original evidence restored")
    (plots.ANALYSIS / "compact_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("Compact layouts generated; numeric evidence unchanged.")


if __name__ == "__main__":
    main()
