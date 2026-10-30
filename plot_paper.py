"""Replot saved experiment artifacts; no training, demos or GPU needed."""
import argparse
from pathlib import Path
from cs2.paperplots import KINDS, render

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(".") if Path("resolved_config.json").exists() else Path("outputs/rtx5070_experiments_3seeds_2026/log"))
    parser.add_argument("--figures", nargs="+", choices=KINDS, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", choices=["pdf", "svg", "png"], default=["pdf", "svg", "png"])
    args = parser.parse_args()
    status = render(args.run, args.figures, args.dpi, args.formats)
    print("Generated:", ", ".join(row["figure"] for row in status if row["status"] == "generated"))
    failed = [row for row in status if row["status"] != "generated"]
    if failed:
        print("Unavailable figures (missing evidence was not invented):", failed)
