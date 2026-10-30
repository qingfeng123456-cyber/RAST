"""Redraw saved comparison tables with no fitting, inference, or GPU."""
from pathlib import Path
import argparse
import pandas as pd
from supplemental_runner import plot_summary, read_json, OLD_MODELS, NEW_MODELS


def cli():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=root/"outputs/baseline_supplement")
    args=parser.parse_args()
    output=args.run.resolve()
    source=output/"log/combined_summary.csv"
    summary=pd.read_csv(source)
    required={"model","metric","seed_mean","seed_std"}
    if not required.issubset(summary.columns):raise ValueError("The saved comparison table is incomplete")
    unknown=set(summary.model)-set(OLD_MODELS)-set(NEW_MODELS)
    if unknown:raise ValueError(f"Unknown methods must be added explicitly, not dropped: {unknown}")
    metadata=read_json(output/"log/combined_summary.json")
    plot_summary(summary,output,smoke=metadata.get("status")=="smoke_only")
    print("Redrawn from saved values: "+str(output/"log/figures"))


if __name__=="__main__":cli()
