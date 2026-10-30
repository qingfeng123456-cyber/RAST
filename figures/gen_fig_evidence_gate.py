"""Replot the final-model gate mechanism from its released raw diagnostics."""
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
if not (RELEASE / 'plot_paper.py').is_file():
    raise FileNotFoundError('Run this script inside the complete RAST repository')
RUN = RELEASE / "results/paper_model"
subprocess.run([sys.executable, str(RELEASE/"plot_paper.py"), "--run", str(RUN), "--figures", "B1"],check=True)
for ext in ("pdf","png","svg"):
    shutil.copy2(RUN/"figures"/f"B1.{ext}",HERE/f"fig_evidence_gate.{ext}")
shutil.copy2(RUN/"figures/B1_source.csv",HERE/"evidence_gate_source.csv")
