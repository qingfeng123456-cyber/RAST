# Additional baseline experiment package

Run this command on the CUDA computer:

```bash
python main.py
```

The default command trains **only** LSTM-count, GRU-count, and HistGBDT, each with seeds 2026, 2027, and 2028. It reuses the nine already saved KNN, Vanilla Transformer, and RAST cohorts as immutable original evidence. All six methods remain in the final comparison regardless of ranking. Completion means the complete experiment finished; it does not mean a journal has accepted the paper or any particular model won.

The prepared data are included. This command neither downloads nor parses the 25.82-GB replay collection. It first checks the exact frozen data SHA-256 and match-disjoint split, hashes the original evidence, and aligns every saved test cutoff and count label. Original files under `results/paper_model/` are never overwritten. New training artifacts go to `outputs/baseline_supplement/`.

## Shared evaluation protocol

All models use the same 53-match frozen archive: 34/5/14 matches and 60,338/10,831/27,034 training/validation/test prefixes. Complete sampled histories are retained. The original Vanilla uses exact joint count NLL. The two recurrent baselines use exact NLL plus 0.25 conditional ranked probability score. RAST retains its original completed results. For the recurrent baselines, the default optimizer and selection protocol are AdamW at 0.0007, weight decay 0.0001, effective batch 128, up to 40 epochs, validation-energy-score early stopping with patience 8, 64 validation draws, and 256 test draws. Every neural epoch has a terminal progress bar; the tree baseline reports its regional fitting progress.

LSTM/GRU encode all 311 observed features with full-prefix recurrence, then use a population-conserving autoregressive count decoder. Their default depth is three, with LSTM hidden width 320 or GRU hidden width 384 and decoder width 192. Packed recurrent histories use FP32 for portability; the frozen Transformer results retain their recorded mixed-precision settings. The package records actual device, runtime, parameter count, objective, batches, training history, selected epoch and checkpoint.

HistGBDT is a conventional CPU histogram-gradient-boosted decision-tree baseline. Its input concatenates the last 311-feature observation, the mean of all frames in the complete prefix, the known hidden population, and the remaining count (624 features). Eleven conditional regional classifiers produce masked count probabilities; the last region receives the remainder. It fits multiclass log loss and selects boosting stages using validation conditional NLL. Its optimizer and selection criterion are recorded separately from neural AdamW/energy-score selection. Defaults use at most 200 boosting iterations, learning rate 0.05, 31 leaves, minimum leaf size 32, and 63 histogram bins. The trained model, feature manifest, iteration history and region-resume files are retained.

## Resume and returned evidence

Rerun `python main.py` after interruption. A contract checks data, code and scientific settings before resuming. `last.pt` resumes incomplete neural training; verified region checkpoints resume the tree baseline. HistGBDT is saved as `best_gbdt.joblib`. Validated completed training can resume only evaluation; validated completed cohorts are reused without training or evaluation. Changing scientific settings or implementation requires a new `run_name` in `supplemental_config.json` so an existing experiment remains interpretable.

On success, failure or keyboard interruption, the program automatically creates `RETURN_...zip` in the package root. `LATEST_RETURN.json` points to the latest archive. The ZIP contains the new code/configuration, processed data, frozen original results and mechanism sources, new checkpoints/history/predictions, full per-cutoff score tables, comparisons, plot sources, file hashes and actual completion/failure status. Previous ZIPs, raw DEMOs, Python caches and virtual environments are excluded. An interrupted or failed archive is still useful for debugging; only successful completion of all nine new plus nine reused cohorts creates `COMPLETE.json`.

Send the generated RETURN ZIP back for manuscript integration. Reports use cutoff averages within matches, equally weighted match means per seed, then the mean and sample standard deviation across seeds. Paired match-bootstrap intervals and signed RAST-versus-each-baseline differences are also saved. A negative performance difference favors RAST; a negative reported relative reduction favors the baseline. No unfinished result is filled in.

## Checks and redraw without training

```bash
python main.py --prepare-only --offline
python main.py --smoke --offline
python main.py --replot-only --offline
python replot_supplement.py --run outputs/baseline_supplement
```

`--prepare-only` verifies prepared data and all original saved cohorts. `--smoke` runs one epoch on four training, two validation and two test examples, using their unchanged full histories and very small CPU models and two tree boosting iterations. Smoke artifacts and its RETURN ZIP are isolated under `smoke_checks/`, marked **NOT FOR PAPER**, and never enter the formal comparison. `--replot-only` validates all completed formal cohorts and regenerates the combined comparison without training or a GPU. For iterative visual styling, `replot_supplement.py` reads the saved summary directly and writes PDF/SVG/PNG plus CSV/JSON sources without changing training contracts or fitting anything. The saved predictions and tables also support later mechanism and subgroup analysis.

The previous three-method reproduction entry remains available as `reproduce_main.py` for an already configured CUDA environment; it is not called by the default supplemental experiment. Environment auto-provisioning is reserved for `main.py`, so the optional legacy entry cannot accidentally re-launch the supplemental entry.
