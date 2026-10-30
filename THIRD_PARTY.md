# Third-party data and software

- **Raw source mirror:** OpenCS2 by `blanchon`, https://huggingface.co/datasets/blanchon/opencs2_dataset_demo . The upstream dataset card specifies CC BY 4.0. The fixed revision and all source HLTV URLs are retained. Raw DEMOs are not included.
- **Original match references:** HLTV match and demo URLs in `data/provenance/source_manifest.csv`. This study obtained binaries through the fixed-revision mirror, not by directly scraping HLTV.
- **DEMO parser:** LaihoE/demoparser, MIT, https://github.com/LaihoE/demoparser . This release uses `demoparser2==0.42.0` for parsing. Radar fields have approximate semantics; see https://github.com/LaihoE/demoparser/issues/194 .
- **Dependencies:** PyTorch, NumPy, pandas, scikit-learn, SciPy, Matplotlib, tqdm and their transitive dependencies retain their own licences.
- **Related research, not copied training data:** TAGNet (https://github.com/weyoung-open-lab/TAGNet-CS2), X-Ego (https://arxiv.org/abs/2510.19150), and probabilistic opponent modelling literature. TAGNet sparse snapshots are not included as training examples or auxiliary data.

No third-party checkpoint or model weight is relabelled as our work. The supplied checkpoints were trained by this project on the derived dataset.
