# notebooks workflow

notebooks are preserved for exploration and reporting.
reproducible execution should use scripts from the repository root.

recommended full run:

```bash
python run_system.py all
```

stage-wise runs:

```bash
python run_system.py train
python run_system.py compete
python run_system.py cluster
python run_system.py ann
```

deep training defaults to 30 epochs and adapts runtime settings by available hardware.

this folder follows a single canonical pipeline:

1. `00_merge_classes.ipynb`
2. `00_merge_and_verify.ipynb`
3. `01_data_prep.ipynb`
4. `02_cnn_classification.ipynb`
5. `03_clustering.ipynb`
6. `04_ann_hazard.ipynb`
7. `05_full_comparison.ipynb`

legacy duplicate notebooks were moved to `notebooks/_legacy/` to avoid drift.

for reproducible model training outside notebooks, use:

```bash
python training/research_benchmark.py --data-dir data --output-dir models/classification
```
