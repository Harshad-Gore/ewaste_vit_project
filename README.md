
# ewaste vit project

research-focused ewaste classification and hazard-aware decision support aligned to sdg 12.4.

## repository layout

- `data/`: train/val/test imagefolder dataset
- `models/`: checkpoints, experiment metrics, plots
- `training/`: script-based reproducible training pipeline
- `evaluation/`: reusable metrics utilities
- `notebooks/`: canonical exploratory and analysis workflow
- `agent/`: agent layer integration points
- `dashboard/`: streamlit app integration points
- `paper/`: report and publication assets

## setup

1. create and activate a virtual environment

```bash
python -m venv .venv
# windows
.venv\Scripts\activate
```

2. install dependencies

```bash
pip install -r requirements.txt
```

## dashboard launch (important)

do not run the dashboard with `python app.py`.
always launch with streamlit:

```bash
streamlit run dashboard/app.py
```

for optional llm-augmented agent explanations, set your api key as an environment variable (never hardcode in code):

```powershell
# current terminal session only
$env:GROQ_API_KEY = "<your_key_here>"

# optional: set default model
$env:GROQ_MODEL = "openai/gpt-oss-20b"
```

## script-first workflow (recommended)

all production/reproducible logic is available as python scripts. notebooks are retained for exploration and paper traceability.

run everything end-to-end:

```bash
python run_system.py all
```

by default, this reuses existing checkpoints in `models/classification` when available.
to force retraining of classification models:

```bash
python run_system.py all --force-train
```

this executes, in order:

1. adaptive deep-learning benchmark training
2. deep + traditional ml model competition and winner selection
3. clustering pipeline
4. ann hazard scoring pipeline

run individual stages:

```bash
python run_system.py train
python run_system.py compete
python run_system.py cluster
python run_system.py ann
```

key training behavior:

- gpu is auto-detected and used when available.
- batch size and num_workers auto-adapt to machine/runtime unless explicitly overridden.
- default epochs for deep benchmark training is `30`.

competition outputs:

- leaderboard and winner: `models/competition/leaderboard.json`
- deep model metrics: `models/competition/deep_results.json`
- traditional ml metrics: `models/competition/traditional_ml_results.json`

## direct benchmark command

if you want only deep benchmark training:

```bash
python training/research_benchmark.py --data-dir data --output-dir models/classification
```

default model set:

- resnet50
- vit_b16
- convnext_tiny
- swin_tiny
- efficientnet_b3

deep benchmark outputs:

- per-model checkpoint: `models/classification/<arch>/<arch>_best.pth`
- per-model metrics: `models/classification/<arch>/results.json`
- aggregate metrics: `models/classification/test_results.json`
- best model pointer: `models/classification/best_model.json`

## canonical notebook order (kept, not deleted)

use the numbered notebooks as the source of truth:

1. `notebooks/00_merge_classes.ipynb`
2. `notebooks/00_merge_and_verify.ipynb`
3. `notebooks/01_data_prep.ipynb`
4. `notebooks/02_cnn_classification.ipynb`
5. `notebooks/03_clustering.ipynb`
6. `notebooks/04_ann_hazard.ipynb`
7. `notebooks/05_full_comparison.ipynb`

legacy duplicate notebooks were moved to `notebooks/_legacy/`.

## notes

- paths in canonical notebooks are repository-relative (no machine-specific absolute paths).
- class imbalance is handled by weighted sampling in the script trainer.
- macro-f1 is used for model selection to improve minority-class quality.
