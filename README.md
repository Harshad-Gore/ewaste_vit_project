
# E-Waste Vision Intelligence System

E-waste component classification, hazard-aware decision support, and analytics tooling aligned with SDG 12.4 and SDG 12.5.

This repository is a modular research system that combines:

- deep visual classification for 18 e-waste component classes
- model benchmarking across multiple CNN and transformer backbones
- competition between deep and traditional machine-learning models
- unsupervised clustering from learned visual embeddings
- hazard scoring from a dedicated tabular ANN pipeline
- a policy-aware decision layer for disposal routing
- a Streamlit dashboard that exposes inference, policy reasoning, benchmark evidence, analytics, and registry views

## 1. Research Scope

The system is designed to answer the following research need:

> Can e-waste components be identified visually and then mapped into hazard-aware disposal guidance using a transparent, modular pipeline suitable for research communication and future operational deployment?

The current implementation answers that question with a **single-label image classification system** plus a **policy and analytics layer**. It does **not** yet implement full object detection. Mixed-scene images are therefore handled as **triage support**, not true detection.

## 2. System Architecture

```
```

## 3. Repository Layout

- `data/`: image dataset organized as `train/`, `val/`, and `test/` in `ImageFolder` format
- `training/`: reproducible training scripts for deep benchmarking and model competition
- `pipelines/`: clustering and ANN hazard pipelines
- `evaluation/`: reusable metrics utilities
- `agent/`: hazard lookup, regulation checks, disposal recommendation logic, and optional LLM augmentation
- `dashboard/`: Streamlit-based research and operations dashboard
- `models/`: checkpoints, metrics JSON files, plots, and downstream analytical artifacts
- `notebooks/`: exploratory and paper-traceable notebook workflow
- `paper/`: publication assets and supporting material

## 4. Dataset and Taxonomy

### 4.1 Dataset structure

The vision dataset is stored as an `ImageFolder` dataset with three splits:

- `data/train`: 23,960 images
- `data/val`: 1,800 images
- `data/test`: 1,800 images

Total dataset size in the current checkout: **27,560 images** across **18 classes**.

### 4.2 Component classes

The current class taxonomy is:

- `Air-Conditioner`
- `Battery`
- `Keyboard`
- `Laptop`
- `Microchip-IC`
- `Microwave`
- `Mobile`
- `Mouse`
- `PCB`
- `Passive-Component`
- `Printer`
- `Refrigerator`
- `Resistor`
- `Television`
- `Washing Machine`
- `heat-sink`
- `light bulbs`
- `transistor`

### 4.3 Hazard-aware mapping

Each component is mapped in the policy layer to:

- a hazard band: `HIGH`, `MEDIUM`, or `LOW`
- a material profile
- a disposal pathway

Examples:

- `Battery` -> `HIGH` hazard -> hazardous battery recycling
- `PCB` -> `HIGH` hazard -> certified e-waste recycler for metal recovery
- `Printer` -> `MEDIUM` hazard -> e-waste routing with toner-safe handling
- `Keyboard` -> `LOW` hazard -> plastics and small-e-waste stream

These mappings are defined centrally in `agent/tools.py`, which makes the disposal reasoning explicit and auditable.

## 5. End-to-End Working of the System

### 5.1 Stage 1: Deep visual benchmark training

The main visual learning pipeline is implemented in `training/research_benchmark.py`.

#### Input

- `data/train`
- `data/val`
- `data/test`

#### Supported architectures

- `resnet18`
- `resnet50`
- `efficientnet_b0`
- `efficientnet_b3`
- `convnext_tiny`
- `swin_tiny`
- `vit_b16`

#### Training strategy

The benchmark pipeline uses transfer learning with architecture-specific classification heads. Key characteristics:

- pretrained ImageNet initialization
- weighted random sampling to reduce class imbalance effects
- class-weighted cross-entropy loss
- label smoothing
- data augmentation for training:
  - resize and random crop
  - horizontal flip
  - random rotation
  - color jitter
  - random erasing
- initial backbone freezing
- scheduled partial unfreezing for fine-tuning
- `AdamW` optimizer
- cosine annealing learning-rate schedule
- optional AMP on CUDA devices
- early stopping based on **validation macro-F1**

#### Why macro-F1 matters here

Model selection is intentionally based on **macro-F1**, not only accuracy, because the project must remain robust across all component classes rather than only favor the most frequent classes.

#### Output artifacts

For each architecture, the script writes:

- `models/classification/<arch>/<arch>_best.pth`
- `models/classification/<arch>/results.json`
- `models/classification/<arch>/classification_report.txt`

At the aggregate level it writes:

- `models/classification/test_results.json`
- `models/classification/best_model.json`
- `models/classification/benchmark_summary.json`

The repository also contains archived benchmark snapshots such as `models/classification/dl_results.json`.

### 5.2 Stage 2: Best-model selection and benchmark interpretation

After benchmarking, the project identifies the best visual classifier from the saved results.

In the current repository state:

- `best_model.json` points to **ResNet50**
- the archived snapshot in `dl_results.json` reports:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| ResNet50 | 95.72% | 0.9564 | 0.9564 |
| EfficientNet-B0 | 95.39% | 0.9534 | 0.9534 |
| ResNet18 | 95.00% | 0.9497 | 0.9497 |
| ViT-B16 | 94.50% | 0.9447 | 0.9447 |

Important interpretation:

- these metrics describe performance on the **single-label held-out classification setup**
- they do **not** mean the system is already a true multi-object scene understanding model
- this is why a cluttered collage image can still produce uncertain live behavior despite a strong benchmark score

### 5.3 Stage 3: Model competition across deep and traditional ML

The competition framework is implemented in `training/model_competition.py`.

This script performs two kinds of comparison:

#### A. Deep model comparison

It reloads every discovered deep checkpoint and evaluates it on the test split.

#### B. Traditional ML comparison using deep embeddings

It extracts learned embeddings from the best deep model or a chosen embedding source model, then trains classical classifiers on those features.

Traditional candidates include:

- KNN
- SVM with RBF kernel
- linear SVM
- random forest
- logistic regression
- naive Bayes
- gradient boosting
- a hierarchical SVM that first predicts hazard band and then predicts component class inside that hazard group

#### Output artifacts

When this pipeline is run, it writes:

- `models/competition/deep_results.json`
- `models/competition/traditional_ml_results.json`
- `models/competition/all_players_results.json`
- `models/competition/leaderboard.json`

This allows the project to compare whether raw deep classifiers or downstream traditional learners are more competitive on the learned representation space.

### 5.4 Stage 4: Unsupervised clustering from learned visual embeddings

The clustering workflow is implemented in `pipelines/clustering_pipeline.py`.

#### Purpose

This stage evaluates whether visually learned embeddings also carry unsupervised structure aligned with semantic or hazard relationships.

#### Current implementation

- the pipeline loads the visual dataset across train, validation, and test
- it uses the **ResNet50 checkpoint** as the embedding extractor backbone
- embeddings are standardized
- PCA is applied to retain 95% of variance
- KMeans clustering is run, currently with `n_clusters=3`
- t-SNE projections are generated for visualization

#### Output artifacts

- `models/clustering/clustering_metrics.json`
- `models/clustering/clustering_results.json`
- `models/clustering/embeddings.npy`
- `models/clustering/embeddings_pca.npy`
- `models/clustering/cluster_labels.npy`
- `models/clustering/tsne_result.npy`
- `models/clustering/tsne_by_class.png`
- `models/clustering/tsne_by_hazard.png`
- additional graphs under `models/clustering/graphs/`

#### Current supporting metrics

From the current saved artifacts:

- silhouette score: `0.1662`
- normalized mutual information: `0.0340`
- adjusted Rand index: `0.0251`

These results should be interpreted as **supporting analytical evidence**, not as a deployed classifier. They help reveal latent structure, but they are not a substitute for supervised prediction.

### 5.5 Stage 5: ANN-based hazard scoring pipeline

The hazard modeling workflow is implemented in `pipelines/ann_hazard_pipeline.py`.

#### Purpose

This stage models **hazard severity** separately from the image classifier. It is meant to provide a quantitative supporting layer for environmental risk reasoning.

#### Important methodological note

This ANN is trained on a **generated tabular dataset**, not directly on the image embeddings.

The script defines component-level hazard profiles with attributes such as:

- lithium content
- lead content
- mercury content
- cadmium content
- CFC presence
- recyclability
- material type
- weight class

It then synthesizes tabular samples by varying:

- component age
- weight
- condition
- regional risk
- disposal history

From these inputs it computes a hazard score in the range `0-100`, trains a feed-forward ANN regressor, and then reports both:

- regression metrics on continuous hazard score
- derived hazard-class metrics after bucketing into `HIGH`, `MEDIUM`, and `LOW`

#### Output artifacts

- `models/ann/ann_best_18cls.pth`
- `models/ann/ann_results_18cls.json`
- `models/ann/ewaste_tabular_18cls.csv`
- `models/ann/feature_importance.png`
- `models/ann/hazard_class_confusion.png`
- `models/ann/predicted_vs_actual.png`

#### Current supporting metrics

From the current saved artifacts:

- hazard class accuracy: `95.06%`
- `R^2`: `0.9846`
- `MAE`: `2.7732`
- `RMSE`: `3.9278`
- `MAPE`: `7.2`

This stage strengthens the project by showing that the system is not limited to visual recognition alone; it also models downstream environmental severity.

### 5.6 Stage 6: Agentic decision layer

The decision layer lives in:

- `agent/tools.py`
- `agent/agent.py`
- `agent/prompts.py`

#### What it does

Once the image classifier predicts a component and a confidence score, the agent layer transforms that output into an operational recommendation.

Internally, the decision workflow is:

1. `hazard_lookup(component)`
2. `regulation_check(hazard_level, confidence, threshold)`
3. `disposal_recommendation(...)`
4. optional LLM augmentation for explanation wording

#### Core design

The important architectural point is that the **core decision logic is deterministic and tool-driven**.

- hazard level comes from the hazard map
- material profile comes from the material map
- disposal pathway comes from the disposal map
- human review is triggered by confidence thresholding

If a `GROQ_API_KEY` is available, the system can augment the explanation using Groq's OpenAI-compatible API. If Anthropic is available and configured, that path is also supported. However:

- the LLM does not replace the hazard mapping
- the LLM does not replace the confidence threshold
- the LLM does not replace the routing rule
- the LLM only augments the explanation layer

#### Agent outputs exposed to the dashboard

The agent returns:

- predicted component
- hazard level
- material profile
- disposal pathway
- short recommendation
- explanation text
- SDG target
- compliance flag
- human review flag
- agent mode
- explanation source
- LLM provider
- tool execution trace

This makes the reasoning inspectable rather than opaque.

### 5.7 Stage 7: Dashboard workflow

The dashboard is implemented in `dashboard/app.py` and is organized as a workflow-oriented interface rather than a simple upload page.

#### Dashboard initialization

When the dashboard starts, it:

- detects runtime hardware
- discovers available classification checkpoints
- loads benchmark metrics from `test_results.json` and/or `dl_results.json`
- loads the best checkpoint and class names
- loads ANN and clustering metrics
- loads the hazard taxonomy for the policy layer

#### Dashboard views

The dashboard exposes five operational views.

#### Operations

This is the live inference view.

It allows the user to:

- upload an image
- load a test image
- set a human-review threshold
- enable or disable composite scene scan
- run inference

For each inference, the dashboard:

1. applies the evaluation transform
2. runs the classifier
3. computes softmax probabilities
4. shows predicted class, confidence, latency, and top-5 class scores
5. evaluates whether confidence is reliable enough for downstream action

##### Composite scene review

If enabled, the dashboard performs a tile-based scene scan:

- the image is divided into a `2 x 2` or `3 x 3` grid
- each tile is classified independently
- the system aggregates repeated component evidence and tile-level hazard counts

This is explicitly a **triage aid**, not an object detector.

#### Policy

This view converts the classifier output into:

- hazard level
- SDG-linked compliance context
- human-review requirement
- recommended pathway
- material profile
- decision rationale
- tool execution trace

#### Benchmarks

This view exposes:

- benchmark tables
- confusion matrices
- training curves
- Grad-CAM interpretability plots
- metric-source discrepancies if multiple snapshots exist

It also supports optional Groq-based drafting of a results paragraph for research writing.

#### Analytics

This view exposes:

- ANN hazard metrics
- regression performance
- clustering metrics
- supporting ANN and clustering figures

It can also draft a short discussion paragraph through Groq if configured.

#### Registry

This view documents:

- active and available model checkpoints
- dataset split composition
- class-level counts
- the hazard taxonomy and disposal pathways used by the policy engine

#### Session-state design

The dashboard stores the active image and latest inference results in Streamlit session state. This allows the same prediction to flow across views without rerunning the model every time the user switches sections.

## 6. What the Dashboard Means Scientifically

The dashboard should be described as:

> A workflow-oriented research console that combines single-label e-waste classification, confidence-aware triage, hazard-aware decision support, benchmark visualization, and supporting analytical modules.

It should **not** be described as:

- a true object detection system
- a conveyor-belt-ready multi-object perception system
- a fully autonomous disposal controller

## 7. Current Limitations

These limitations are important for research honesty:

- the visual model is a **single-label classifier**, so cluttered scenes and collages violate the training assumption
- the composite scene review is **tile-based classification**, not bounding-box detection
- the ANN hazard model is trained on a **generated tabular proxy dataset**, not measured plant-floor sensor data
- benchmark snapshots from archived outputs and freshly regenerated outputs should not be conflated as a single universal deployment number
- LLM augmentation is optional and explanatory; it is not the source of the core classification or policy decision

## 8. Future Research Direction

The clearest next research step is to move from single-label classification to **multi-object detection or multi-label recognition**. This would allow the system to process cluttered scenes, conveyor-belt scenarios, and mixed e-waste inputs more faithfully.

Until then, the current system should be positioned as:

- a strong **single-object benchmarked classifier**
- a **hazard-aware decision-support layer**
- a **research dashboard with transparent evidence**
- a foundation for future detector-based deployment

## 9. Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 10. Running the Dashboard

Always launch the dashboard with Streamlit:

```bash
streamlit run dashboard/app.py
```

Do not run it with `python dashboard/app.py`.

Optional LLM-augmented explanations use environment variables:

```powershell
$env:GROQ_API_KEY = "<your_key_here>"
$env:GROQ_MODEL = "openai/gpt-oss-20b"
```

Never hardcode API keys in source files.

## 11. Script-First Pipeline Workflow

The repository is designed to be runnable end to end from scripts.

Run the full pipeline:

```bash
python run_system.py all
```

Force retraining of the classification benchmark:

```bash
python run_system.py all --force-train
```

Run individual stages:

```bash
python run_system.py train
python run_system.py compete
python run_system.py cluster
python run_system.py ann
```

### What `run_system.py all` does

1. checks whether classification checkpoints already exist
2. runs deep benchmark training if needed
3. runs model competition
4. runs clustering analysis
5. runs ANN hazard modeling

## 12. Direct Commands

Run only the deep classification benchmark:

```bash
python training/research_benchmark.py --data-dir data --output-dir models/classification
```

Run only the model competition:

```bash
python training/model_competition.py --data-dir data --classification-dir models/classification --output-dir models/competition
```

Run only clustering:

```bash
python pipelines/clustering_pipeline.py --data-dir data --classification-dir models/classification --output-dir models/clustering
```

Run only ANN hazard modeling:

```bash
python pipelines/ann_hazard_pipeline.py --output-dir models/ann
```

## 13. Key Artifact Map

### Classification

- checkpoints: `models/classification/<arch>/<arch>_best.pth`
- per-model metrics: `models/classification/<arch>/results.json`
- benchmark snapshot: `models/classification/test_results.json`
- archived benchmark snapshot: `models/classification/dl_results.json`
- best model pointer: `models/classification/best_model.json`
- plots: `models/classification/graphs/`

### Competition

- `models/competition/leaderboard.json`
- `models/competition/deep_results.json`
- `models/competition/traditional_ml_results.json`
- `models/competition/all_players_results.json`

### Clustering

- `models/clustering/clustering_metrics.json`
- `models/clustering/clustering_results.json`
- `models/clustering/graphs/`

### ANN hazard model

- `models/ann/ann_results_18cls.json`
- `models/ann/ann_best_18cls.pth`
- `models/ann/graphs/`

## 14. Canonical Notebook Order

The notebooks are retained for exploration and paper traceability.

Suggested order:

1. `notebooks/00_merge_classes.ipynb`
2. `notebooks/00_merge_and_verify.ipynb`
3. `notebooks/01_data_prep.ipynb`
4. `notebooks/02_cnn_classification.ipynb`
5. `notebooks/03_clustering.ipynb`
6. `notebooks/04_ann_hazard.ipynb`
7. `notebooks/05_full_comparison.ipynb`

Legacy duplicates were moved to `notebooks/_legacy/`.

## 15. Summary

This repository implements a complete research system for e-waste analysis:

- vision-based component classification
- downstream hazard-aware reasoning
- analytical support through clustering and ANN hazard modeling
- transparent dashboard-based communication

Its current strength lies in **single-label benchmarked classification with explainable downstream policy logic**. Its next frontier is **true multi-object detection for mixed e-waste scenes**.
