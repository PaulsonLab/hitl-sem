# Human-in-the-loop segmentation and classification of SEM micrographs

Code for the three case studies in the paper. A frozen DINOv3 ViT-L/16 backbone
supplies patch embeddings; lightweight scikit-learn models are trained on top of
them from a handful of human annotations, and the human is asked for input only
where the model is uncertain.

| Case study | Task | Entry point |
|---|---|---|
| 1 | Patch-level segmentation of one large micrograph into three phases | `run_segmentation_case1.ipynb` |
| 2 | Pixel-level segmentation of a 16-tile sequence into four classes | `run_segmentation_case2.ipynb` |
| 3 | Hierarchical image classification with active learning, plus the ablation | `run_classification.ipynb` |

All computations were performed in Python 3.10 using PyTorch 2.5.1 (CUDA 11.8)
and scikit-learn 1.6.1, on a single workstation equipped with an Intel Core
i7-11800H CPU, 16 GB of RAM and an NVIDIA GeForce RTX 3070 Laptop GPU (8 GB
VRAM).

## Installation

```bash
conda create -n hitl-sem python=3.10
conda activate hitl-sem

# Pick the PyTorch build that matches your CUDA version; this is what was tested
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

pip install -e .
```

`pip install -e .` installs the dependencies and puts `hitl_sem` on the import
path, so the `python -m hitl_sem.…` commands below work from any directory. It is
an editable install by design: the case-study commands resolve their default
paths relative to this checkout. `pip install -r requirements.txt` also works if
you would rather not install the package and always run from the repository root.

If you already have a working environment, `pip install -e . --no-deps` registers
the package without touching anything you have installed. This matters because
`opencv-python` declares a dependency on NumPy 2, so a plain install pulls NumPy 2
in even where NumPy 1.24 works fine.

The interactive dashboards need `ipympl`; run the notebooks in JupyterLab and
keep the `%matplotlib widget` line at the top. A GPU is only needed for feature
extraction — every classifier trains on CPU.

Tested versions: Python 3.10.19, PyTorch 2.5.1, torchvision 0.20.1,
scikit-learn 1.6.1, NumPy 1.24.4, SciPy 1.15.1, pandas 2.2.3, OpenCV 4.12.0,
Pillow 11.1.0, Matplotlib 3.8.0, ipywidgets 8.0.3, ipympl 0.9.7.

## DINOv3 setup

The backbone is loaded from a local clone plus a downloaded checkpoint; neither
is redistributed here.

```bash
git clone https://github.com/facebookresearch/dinov3.git dinov3
```

Then request access to the pretrained backbones from the DINOv3 repository and
place `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` in `dinov3_weights/`. The
result must look like:

```
dinov3/hubconf.py
dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

See `dinov3/README.md` and `dinov3_weights/README.md` for details.

## Data

Images are distributed on Zenodo: **[DOI to be added]**. Unpack each archive into
the matching placeholder folder:

| Archive | Contents | Unpack into |
|---|---|---|
| `segmentation_case1_images.zip` | 1 micrograph, 2040 px tall | `data/segmentation/case1/images/` |
| `segmentation_case2_images.zip` | 16 tiles, `img_00.png` … `img_15.png`, 510 px tall | `data/segmentation/case2/patches/` |
| `classification_images.zip` | 58 micrographs in `Dendritic/`, `DualPhase/`, `SinglePhase/` | `data/classification/images_all/` |

The same record also carries the precomputed DINOv3 embeddings. They are optional
— every extraction step below regenerates them — but unpacking them lets you run
the whole pipeline on CPU without installing the DINOv3 backbone at all:

| Archive | Contents | Unpack into |
|---|---|---|
| `embeddings_classification.zip` | 5.0 GB — `embeddings_init/<class>/` and `embeddings_test/` | `data/classification/` |
| `embeddings_segmentation.zip` | 173 MB — `case1/features/` and `case2/features/` | `data/segmentation/` |

Filenames matter: the classification ground truth CSV and the segmentation
annotations are keyed by image name, and the sequence order in case study 2 is
the alphabetical order of the tiles.

Feature files live next to the images as `<image stem>_<image height>.npz` and
take roughly 5 GB across all three case studies. Every stage skips extraction for
any image that already has one, so a populated `features/` or `embeddings_*/`
folder simply makes that step a no-op.

## Running the case studies

### Case studies 1 and 2 — segmentation

Open `run_segmentation_case1.ipynb` or `run_segmentation_case2.ipynb` and run
the cells. Features are extracted on first use and cached, so the first image is
slow and the rest are not. Draw boxes, press **Train & Predict**, refine, then
**Save & Next Image**. Predictions land in `outputs/segmentation_case<N>/`.

To pre-extract features instead of waiting inside the notebook:

```bash
python -m hitl_sem.features \
    --data-folder data/segmentation/case2/patches \
    --output-folder data/segmentation/case2/features
```

The published sessions can also be reproduced without any clicking. The human
boxes are shipped in `annotations/`, and the replay reruns the exact train →
predict → save sequence headlessly:

```bash
python -m hitl_sem.segmentation.replay --case 2
python -m hitl_sem.segmentation.replay --case 2 --validate --reference-dir <your predictions folder>
```

Note that pixel mode (case study 2) interpolates patch features to full
resolution before predicting, which needs several GB of RAM per tile.

### Case study 3 — classification

1. **Reproduce the published split.** `classification_ground_truth_mapping.csv`
   records which images were used to initialize the models and which were held
   out; the shipped models were trained on exactly that split.

   ```bash
   python -m hitl_sem.classification.split_data --from-csv
   ```

   Drop `--from-csv` to draw a fresh random split (seed 42, 40 % init) instead.

2. **Extract embeddings** for both halves — skip this if you unpacked
   `embeddings_classification.zip`.

   ```bash
   python -m hitl_sem.features --recursive \
       --data-folder data/classification/images_init \
       --output-folder data/classification/embeddings_init

   python -m hitl_sem.features \
       --data-folder data/classification/images_test \
       --output-folder data/classification/embeddings_test
   ```

   `--recursive` mirrors the class subfolders, which is how `initialize` reads
   the label hierarchy.

3. **Train the initial models** — optional, since `classification_models/` is
   shipped.

   ```bash
   python -m hitl_sem.classification.initialize
   ```

   This overwrites `classification_models/`; pass `--output-dir` to write
   elsewhere.

4. **Run the active learning session**: open `run_classification.ipynb`. The log
   and the updated models are written under `outputs/classification/`, leaving
   the shipped assets untouched.

5. **Run the ablation.**

   ```bash
   python -m hitl_sem.classification.benchmark
   ```

   By default this reads the published session log in `classification_models/`
   and reproduces the numbers in the paper; point `--al-log` at
   `outputs/classification/active_learning_logs.csv` to score your own session.
   The result is a CSV with the ground truth and all three predictions per image.

## Using the segmentation tools on your own images

The segmentation stack is not specific to this paper or to SEM data. `SeqLoader`
and `ActiveSegmentationDashboard` take every path as an argument and learn
whatever classes you type into the UI, so any sequence of grayscale images can be
segmented the same way: draw a few boxes, train, correct where the entropy map
says the model is unsure, move on.

Point the package at your DINOv3 install once, and it no longer needs this
checkout:

```bash
export DINOV3_REPO=/path/to/dinov3            # the clone containing hubconf.py
export DINOV3_WEIGHTS=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

On Windows use `setx DINOV3_REPO "C:\path\to\dinov3"`, or set the variables in
Python before the first extraction. Both are also settable per call, via
`Dinov3FeatureExtractor(repo_dir=..., weights_path=...)`.

Then, in a notebook with `%matplotlib widget`:

```python
from hitl_sem.segmentation import SeqLoader
from hitl_sem.segmentation.dashboard import ActiveSegmentationDashboard

loader = SeqLoader(
    img_dir="my_project/images",        # the images to segment, in sequence
    features_dir="my_project/features", # DINOv3 features, extracted on first use
    labels_dir="my_project/predictions",# per-image predictions are written here
    boxes_dir="my_project/boxes",       # annotated patches banked for later images
)

dashboard = ActiveSegmentationDashboard(
    loader, save_dir="my_project/history", mode="patch", track_progress=True
)
```

Things worth knowing:

- Images must be `.png`, grayscale, 8- or 16-bit. Pass `img_ext=` to `SeqLoader`
  if yours use a different extension, and note that images are visited in
  alphabetical order — that order is the sequence the model learns along.
- `mode="patch"` predicts one label per 16 × 16 patch and is fast. `mode="pixel"`
  interpolates the features to full resolution for a smooth boundary, at a cost
  of several GB of RAM per image.
- Classes are created in the UI with **Add Class**; there is no fixed class list
  and no retraining from scratch when you add one.
- From the second image onward the boxes drawn so far are subsampled (`k=1000`
  patches by default) and used to pre-segment the next image, so annotation
  effort drops as you go.
- Everything is written under the directories you pass; nothing is written back
  into the package.

To extract features up front rather than on first use:

```bash
python -m hitl_sem.features --data-folder my_project/images \
                            --output-folder my_project/features
```

The classification stack is usable the same way — `HierarchicalInitializer`
derives its label tree from your folder structure, and `SequentialActiveLearner`
takes its directories as arguments — but its command-line defaults assume this
checkout's layout, so pass the paths explicitly.

## What is in the repository

### `hitl_sem/` — shared

| Module | Contents |
|---|---|
| `features.py` | `Dinov3FeatureExtractor` and the folder-level extraction CLI. Writes `<stem>_<height>.npz` with a per-patch feature matrix under key `X`. |
| `models.py` | The scaler + L2 logistic regression pipeline used for segmentation, its grid search, prediction with Shannon entropy, and the guarded sigmoid used by the SGD classifiers. |
| `boxes.py` | Maps pixel-space boxes onto the 16×-downsampled feature grid, crops the enclosed features, and flattens them into training arrays. Also stratified subsampling and feature interpolation to pixel resolution. |
| `viz.py` | The prediction-overlay and entropy panels of the segmentation dashboard. |

### `hitl_sem/segmentation/`

| Module | Contents |
|---|---|
| `loader.py` | `SeqLoader`: streams the image sequence, loads or extracts features, and for every image after the first trains on the banked annotations to offer a zero-shot prediction. |
| `dashboard.py` | `ActiveSegmentationDashboard`: the ipywidgets box-drawing UI and the train/save loop. |
| `replay.py` | Headless rerun of a finished session from the saved boxes, with optional validation against a reference. |

### `hitl_sem/classification/`

| Module | Contents |
|---|---|
| `split_data.py` | Init/test split, either fresh or reproduced from the ground truth CSV. |
| `initialize.py` | `HierarchicalInitializer`: freezes one global scaler, trains an SGD classifier per branching node of the folder hierarchy, and a PCA + Isolation Forest anomaly detector per class. |
| `active_learner.py` | `SequentialActiveLearner`: hierarchical majority-vote inference with confidence and anomaly gates, the human query UI, and the online model updates. |
| `benchmark.py` | `AblationExperiment`: image-level (global average pooling) vs patch-level (majority vote) vs human-in-the-loop. |

### Data and assets

| Path | Contents |
|---|---|
| `classification_models/` | The published scaler, root classifier, per-class anomaly detectors, and the active learning session log the ablation is computed from. |
| `annotations/` | The human bounding boxes from both segmentation sessions, as JSON. |
| `classification_ground_truth_mapping.csv` | The published init/test split and per-image labels. |
| `data/`, `outputs/` | Empty placeholders; see the tables above. |
