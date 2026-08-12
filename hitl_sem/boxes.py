"""Turn human-drawn bounding boxes into patch-level training data.

Boxes are drawn in pixel space on the displayed image; DINOv3 features live on a
16x-downsampled grid, so every box is first mapped onto that grid, then the
enclosed feature cells are flattened into (n_samples, n_features) arrays.
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_boxes_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load a list of boxes, each a dict with keys x0, y0, x1, y1, cls."""
    with open(json_path, "r") as f:
        data = json.load(f)
    required = {"x0", "y0", "x1", "y1", "cls"}
    for i, d in enumerate(data):
        if not required.issubset(d.keys()):
            raise ValueError(f"Box #{i} missing keys. Found: {list(d.keys())}")
    return data


def map_box_to_feature_xyxy(
    box: Dict[str, Any],
    img_hw: Tuple[int, int],
    fmap_hw: Tuple[int, int],
    inclusive_xyxy: bool = True,
    pad_cells: int = 0,
) -> Tuple[int, int, int, int, str]:
    """Map a pixel-space box to feature-grid coords (slice end exclusive).

    Returns (x0f, y0f, x1f, y1f, cls).
    """
    H, W = img_hw
    Hf, Wf = fmap_hw

    scale_h = Hf / float(H)
    scale_w = Wf / float(W)

    x0, y0, x1, y1 = int(box["x0"]), int(box["y0"]), int(box["x1"]), int(box["y1"])
    cl = str(box["cls"])

    x0f = int(np.floor(x0 * scale_w))
    y0f = int(np.floor(y0 * scale_h))
    if inclusive_xyxy:
        x1f = int(np.floor(x1 * scale_w)) + 1
        y1f = int(np.floor(y1 * scale_h)) + 1
    else:
        x1f = int(np.ceil(x1 * scale_w))
        y1f = int(np.ceil(y1 * scale_h))

    # Padding is applied in feature cells, not pixels
    if pad_cells:
        x0f -= pad_cells; y0f -= pad_cells
        x1f += pad_cells; y1f += pad_cells

    x0f = max(0, min(Wf - 1, x0f))
    y0f = max(0, min(Hf - 1, y0f))
    x1f = max(0, min(Wf,     x1f))  # may equal Wf so the slice stays valid
    y1f = max(0, min(Hf,     y1f))

    if x1f <= x0f: x1f = min(Wf, x0f + 1)
    if y1f <= y0f: y1f = min(Hf, y0f + 1)

    return x0f, y0f, x1f, y1f, cl


def map_all_boxes_to_feature(
    boxes: List[Dict[str, Any]],
    img_hw: Tuple[int, int],
    fmap_hw: Tuple[int, int],
    inclusive_xyxy: bool = True,
    pad_cells: int = 0,
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """Map every box to the feature grid, grouped as {cls: [(x0f,y0f,x1f,y1f), ...]}."""
    out = defaultdict(list)
    for b in boxes:
        x0f, y0f, x1f, y1f, cl = map_box_to_feature_xyxy(
            b, img_hw, fmap_hw, inclusive_xyxy=inclusive_xyxy, pad_cells=pad_cells
        )
        out[cl].append((x0f, y0f, x1f, y1f))
    return out


def extract_feature_crops(
    feat: np.ndarray,
    fmap_boxes_by_cls: Dict[str, List[Tuple[int, int, int, int]]]
) -> Dict[str, List[np.ndarray]]:
    """Crop a (Hf, Wf, C) feature map with the given boxes.

    Returns {label: [crop of shape (hi, wi, C), ...]}.
    """
    Hf, Wf, C = feat.shape
    crops_by_cls: Dict[str, List[np.ndarray]] = defaultdict(list)

    for cls_name, boxes in fmap_boxes_by_cls.items():
        for (x0, y0, x1, y1) in boxes:
            x0 = max(0, min(Wf - 1, int(x0)))
            y0 = max(0, min(Hf - 1, int(y0)))
            x1 = max(0, min(Wf,     int(x1)))
            y1 = max(0, min(Hf,     int(y1)))
            if x1 <= x0:
                x1 = min(Wf, x0 + 1)
            if y1 <= y0:
                y1 = min(Hf, y0 + 1)

            crops_by_cls[cls_name].append(feat[y0:y1, x0:x1, :])

    return dict(crops_by_cls)


def crops_to_ml_data(
    crops_by_cls: Dict[str, List[np.ndarray]],
    class_to_label: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[int, str]]:
    """Flatten cropped feature maps into training arrays.

    Parameters
    crops_by_cls : {class name -> list of (hi, wi, C) crops}
    class_to_label : {class name -> integer label}; if None, classes are numbered
        alphabetically from whatever is present in the crops.

    Returns (X_all, y_all, X_by_cls, label_to_class).
    """
    X_list = []
    y_list = []
    X_by_cls = {}

    if class_to_label is None:
        class_to_label = {cls_name: i for i, cls_name in enumerate(sorted(crops_by_cls.keys()))}

    label_to_class = {i: cls for cls, i in class_to_label.items()}

    for cls_name, label in class_to_label.items():
        if cls_name not in crops_by_cls:
            continue

        crops = crops_by_cls[cls_name]
        flat_list = []

        for crop in crops:
            if crop.ndim != 3:
                raise ValueError(f"Crop for {cls_name} must be 3D (hi, wi, C) but got {crop.shape}")
            hi, wi, C = crop.shape
            flat_list.append(crop.reshape(hi * wi, C))

        if flat_list:
            X_cls = np.vstack(flat_list)
        else:
            X_cls = np.empty((0, 0))

        X_by_cls[cls_name] = X_cls

        if X_cls.size > 0:
            X_list.append(X_cls)
            y_list.append(np.full(X_cls.shape[0], label, dtype=np.int64))

    if not X_list:
        raise ValueError("No crops found for any of the classes provided in class_to_label.")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    return X_all, y_all, X_by_cls, label_to_class


def sample_data(X, y, k=1000, seed=42):
    """Stratified subsample of k rows, tolerant of classes too small to split evenly."""
    n_samples = len(X)

    if n_samples <= k:
        print(f"Dataset size ({n_samples}) is <= {k}. Returning original data.")
        return X, y

    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / n_samples

    target_counts = np.floor(proportions * k).astype(int)

    # Hand any rows lost to flooring to the classes with the largest remainder
    remainder = k - target_counts.sum()
    if remainder > 0:
        fractional_parts = (proportions * k) - target_counts
        extra_indices = np.argsort(fractional_parts)[-remainder:]
        target_counts[extra_indices] += 1

    sampled_indices = []
    for cls, count in zip(classes, target_counts):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        sampled_indices.extend(cls_indices[:count])

    sampled_indices = np.array(sampled_indices)
    rng.shuffle(sampled_indices)

    return X[sampled_indices], y[sampled_indices]


def get_training_data_from_crops(data, box_path, class_names=None):
    """Build training arrays for the current image from its saved boxes.

    Returns (X_train, y_train, X_train_original) where X_train comes from the
    L2-normalized features and X_train_original from the raw features (the latter
    is what gets banked for later images in the sequence).
    """
    f = data["features_normalized"]
    f_original = data["features"]
    boxes = load_boxes_from_json(box_path)

    h, w = data["img_array"].shape[:2]
    img_hw = (h, w)

    patch_h, patch_w = h // 16, w // 16
    fmap_hw = (patch_h, patch_w)

    fmap_boxes_by_cls = map_all_boxes_to_feature(
        boxes, img_hw=img_hw, fmap_hw=fmap_hw, inclusive_xyxy=False, pad_cells=0
    )

    # Fixing the mapping to the UI's class list keeps integer labels stable
    # across images even when a class is absent from the current annotation set
    class_to_label = None
    if class_names is not None:
        class_to_label = {name: i for i, name in enumerate(class_names)}

    crops_by_cls = extract_feature_crops(f.reshape(patch_h, patch_w, -1), fmap_boxes_by_cls)
    crops_by_cls_original = extract_feature_crops(f_original.reshape(patch_h, patch_w, -1), fmap_boxes_by_cls)

    X_train_c, y_train_c, _, _ = crops_to_ml_data(crops_by_cls, class_to_label=class_to_label)
    X_train_c_original, _, _, _ = crops_to_ml_data(crops_by_cls_original, class_to_label=class_to_label)

    return X_train_c, y_train_c, X_train_c_original


def combine_datasets(X1, y1, X2, y2):
    """Stack two (X, y) pairs."""
    return np.concatenate([X1, X2], axis=0), np.concatenate([y1, y2], axis=0)


def interpolate_features(features, patch_hw, img_hw):
    """Bilinearly upsample a (Hf*Wf, C) feature map to one vector per pixel."""
    h, w = patch_hw
    H, W = img_hw
    C = features.shape[1]

    feat_tensor = torch.from_numpy(features).float()
    feat_tensor = feat_tensor.reshape(1, h, w, C).permute(0, 3, 1, 2)
    feat_interp = F.interpolate(feat_tensor, size=(H, W), mode='bilinear', align_corners=False)

    return feat_interp.permute(0, 2, 3, 1).reshape(H * W, C).numpy()
