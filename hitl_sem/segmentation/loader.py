"""Sequential image loader for the segmentation dashboard.

Streams a folder of images one at a time. For every image after the first, the
boxes banked from all earlier images are subsampled, a classifier is trained on
them, and its prediction is offered as a zero-shot starting point which the human
can accept or correct.
"""

from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import normalize

from ..boxes import interpolate_features, sample_data
from ..models import predict_with_uncertainty, train_and_optimize


class SeqLoader:
    def __init__(self, img_dir, features_dir, labels_dir, boxes_dir, k=1000,
                 img_ext=".png", feature_ext=".npz"):
        self.img_dir = Path(img_dir)
        self.features_dir = Path(features_dir)
        self.labels_dir = Path(labels_dir)
        self.boxes_dir = Path(boxes_dir)
        self.k = k
        self.img_files = sorted(list(self.img_dir.glob(f"*{img_ext}")))
        self.index = 0
        self.total = len(self.img_files)
        self.out = None
        self.out_1 = None
        self.extractor = None

        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.boxes_dir.mkdir(parents=True, exist_ok=True)

        # Feature files are named <img stem>_<height>.npz; strip the height to key by image
        self.data_map = {}
        for f in self.features_dir.glob(f"*{feature_ext}"):
            key = f.stem.rsplit('_', 1)[0]
            self.data_map[key] = f

    def _get_or_extract_features(self, img_path, img_id, img_array):
        """Return precomputed features, extracting and caching them if absent."""
        if img_id in self.data_map and Path(self.data_map[img_id]).exists():
            data = np.load(self.data_map[img_id])
            return data['X']

        print(f"Features missing for {img_id}. Extracting on the fly...")

        if self.extractor is None:
            print("Loading DINOv3 PyTorch model into VRAM...")
            from ..features import Dinov3FeatureExtractor
            self.extractor = Dinov3FeatureExtractor()

        target_height = img_array.shape[0]
        save_dict = self.extractor.extract(str(img_path), target_height=target_height, mode="both")

        self.features_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.features_dir / f"{img_id}_{target_height}.npz"
        np.savez_compressed(out_path, **save_dict)

        self.data_map[img_id] = out_path

        return save_dict['X']

    def next_image(self, mode='patch'):
        """Advance to the next image and return its data dict, or None when done."""
        if self.index >= self.total:
            print("Finished all files.")
            return None

        img_path = self.img_files[self.index]
        img_id = img_path.stem
        img_array = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        features = self._get_or_extract_features(img_path, img_id, img_array)

        self.out = self.labels_dir / f"{img_id}_pred.npz"
        self.out_1 = self.boxes_dir / f"{img_id}_boxes.npz"

        patch_hw = (img_array.shape[0] // 16, img_array.shape[1] // 16)
        img_hw = img_array.shape[:2]

        if self.index == 0:
            X_train, y_train, X_test, y_pred, probs, entropy = None, None, None, None, None, None
            F = normalize(features.astype(np.float32), norm='l2', axis=1)
            X_test = F
            X_test_pixel = interpolate_features(F, patch_hw, img_hw) if mode == 'pixel' else None
        else:
            # Normalize the banked training patches jointly with this image's
            # features so both live in the same L2 space
            seen_ids = [p.stem for p in self.img_files[:self.index]]
            f_l, y_l = load_and_concat_npz(self.boxes_dir, seen_ids)
            f_sample, y_sample = sample_data(f_l, y_l, k=self.k, seed=42)
            f_all = np.concatenate([features, f_sample], axis=0)
            F = normalize(f_all.astype(np.float32), norm='l2', axis=1)
            X_train = F[len(features):]
            y_train = y_sample
            X_test = F[:len(features)]

            X_test_pixel = interpolate_features(X_test, patch_hw, img_hw) if mode == 'pixel' else None

            best_model = train_and_optimize(X_train, y_train)

            if mode == 'pixel':
                y_pred, probs, entropy = predict_with_uncertainty(best_model, X_test_pixel)
            else:
                y_pred, probs, entropy = predict_with_uncertainty(best_model, X_test)

        self.index += 1
        print(f"Loaded {img_id} ({self.index}/{self.total}) | Mode: {mode}")

        return {
            "img_array": img_array,
            "features": features,
            "features_normalized": X_test,
            "features_pixel_normalized": X_test_pixel,
            "X_train": X_train,
            "y_train": y_train,
            "y_pred": y_pred,
            "probabilities": probs,
            "entropy": entropy,
            "img_id": img_id
        }

    def reset(self):
        self.index = 0

    def save(self, y_pred, probs, entropy, X_train_c_original, y_train_c):
        """Write this image's prediction, and bank its annotated patches for later images."""
        if self.out is not None:
            np.savez_compressed(self.out, y_pred=y_pred, probs=probs, entropy=entropy)
            print(f"Saved predictions to {self.out}")
        else:
            print("No output to be saved.")

        if X_train_c_original is not None and y_train_c is not None:
            np.savez_compressed(self.out_1, X_train_c_original=X_train_c_original, y_train_c=y_train_c)
            print(f"Saved boxes to {self.out_1}")


def load_and_concat_npz(folder_path: str, img_ids) -> tuple:
    """Concatenate the banked patches of the given images, in the order given.

    ``img_ids`` is the sequence of image stems already visited; each contributes
    ``<stem>_boxes.npz`` if that file exists. Images the human left unannotated
    bank no boxes, so gaps in the sequence are tolerated: given four images with
    the second unannotated, the other three are used.
    """
    path = Path(folder_path)
    img_ids = list(img_ids)
    all_X, all_y = [], []
    loaded_ids = []

    for img_id in img_ids:
        f = path / f"{img_id}_boxes.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=True) as data:
            all_X.append(data[data.files[0]])
            all_y.append(data[data.files[1]])
            loaded_ids.append(img_id)

    if not all_X:
        raise FileNotFoundError(
            f"No banked boxes found in {folder_path} for any of: {img_ids}"
        )

    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)

    print(f"Loaded boxes from: {loaded_ids}")
    print(f"Final shape: {X_final.shape}, {y_final.shape}")

    return X_final, y_final
