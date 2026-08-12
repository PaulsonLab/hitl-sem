"""Fit the initial hierarchical classifiers from the class-foldered init set.

The folder structure under the init directory defines the label tree: one SGD
classifier is trained per branching node, plus a PCA + Isolation Forest anomaly
detector per class. All models consume patch embeddings that have been
L2-normalized per patch and then standardized by one global scaler, which is
frozen here so the active learner sees the same feature space.

    python -m hitl_sem.classification.initialize
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.class_weight import compute_class_weight

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INIT_DIR = REPO_ROOT / "data" / "classification" / "images_init"
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "data" / "classification" / "embeddings_init"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "classification_models"


class HierarchicalInitializer:
    def __init__(self, init_dir, embedding_dir, output_dir, max_anomaly_samples=50000):
        """
        Args:
            init_dir: structured initialization images; folder depth = hierarchy depth.
            embedding_dir: where DINOv3 features are loaded from, or saved to if missing.
            output_dir: where trained models and the global scaler are written.
            max_anomaly_samples: patch cap for Isolation Forest, to bound RAM and CPU.
        """
        self.init_dir = Path(init_dir)
        self.embedding_dir = Path(embedding_dir)
        self.output_dir = Path(output_dir)
        self.max_anomaly_samples = max_anomaly_samples

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = []
        self.extractor = None

    def run(self):
        print("Scanning directory structure")
        self._scan_dataset()

        print("\nExtracting/loading features")
        self._extract_or_load_features()

        print("\nFitting global standard scaler")
        self._build_global_scaler()

        print("\nTraining hierarchical SGD classifiers")
        self._train_classifiers()

        print("\nTraining anomaly detectors")
        self._train_anomaly_detectors()

        print(f"\nInitialization complete. All assets saved to: {self.output_dir}")

    def _scan_dataset(self):
        """Read the classification hierarchy off the folder structure."""
        valid_exts = {'.png'}

        for img_path in self.init_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                # Dendritic/01_NoIDPhase/img.png -> ['Dendritic', '01_NoIDPhase']
                rel_path = img_path.relative_to(self.init_dir)
                hierarchy = list(rel_path.parent.parts)

                self.dataset.append({
                    "img_path": img_path,
                    "rel_dir": rel_path.parent,
                    "hierarchy": hierarchy
                })
        print(f"Found {len(self.dataset)} images across "
              f"{len(set(tuple(d['hierarchy']) for d in self.dataset))} unique paths.")

    def _extract_or_load_features(self):
        """Load cached .npz features, extracting any that are missing."""
        for data in self.dataset:
            img_path = data["img_path"]
            img_array = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            h, w = img_array.shape[:2]

            emb_folder = self.embedding_dir / data["rel_dir"]
            emb_folder.mkdir(parents=True, exist_ok=True)
            emb_path = emb_folder / f"{img_path.stem}_{h}.npz"

            if emb_path.exists():
                raw_features = np.load(emb_path)['X']
            else:
                print(f"Features missing for {img_path.stem}. Extracting via DINOv3...")
                if self.extractor is None:
                    from ..features import Dinov3FeatureExtractor
                    self.extractor = Dinov3FeatureExtractor()

                extraction_dict = self.extractor.extract(str(img_path), target_height=h, mode="both")
                raw_features = extraction_dict['X']
                np.savez_compressed(emb_path, X=raw_features)

            data["features"] = normalize(raw_features.astype(np.float32), norm='l2', axis=1)

    def _build_global_scaler(self):
        """Fit one StandardScaler on all init patches, freezing the feature space."""
        all_features = np.vstack([d["features"] for d in self.dataset])
        print(f"Global feature matrix shape: {all_features.shape}")

        self.scaler = StandardScaler()
        self.scaler.fit(all_features)

        del all_features

        joblib.dump(self.scaler, self.output_dir / "global_scaler.joblib")
        print("Global StandardScaler fitted and frozen.")

    def _train_classifiers(self):
        """Train one SGD classifier per branching node of the discovered hierarchy."""
        tasks = defaultdict(lambda: {"X": [], "y": []})

        for data in self.dataset:
            scaled_features = self.scaler.transform(data["features"])
            hierarchy = data["hierarchy"]

            if len(hierarchy) > 0:
                tasks["root"]["X"].append(scaled_features)
                tasks["root"]["y"].extend([hierarchy[0]] * len(scaled_features))

            for i in range(len(hierarchy) - 1):
                parent = hierarchy[i]
                child = hierarchy[i + 1]
                tasks[parent]["X"].append(scaled_features)
                tasks[parent]["y"].extend([child] * len(scaled_features))

        for node, task_data in tasks.items():
            X_train = np.vstack(task_data["X"])
            y_train = np.array(task_data["y"])
            unique_classes = np.unique(y_train)

            if len(unique_classes) < 2:
                continue

            print(f"Training Classifier -> Node: '{node}' | Classes: {unique_classes} | "
                  f"Total Samples: {len(X_train)}")

            # Tune on a subsample; the full patch matrix across parallel CV folds
            # would blow up RAM
            max_cv_samples = 40000
            if len(X_train) > max_cv_samples:
                idx = np.random.choice(len(X_train), max_cv_samples, replace=False)
                X_cv, y_cv = X_train[idx], y_train[idx]
            else:
                X_cv, y_cv = X_train, y_train

            param_grid = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
            sgd = SGDClassifier(loss='log_loss', penalty='l2', class_weight='balanced', random_state=42)

            grid = GridSearchCV(sgd, param_grid, cv=5, n_jobs=2)
            grid.fit(X_cv, y_cv)

            best_alpha = grid.best_params_['alpha']
            print(f"  [Tuning] Best alpha: {best_alpha} | Subset CV Acc: {(grid.best_score_ * 100):.2f}%")

            print("  [Training] Fitting final model on ALL data in batches...")

            # Weights come from the full label distribution, not the batch's
            computed_weights = compute_class_weight(class_weight='balanced',
                                                    classes=unique_classes, y=y_train)
            weight_dict = dict(zip(unique_classes, computed_weights))

            best_sgd = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=best_alpha,
                class_weight=weight_dict,
                random_state=42
            )

            batch_size = 50000
            for i in range(0, len(X_train), batch_size):
                end_idx = min(i + batch_size, len(X_train))
                X_batch = X_train[i:end_idx]
                y_batch = y_train[i:end_idx]

                # partial_fit needs the full class list on every call
                best_sgd.partial_fit(X_batch, y_batch, classes=unique_classes)

            joblib.dump(best_sgd, self.output_dir / f"classifier_{node}.joblib")
            print(f"  -> Model saved for '{node}'")

    def _train_anomaly_detectors(self):
        """Train a PCA + Isolation Forest pipeline for every class in the hierarchy."""
        anomaly_tasks = defaultdict(list)

        for data in self.dataset:
            scaled_features = self.scaler.transform(data["features"])
            anomaly_tasks["root"].append(scaled_features)
            for node in data["hierarchy"]:
                anomaly_tasks[node].append(scaled_features)

        for cls_name, x_list in anomaly_tasks.items():
            X_train = np.vstack(x_list)

            if len(X_train) > self.max_anomaly_samples:
                idx = np.random.choice(len(X_train), self.max_anomaly_samples, replace=False)
                X_train = X_train[idx]

            print(f"Training Anomaly Detector -> Class: '{cls_name}' | Samples: {len(X_train)}")

            n_comp = min(64, X_train.shape[1], X_train.shape[0])
            pca = PCA(n_components=n_comp, random_state=42)
            X_pca = pca.fit_transform(X_train)

            iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            iso.fit(X_pca)

            joblib.dump({"pca": pca, "iso": iso}, self.output_dir / f"anomaly_{cls_name}.joblib")


def main():
    parser = argparse.ArgumentParser(description="Train the initial hierarchical classifiers.")
    parser.add_argument("--init-dir", type=str, default=str(DEFAULT_INIT_DIR))
    parser.add_argument("--embedding-dir", type=str, default=str(DEFAULT_EMBEDDING_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-anomaly-samples", type=int, default=50000)
    args = parser.parse_args()

    HierarchicalInitializer(
        init_dir=args.init_dir,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        max_anomaly_samples=args.max_anomaly_samples,
    ).run()


if __name__ == "__main__":
    main()
