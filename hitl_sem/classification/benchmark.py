"""Ablation: image-level vs patch-level vs human-in-the-loop classification.

Three predictions are produced for every test image and written to one CSV:

    pred_image_level  a classifier trained on one globally average-pooled
                      embedding per image (the conventional baseline)
    pred_patch_level  the patch-level root classifier from `initialize`, with the
                      image's class decided by majority vote over its patches
    pred_hitl         what the human-in-the-loop session actually concluded,
                      read back from the active learning log

    python -m hitl_sem.classification.benchmark
"""

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, normalize

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INIT_EMBEDDINGS = REPO_ROOT / "data" / "classification" / "embeddings_init"
DEFAULT_TEST_EMBEDDINGS = REPO_ROOT / "data" / "classification" / "embeddings_test"
DEFAULT_MODELS_DIR = REPO_ROOT / "classification_models"
DEFAULT_AL_LOG = REPO_ROOT / "classification_models" / "active_learning_logs.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "classification" / "ablation_predictions.csv"


class AblationExperiment:
    def __init__(self, init_embeddings_dir, test_embeddings_dir, models_dir, al_log_path,
                 output_path=DEFAULT_OUTPUT, embedding_suffix="_2040.npz"):
        """embedding_suffix matches how the test embeddings were named; the default
        corresponds to the 2040 px tall micrographs used in the paper."""
        self.init_embeddings_dir = Path(init_embeddings_dir)
        self.test_embeddings_dir = Path(test_embeddings_dir)
        self.models_dir = Path(models_dir)
        self.al_log_path = Path(al_log_path)
        self.output_path = Path(output_path)
        self.embedding_suffix = embedding_suffix

        print("Loading pre-trained Patch-Level model and scaler...")
        self.patch_scaler = joblib.load(self.models_dir / "global_scaler.joblib")
        self.patch_model = joblib.load(self.models_dir / "classifier_root.joblib")

        self.image_scaler = StandardScaler()
        self.image_model = None

    def run_experiment(self):
        print("\nPhase 1: Training Image-Level Classifier")
        self._train_image_classifier()

        print("\nPhase 2: Streaming Inference & Extraction")
        self._run_streaming_inference_and_save()

    def _train_image_classifier(self):
        """Train the image-level baseline on globally average-pooled embeddings."""
        X_train_list = []
        y_train_list = []

        print("Extracting Global Average Pooling (GAP) embeddings from Init directory...")
        for emb_path in self.init_embeddings_dir.rglob('*.npz'):
            leaf_label = emb_path.parent.name

            with np.load(emb_path) as npz_file:
                if 'X' not in npz_file:
                    continue
                raw_data = npz_file['X']
                if raw_data.ndim == 3:
                    raw_data = raw_data.reshape(-1, raw_data.shape[-1])

            normed_patches = normalize(raw_data, norm='l2', axis=1)
            gap_embedding = np.mean(normed_patches, axis=0)

            X_train_list.append(gap_embedding)
            y_train_list.append(leaf_label)

        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)
        print(f"Constructed Image-Level training set: {X_train.shape[0]} images, "
              f"{X_train.shape[1]} features.")

        X_train_scaled = self.image_scaler.fit_transform(X_train)

        # Heavy L2 regularization: this baseline sees ~22 images in 1024 dimensions
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        sgd = SGDClassifier(loss='log_loss', penalty='l2', class_weight='balanced', random_state=42)

        # cv=3 keeps stratification feasible on so few images
        grid = GridSearchCV(sgd, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        self.image_model = grid.best_estimator_
        print(f"Image-Level model trained. Optimal L2 penalty (alpha): {grid.best_params_['alpha']}")

    def _run_streaming_inference_and_save(self):
        """Score test images one at a time; a full stack of patch matrices would not fit in RAM."""
        print("Parsing Active Learning logs...")
        al_df = pd.read_csv(self.al_log_path)

        results = []

        for idx, row in al_df.iterrows():
            filename_col = 'filename' if 'filename' in row else 'image' if 'image' in row else al_df.columns[0]
            fname = row[filename_col]
            base_name = Path(fname).stem

            gt_leaf = str(row['ground_truth']).split(' -> ')[-1]

            if row.get('warning', False) and pd.notna(row.get('human_path')) and row['human_path'] != "None":
                hitl_leaf = str(row['human_path']).split(' -> ')[-1]
            else:
                hitl_leaf = str(row['model_path']).split(' -> ')[-1]

            emb_path = self.test_embeddings_dir / f"{base_name}{self.embedding_suffix}"

            if not emb_path.exists():
                print(f"Warning: Missing test embedding {emb_path}. Skipping.")
                continue

            with np.load(emb_path) as npz_file:
                raw_data = npz_file['X']
                if raw_data.ndim == 3:
                    raw_data = raw_data.reshape(-1, raw_data.shape[-1])

            normed_patches = normalize(raw_data, norm='l2', axis=1)

            gap_embedding = np.mean(normed_patches, axis=0).reshape(1, -1)
            gap_scaled = self.image_scaler.transform(gap_embedding)
            img_pred = self.image_model.predict(gap_scaled)[0]

            patches_scaled = self.patch_scaler.transform(normed_patches)
            patch_preds = self.patch_model.predict(patches_scaled)
            majority_pred = Counter(patch_preds).most_common(1)[0][0]

            results.append({
                'filename': fname,
                'ground_truth': gt_leaf,
                'pred_image_level': img_pred,
                'pred_patch_level': majority_pred,
                'pred_hitl': hitl_leaf
            })

            del raw_data, normed_patches, patches_scaled

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(self.output_path, index=False)
        print(f"\nExperiment Complete! Predictions saved to {self.output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Run the classification ablation.")
    parser.add_argument("--init-embeddings-dir", type=str, default=str(DEFAULT_INIT_EMBEDDINGS))
    parser.add_argument("--test-embeddings-dir", type=str, default=str(DEFAULT_TEST_EMBEDDINGS))
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--al-log", type=str, default=str(DEFAULT_AL_LOG))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--embedding-suffix", type=str, default="_2040.npz")
    args = parser.parse_args()

    AblationExperiment(
        init_embeddings_dir=args.init_embeddings_dir,
        test_embeddings_dir=args.test_embeddings_dir,
        models_dir=args.models_dir,
        al_log_path=args.al_log,
        output_path=args.output,
        embedding_suffix=args.embedding_suffix,
    ).run_experiment()


if __name__ == "__main__":
    main()
