"""Human-in-the-loop hierarchical classification of a test image sequence.

For each image the patch embeddings are pushed down the label tree: at every node
the per-patch predictions are tallied into a majority vote, and the patches are
scored by that class's anomaly detector. The image is classified automatically
only while the majority is decisive enough and few patches look anomalous;
otherwise the run stops, shows the anomaly and probability maps, and asks the
human to pick the class (or flag the image as out-of-distribution).

Whatever path is confirmed — automatically or by the human — is then used to
update the classifiers (`partial_fit`) and refit the anomaly detectors, so the
models improve over the course of the session.
"""

from pathlib import Path

import cv2
import ipywidgets as widgets
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from sklearn.preprocessing import normalize

from ..models import safe_predict_proba

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TEST_DIR = REPO_ROOT / "data" / "classification" / "images_test"
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "data" / "classification" / "embeddings_test"
DEFAULT_MODELS_DIR = REPO_ROOT / "classification_models"


class SequentialActiveLearner:
    def __init__(self, test_dir=DEFAULT_TEST_DIR, embedding_dir=DEFAULT_EMBEDDING_DIR,
                 models_dir=DEFAULT_MODELS_DIR, ground_truth_csv=None,
                 conf_thresh=0.80, anomaly_thresh=0.10, max_buffer=50000,
                 save_diagnostics=False, diagnostics_dir=None, log_path=None):
        """
        Args:
            conf_thresh: minimum majority-vote share to accept a node automatically.
            anomaly_thresh: maximum fraction of anomalous patches tolerated.
            max_buffer: patch cap on the Isolation Forest refit buffer.
            ground_truth_csv: enables test mode, which logs correctness alongside
                the decisions. Leave as None to run blind (deployment).
            log_path: where the session log is written. Defaults to
                models_dir/active_learning_logs.csv.
        """
        self.test_dir = Path(test_dir)
        self.embedding_dir = Path(embedding_dir)
        self.models_dir = Path(models_dir)

        self.conf_thresh = conf_thresh
        self.anomaly_thresh = anomaly_thresh
        self.max_buffer = max_buffer

        self.save_diagnostics = save_diagnostics
        self.diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir else None
        self.log_path = Path(log_path) if log_path else self.models_dir / "active_learning_logs.csv"

        if self.save_diagnostics and self.diagnostics_dir:
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
            print(f"Diagnostics tracking ENABLED. Saving to: {self.diagnostics_dir}")

        self.is_test_mode = ground_truth_csv is not None
        if self.is_test_mode:
            self.gt_df = pd.read_csv(ground_truth_csv).set_index("filename")
            print("Mode: TEST (Ground Truth Loaded for Evaluation Logging)")
        else:
            print("Mode: DEPLOYMENT (Running blind, human is the only source of truth)")

        self.logs = []

        self.scaler = joblib.load(self.models_dir / "global_scaler.joblib")
        self._load_models()

        self.image_files = [p for p in self.test_dir.iterdir()
                            if p.suffix.lower() in {'.png', '.jpg', '.tif'}]
        self.current_idx = 0

        self.output_log = widgets.Output()
        self.ui_container = widgets.VBox()

    def _load_models(self):
        """Load every node classifier and anomaly pipeline into memory."""
        self.classifiers = {}
        self.anomaly_detectors = {}
        self.buffers = {}

        for file in self.models_dir.glob("classifier_*.joblib"):
            node = file.stem.replace("classifier_", "")
            self.classifiers[node] = joblib.load(file)

        for file in self.models_dir.glob("anomaly_*.joblib"):
            cls_name = file.stem.replace("anomaly_", "")
            self.anomaly_detectors[cls_name] = joblib.load(file)
            self.buffers[cls_name] = []

    def _safe_predict_proba(self, model, X):
        return safe_predict_proba(model, X)

    def run(self):
        """Launch the active learning dashboard."""
        display(self.ui_container)
        display(self.output_log)
        self._process_next_image()

    def _process_next_image(self):
        """Classify images until one trips a threshold and needs the human."""
        # A loop rather than recursion: the sequence can be long and each frame
        # would otherwise pin its features in memory
        while self.current_idx < len(self.image_files):

            self.img_path = self.image_files[self.current_idx]
            self.img_array = cv2.imread(str(self.img_path), cv2.IMREAD_GRAYSCALE)
            self.h, self.w = self.img_array.shape[:2]
            self.patch_h, self.patch_w = self.h // 16, self.w // 16

            emb_path = self.embedding_dir / f"{self.img_path.stem}_{self.h}.npz"
            raw_features = np.load(emb_path)['X']
            f_l2 = normalize(raw_features.astype(np.float32), norm='l2', axis=1)
            self.scaled_features = self.scaler.transform(f_l2)

            self.pred_path = []
            self.warning_triggered = False
            self.warning_details = None
            self.diagnostic_maps = {}

            current_node = "root"

            while current_node in self.classifiers:
                model = self.classifiers[current_node]
                probs = self._safe_predict_proba(model, self.scaled_features)
                preds = model.classes_[np.argmax(probs, axis=1)]

                unique, counts = np.unique(preds, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                majority_ratio = np.max(counts) / len(preds)

                if majority_class in self.anomaly_detectors:
                    pipeline = self.anomaly_detectors[majority_class]
                    X_pca = pipeline['pca'].transform(self.scaled_features)
                    anomaly_preds = pipeline['iso'].predict(X_pca)
                    anomaly_ratio = np.sum(anomaly_preds == -1) / len(anomaly_preds)
                    anomaly_scores = -pipeline['iso'].decision_function(X_pca)  # higher = more anomalous
                else:
                    anomaly_ratio = 0.0
                    anomaly_scores = np.zeros(len(preds))

                majority_prob_array = probs[:, list(model.classes_).index(majority_class)]
                self.diagnostic_maps[current_node] = {
                    "prob_map": majority_prob_array.reshape(self.patch_h, self.patch_w),
                    "anomaly_map": anomaly_scores.reshape(self.patch_h, self.patch_w),
                    "full_prob_map": probs.reshape(self.patch_h, self.patch_w, len(model.classes_)),
                    "class_names": model.classes_,
                    "predicted_class": majority_class
                }

                self.pred_path.append(majority_class)

                if anomaly_ratio > self.anomaly_thresh:
                    self.warning_triggered = True
                    self.warning_details = {"level": current_node, "type": "Anomaly", "val": anomaly_ratio}
                    break
                elif majority_ratio < self.conf_thresh:
                    self.warning_triggered = True
                    self.warning_details = {"level": current_node, "type": "Confidence", "val": majority_ratio}
                    break

                current_node = majority_class

            if self.warning_triggered:
                if self.save_diagnostics and self.diagnostics_dir:
                    self._save_diagnostic_maps(level_node=self.warning_details['level'])

                # Hand control to the UI; the human's click resumes the sequence.
                # Questioning restarts from the top of the tree.
                self._build_ui_for_human(level_node="root")
                return
            else:
                self._log_automated_success()
                self.current_idx += 1

        with self.output_log:
            clear_output()
            print("All images processed! Experiment complete.")
            self._save_logs()

    def _build_ui_for_human(self, level_node):
        """Show the image, anomaly map and probability map, plus one button per class."""
        self.ui_container.children = []

        maps = self.diagnostic_maps.get(level_node, self.diagnostic_maps.get("root"))

        with self.output_log:
            clear_output()
            print(f"File: {self.img_path.name}")
            print(f"WARNING RAISED: Level [{self.warning_details['level']}] | "
                  f"Type: {self.warning_details['type']} ({self.warning_details['val']:.2%})")
            print(f"Model's intended path: {' -> '.join(self.pred_path)}")

            if self.is_test_mode:
                try:
                    gt_row = self.gt_df.loc[self.img_path.name]
                    gt_path = " -> ".join([str(x) for x in [gt_row['level_1'], gt_row['level_2'], gt_row['level_3']]
                                           if str(x) not in ["None", "nan", "NaN"]])
                    print(f"Ground Truth Path:     {gt_path}  <-- [TEST MODE ONLY]")
                except KeyError:
                    print("Ground Truth Path:     [Not found in CSV]")

        out_plot = widgets.Output()
        with out_plot:
            fig = plt.figure(figsize=(16, 5), constrained_layout=True)

            gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05])

            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            cax1_full = fig.add_subplot(gs[0, 2])
            ax2 = fig.add_subplot(gs[0, 3])
            cax2_full = fig.add_subplot(gs[0, 4])

            cax1_full.axis("off")
            cax2_full.axis("off")

            # Shorter colorbars, inset so they match the image height
            cax1 = cax1_full.inset_axes([0.0, 0.188, 1.0, 0.632])
            cax2 = cax2_full.inset_axes([0.0, 0.188, 1.0, 0.632])

            ax0.imshow(self.img_array, cmap='gray')
            ax0.set_title("SEM Image")
            ax0.axis('off')

            anom_data = maps["anomaly_map"]
            max_abs = np.max(np.abs(anom_data))
            max_abs = max_abs if max_abs > 0 else 1.0

            im_anom = ax1.imshow(anom_data, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs,
                                 interpolation='nearest')
            ax1.set_title(f"Anomaly Map ({maps['predicted_class']})")
            ax1.axis('off')
            fig.colorbar(im_anom, cax=cax1)

            im_prob = ax2.imshow(maps["prob_map"], cmap='magma', vmin=0, vmax=1)
            ax2.set_title(f"Probability Map ({maps['predicted_class']})")
            ax2.axis('off')
            fig.colorbar(im_prob, cax=cax2)

            plt.show()

        model = self.classifiers[level_node]
        button_box = widgets.HBox()
        buttons = []

        label = widgets.Label(f"Please specify the correct class for [{level_node}]: ")

        for cls in model.classes_:
            btn = widgets.Button(description=cls, button_style='primary')
            btn.on_click(lambda b, c=cls: self._on_human_choice(c, level_node))
            buttons.append(btn)

        btn_new = widgets.Button(description="NEW (OOD)", button_style='danger')
        btn_new.on_click(lambda b: self._on_human_choice("NEW", level_node))
        buttons.append(btn_new)

        button_box.children = [label] + buttons
        self.ui_container.children = [out_plot, button_box]

    def _on_human_choice(self, choice, current_node):
        """Descend to the next level, or finish the image and update the models."""
        if not hasattr(self, 'human_path'):
            self.human_path = []

        if choice == "NEW":
            self._log_human_intervention(is_ood=True)
            self._update_models_and_advance(self.human_path)
            return

        self.human_path.append(choice)

        if choice in self.classifiers:
            self._build_ui_for_human(level_node=choice)
        else:
            self._log_human_intervention(is_ood=False)
            self._update_models_and_advance(self.human_path)

    def _update_models_and_advance(self, confirmed_path):
        """Update the classifiers and anomaly detectors, then load the next image."""
        if not confirmed_path:
            # The whole image was flagged NEW at the root; nothing to learn from
            self.current_idx += 1
            if hasattr(self, 'human_path'): del self.human_path
            self._process_next_image()
            return

        with self.output_log:
            print("Updating models in the background...")

            # Each node on the path is taught the label of the child that follows it,
            # e.g. root -> DualPhase, DualPhase -> 01_ParticlePrecipitates
            node_progression = ["root"] + confirmed_path[:-1]
            labels = confirmed_path

            for node, label in zip(node_progression, labels):
                model = self.classifiers[node]
                y_new = np.array([label] * len(self.scaled_features))
                model.partial_fit(self.scaled_features, y_new)

            for cls_label in confirmed_path:
                if cls_label in self.anomaly_detectors:
                    self.buffers[cls_label].append(self.scaled_features)

                    stacked_buffer = np.vstack(self.buffers[cls_label])
                    if len(stacked_buffer) > self.max_buffer:
                        idx = np.random.choice(len(stacked_buffer), self.max_buffer, replace=False)
                        stacked_buffer = stacked_buffer[idx]
                        self.buffers[cls_label] = [stacked_buffer]

                    pipeline = self.anomaly_detectors[cls_label]
                    X_pca = pipeline['pca'].fit_transform(stacked_buffer)
                    pipeline['iso'].fit(X_pca)

        self.current_idx += 1
        if hasattr(self, 'human_path'): del self.human_path
        self._process_next_image()

    def _ground_truth_path(self):
        gt_row = self.gt_df.loc[self.img_path.name]
        return " -> ".join([str(x) for x in [gt_row['level_1'], gt_row['level_2'], gt_row['level_3']]
                            if str(x) not in ["None", "nan", "NaN"]])

    def _log_automated_success(self):
        log_entry = {"filename": self.img_path.name, "warning": False,
                     "model_path": " -> ".join(self.pred_path)}
        if self.is_test_mode:
            gt_path = self._ground_truth_path()
            log_entry["ground_truth"] = gt_path
            log_entry["silent_failure"] = (log_entry["model_path"] != gt_path)
        self.logs.append(log_entry)

    def _log_human_intervention(self, is_ood):
        log_entry = {"filename": self.img_path.name, "warning": True, "trigger": self.warning_details}
        log_entry["model_path"] = " -> ".join(self.pred_path)
        log_entry["human_path"] = " -> ".join(self.human_path) + (" -> NEW" if is_ood else "")
        if self.is_test_mode:
            gt_path = self._ground_truth_path()
            log_entry["ground_truth"] = gt_path
            log_entry["intercepted_error"] = (log_entry["model_path"] != gt_path)
        self.logs.append(log_entry)

    def _save_logs(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.logs).to_csv(self.log_path, index=False)
        print(f"Logs saved to {self.log_path}")

    def _save_diagnostic_maps(self, level_node):
        """Save the full probability and anomaly maps for the node that raised the warning."""
        maps = self.diagnostic_maps.get(level_node)
        if not maps: return

        save_path = self.diagnostics_dir / f"{self.img_path.stem}_{level_node}_diagnostics.npz"

        np.savez_compressed(
            save_path,
            anomaly_map=maps["anomaly_map"],
            full_prob_map=maps["full_prob_map"],
            class_names=maps["class_names"],
            warning_type=self.warning_details['type'],
            warning_val=self.warning_details['val']
        )

    def save_models(self, target_dir=None):
        """Write the in-memory (human-updated) models to disk.

        Defaults to models_dir, which overwrites the initial models; pass a
        target_dir to keep the session's models separate.
        """
        out_path = Path(target_dir) if target_dir else self.models_dir
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving updated models to {out_path}...")

        for node, model in self.classifiers.items():
            joblib.dump(model, out_path / f"classifier_{node}.joblib")

        for cls_name, pipeline in self.anomaly_detectors.items():
            joblib.dump(pipeline, out_path / f"anomaly_{cls_name}.joblib")

        print("All models saved!")
