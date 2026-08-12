"""Headless replay of a finished segmentation session.

Reruns the exact model sequence the interactive dashboard produced, but with no
GUI: instead of live human clicks it replays the stored bounding boxes from
``annotations/segmentation_case<N>/``. Use it to regenerate predictions without
a notebook, or to check that a change to the pipeline still reproduces the
published results.

    python -m hitl_sem.segmentation.replay --case 2
    python -m hitl_sem.segmentation.replay --case 2 --validate --reference-dir outputs/segmentation_case2/predictions

The dashboard's train -> save control flow is reimplemented here; all model code
(SeqLoader, boxes, models) is imported and reused unchanged.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from ..boxes import combine_datasets, get_training_data_from_crops
from ..models import predict_with_uncertainty, train_and_optimize
from .loader import SeqLoader

REPO_ROOT = Path(__file__).resolve().parents[2]

CASES = {
    1: {
        "img_dir": REPO_ROOT / "data" / "segmentation" / "case1" / "images",
        "features_dir": REPO_ROOT / "data" / "segmentation" / "case1" / "features",
        "annotation_dir": REPO_ROOT / "annotations" / "segmentation_case1",
        "output_dir": REPO_ROOT / "outputs" / "segmentation_case1" / "replay",
        "mode": "patch",
        "class_names": ["phase 1", "phase 2", "phase 3"],
    },
    2: {
        "img_dir": REPO_ROOT / "data" / "segmentation" / "case2" / "patches",
        "features_dir": REPO_ROOT / "data" / "segmentation" / "case2" / "features",
        "annotation_dir": REPO_ROOT / "annotations" / "segmentation_case2",
        "output_dir": REPO_ROOT / "outputs" / "segmentation_case2" / "replay",
        "mode": "pixel",
        "class_names": ["class 1", "class 2", "class 3", "class 4"],
    },
}

K = 1000


def load_annotation_boxes(annotation_dir, img_id):
    """Return the final human boxes for an image, or [] if it was never annotated."""
    box_path = Path(annotation_dir) / f"box_{img_id}.json"
    if not box_path.exists():
        return []
    with open(box_path) as f:
        return json.load(f)


def run_replay(img_dir, features_dir, annotation_dir, output_dir, mode, class_names, k=K):
    """Replay the whole sequence and write predictions/boxes under output_dir."""
    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    boxes_dir = output_dir / "boxes"
    pred_dir.mkdir(parents=True, exist_ok=True)
    boxes_dir.mkdir(parents=True, exist_ok=True)

    loader = SeqLoader(str(img_dir), str(features_dir), str(pred_dir), str(boxes_dir), k=k)

    for _ in range(loader.total):
        data = loader.next_image(mode=mode)
        img_id = data["img_id"]
        boxes = load_annotation_boxes(annotation_dir, img_id)

        if boxes:
            # Mirror ActiveSegmentationDashboard._on_train_predict using the final box set
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump(boxes, tf)
                tmp_box_path = tf.name

            X_train_c, y_train_c, X_train_c_original = get_training_data_from_crops(
                data, tmp_box_path, class_names=class_names
            )
            Path(tmp_box_path).unlink()

            if data.get("X_train") is None:
                X_comb, y_comb = X_train_c, y_train_c
            else:
                X_comb, y_comb = combine_datasets(
                    X_train_c, y_train_c, data["X_train"], data["y_train"]
                )

            best_model = train_and_optimize(X_comb, y_comb)
            infer = (data["features_pixel_normalized"] if mode == "pixel"
                     else data["features_normalized"])
            y_pred, probs, entropy = predict_with_uncertainty(best_model, infer)
            source = f"annotated ({len(boxes)} boxes)"
        else:
            # No annotation round: the human accepted the zero-shot prediction
            y_pred = data["y_pred"]
            probs = data["probabilities"]
            entropy = data["entropy"]
            X_train_c_original, y_train_c = None, None
            source = "zero-shot accepted"

        loader.save(y_pred, probs, entropy, X_train_c_original, y_train_c)

        shape = None if y_pred is None else np.asarray(y_pred).shape
        print(f"  [{img_id}] {source:28s} -> y_pred {shape}")

    print("\nReplay complete. Outputs in:", pred_dir)
    return pred_dir


def _load_pred(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d["y_pred"], d["probs"], d["entropy"]


def validate(replay_pred_dir, reference_dir):
    """Compare replayed predictions against a reference predictions folder."""
    print("\n=== Validation ===")
    replay_pred_dir = Path(replay_pred_dir)
    reference_dir = Path(reference_dir)

    all_ok = True
    for replay_path in sorted(replay_pred_dir.glob("*_pred.npz")):
        img_id = replay_path.stem.replace("_pred", "")
        replay_y, replay_p, replay_e = _load_pred(replay_path)

        if replay_y is None or replay_y.dtype == object:
            print(f"  [{img_id}] FAIL: replay produced no prediction")
            all_ok = False
            continue

        ref_path = reference_dir / replay_path.name
        if not ref_path.exists():
            print(f"  [{img_id}] no reference file, replay y_pred {replay_y.shape}")
            continue

        ref_y, ref_p, ref_e = _load_pred(ref_path)
        if ref_y is None or ref_y.dtype == object:
            print(f"  [{img_id}] reference holds no prediction, skipping")
            continue

        agree = float(np.mean(replay_y == ref_y)) * 100
        p_close = np.allclose(replay_p, ref_p, atol=1e-5)
        e_close = np.allclose(replay_e, ref_e, atol=1e-5)
        status = "OK" if (agree > 99.9 and p_close and e_close) else "DIFF"
        if status != "OK":
            all_ok = False
        print(f"  [{img_id}] label agree {agree:6.2f}%  probs~{p_close}  entropy~{e_close}  [{status}]")

    print("\nValidation:", "ALL OK" if all_ok else "DIFFERENCES FOUND")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Replay a segmentation session without the GUI.")
    parser.add_argument("--case", type=int, choices=sorted(CASES), required=True,
                        help="Which case study to replay.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write predictions/ and boxes/. Default: outputs/segmentation_case<N>/replay")
    parser.add_argument("--validate", action="store_true",
                        help="After replaying, compare against --reference-dir.")
    parser.add_argument("--reference-dir", type=str, default=None,
                        help="Predictions folder to validate against. Default: outputs/segmentation_case<N>/predictions")
    args = parser.parse_args()

    cfg = CASES[args.case]
    output_dir = Path(args.output_dir) if args.output_dir else cfg["output_dir"]

    pred_dir = run_replay(
        cfg["img_dir"], cfg["features_dir"], cfg["annotation_dir"], output_dir,
        cfg["mode"], cfg["class_names"],
    )

    if args.validate:
        reference = (Path(args.reference_dir) if args.reference_dir
                     else REPO_ROOT / "outputs" / f"segmentation_case{args.case}" / "predictions")
        validate(pred_dir, reference)


if __name__ == "__main__":
    main()
