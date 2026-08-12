"""DINOv3 patch-embedding extraction.

The backbone is loaded from a local clone of the DINOv3 repository (see
``dinov3/README.md``) together with a downloaded checkpoint (see
``dinov3_weights/README.md``); neither is redistributed here.

Both are located in this order: an explicit ``repo_dir=`` / ``weights_path=``
argument, then the ``DINOV3_REPO`` / ``DINOV3_WEIGHTS`` environment variables,
then ``dinov3/`` and ``dinov3_weights/`` beside this package. Set the environment
variables when using this package away from a repository checkout.

Run as a script to embed a folder of images:

    python -m hitl_sem.features --data-folder data/segmentation/case2/patches \
                                --output-folder data/segmentation/case2/features

Files are named ``<image stem>_<image height>.npz``, which is the naming the
segmentation loader and the classification pipeline both look for.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_NAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def resolve_repo_dir(repo_dir=None):
    """Locate the DINOv3 clone: explicit argument, then $DINOV3_REPO, then ./dinov3."""
    if repo_dir is not None:
        return Path(repo_dir)
    env = os.environ.get("DINOV3_REPO")
    return Path(env) if env else REPO_ROOT / "dinov3"


def resolve_weights(weights_path=None):
    """Locate the checkpoint: explicit argument, then $DINOV3_WEIGHTS, then ./dinov3_weights."""
    if weights_path is not None:
        return Path(weights_path)
    env = os.environ.get("DINOV3_WEIGHTS")
    return Path(env) if env else REPO_ROOT / "dinov3_weights" / DEFAULT_WEIGHTS_NAME


class Dinov3FeatureExtractor:
    """DINOv3 feature extractor for grayscale SEM images."""

    def __init__(self, model_name="dinov3_vitl16", repo_dir=None, weights_path=None):
        repo_dir = resolve_repo_dir(repo_dir)
        weights_path = resolve_weights(weights_path)

        if not (repo_dir / "hubconf.py").exists():
            raise FileNotFoundError(
                f"No DINOv3 clone at {repo_dir} (expected {repo_dir / 'hubconf.py'}). "
                "Clone https://github.com/facebookresearch/dinov3 there, or set "
                "$DINOV3_REPO, or pass repo_dir=."
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No DINOv3 checkpoint at {weights_path}. Download it from the DINOv3 "
                "repository, or set $DINOV3_WEIGHTS, or pass weights_path=."
            )

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.patch_size = 16
        self.imagenet_mean = (0.485, 0.456, 0.406)
        self.imagenet_std = (0.229, 0.224, 0.225)

        model_to_num_layers = {"dinov3_vitl16": 24}
        self.n_layers = model_to_num_layers.get(self.model_name, 24)

        print(f"Loading DINOv3 model: {self.model_name} on {self.device}")
        self.model = torch.hub.load(
            str(repo_dir), self.model_name, source='local', weights=str(weights_path)
        ).to(self.device)
        self.model.eval()

    def _read_image(self, file_path: str) -> np.ndarray:
        """Read a PNG into a 2D array."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext != '.png':
            raise ValueError(f"Unsupported file type: {ext}. Only .png is supported.")

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to read PNG image: {file_path}")

        return img

    def _to_pil_rgb(self, img_array: np.ndarray) -> Image.Image:
        """Expand a grayscale array to 8-bit RGB, as the backbone expects."""
        if len(img_array.shape) != 2:
            raise ValueError("Input array must be 2D")

        if img_array.dtype == np.uint16:
            img_8bit = (img_array / 256).astype(np.uint8)
        elif img_array.dtype == np.uint8:
            img_8bit = img_array
        else:
            raise ValueError(f"Unsupported dtype {img_array.dtype}. Expected uint8 or uint16.")

        img_rgb = np.stack([img_8bit] * 3, axis=-1)
        return Image.fromarray(img_rgb)

    def _resize_transform(self, image: Image.Image, target_height: int) -> torch.Tensor:
        """Resize to a whole number of patches while preserving aspect ratio."""
        w, h = image.size
        h_patches = int(target_height / self.patch_size)
        w_patches = int((w * target_height) / (h * self.patch_size))
        return TF.to_tensor(TF.resize(image, (h_patches * self.patch_size, w_patches * self.patch_size)))

    def extract(self, image_path: str, target_height: int, mode: str = "dense") -> dict:
        """Extract features from one image.

        mode: 'dense' (per-patch, key 'X'), 'cls' (global token, key 'cls'), or 'both'.
        """
        img_array = self._read_image(image_path)
        pil_img = self._to_pil_rgb(img_array)
        img_tensor = self._resize_transform(pil_img, target_height)
        img_tensor = TF.normalize(img_tensor, mean=self.imagenet_mean, std=self.imagenet_std)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        save_dict = {}
        ret_cls = mode in ["cls", "both"]

        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float32):
            layers_output = self.model.get_intermediate_layers(
                img_tensor, n=range(self.n_layers),
                reshape=True, norm=True, return_class_token=ret_cls
            )

        if ret_cls:
            patch_feats, cls_token = layers_output[-1]
        else:
            patch_feats = layers_output[-1]
            cls_token = None

        if mode in ["dense", "both"]:
            dim = patch_feats.shape[1]
            dense_vec = patch_feats.squeeze().view(dim, -1).permute(1, 0).detach().cpu().numpy()
            save_dict["X"] = dense_vec

        if mode in ["cls", "both"] and cls_token is not None:
            save_dict["cls"] = cls_token.detach().cpu().numpy()

        del layers_output, img_tensor, patch_feats
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return save_dict


def print_gpu_utilization():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   [GPU] Allocated: {used:.2f}GB | Reserved: {reserved:.2f}GB")


def extract_folder(data_folder, output_folder, image_size=None, mode="dense",
                   recursive=False, overwrite=False, extractor=None):
    """Embed every image in a folder, skipping images already embedded.

    image_size : target height in pixels, or None to use each image's own height.
    recursive  : also walk subfolders, mirroring their structure in the output.
    """
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    image_files = sorted(
        p for p in data_folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_files:
        print(f"No images found in {data_folder}")
        return

    print(f"Found {len(image_files)} images in {data_folder} | Mode: {mode}")

    extracted, skipped = 0, 0

    for i, img_path in enumerate(image_files, 1):
        img_array = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            print(f"[{i}/{len(image_files)}] Could not read {img_path.name}. Skipping.")
            continue

        target_height = image_size if image_size else img_array.shape[0]

        out_dir = output_folder / img_path.parent.relative_to(data_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{img_path.stem}_{target_height}.npz"

        if out_path.exists() and not overwrite:
            print(f"[{i}/{len(image_files)}] Features exist for {img_path.name}. Skipping.")
            skipped += 1
            continue

        print(f"[{i}/{len(image_files)}] Extracting {img_path.name} (height {target_height})...")

        if extractor is None:
            extractor = Dinov3FeatureExtractor()

        save_dict = extractor.extract(str(img_path), target_height=target_height, mode=mode)
        np.savez_compressed(out_path, **save_dict)

        print(f"    saved {list(save_dict.keys())} -> {out_path}")
        print_gpu_utilization()
        extracted += 1

    print(f"\nDone. Extracted: {extracted} | Skipped: {skipped}")

    if extractor is not None and torch.cuda.is_available():
        del extractor
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv3 features for a folder of images.")
    parser.add_argument("--data-folder", type=str, required=True, help="Input image folder.")
    parser.add_argument("--output-folder", type=str, required=True, help="Output folder for .npz files.")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Target image height. Default: each image's own height.")
    parser.add_argument("--mode", type=str, default="dense", choices=["dense", "cls", "both"],
                        help="'dense' (patch features), 'cls' (global token), or 'both'.")
    parser.add_argument("--recursive", action="store_true",
                        help="Walk subfolders and mirror them in the output (class-foldered data).")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract files that already exist.")
    args = parser.parse_args()

    extract_folder(
        args.data_folder, args.output_folder,
        image_size=args.image_size, mode=args.mode,
        recursive=args.recursive, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
