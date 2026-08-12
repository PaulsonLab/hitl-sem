"""Live plots for the segmentation dashboard.

Both helpers accept predictions on either grid: one value per pixel (pixel mode)
or one value per 16x16 patch (patch mode), inferred from the array length.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_prediction_overlay_ax(ax, img, y_pred, class_names=None, alpha=0.5):
    """Draw the predicted class map over the image, with a class legend."""
    ax.clear()
    h, w = img.shape[:2]
    y_pred_arr = np.asarray(y_pred)

    if len(y_pred_arr) == h * w:
        mask_large = y_pred_arr.reshape((h, w)).astype(np.uint8)
    else:
        patch_h, patch_w = h // 16, w // 16
        mask_small = y_pred_arr.reshape((patch_h, patch_w)).astype(np.uint8)
        mask_large = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    ax.imshow(img, cmap='gray' if img.ndim == 2 else None, aspect='equal')

    cmap = plt.get_cmap('tab20')
    ax.imshow(mask_large, cmap=cmap, alpha=alpha, aspect='equal', interpolation='nearest', vmin=0, vmax=19)
    ax.set_title("Predicted Segmentation", pad=10)
    ax.axis('off')

    if class_names:
        legend_elements = [
            Patch(facecolor=cmap(i % 20), edgecolor='w', label=name)
            for i, name in enumerate(class_names)
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)


def plot_entropy_ax(ax, entropy, img_shape):
    """Draw the predictive entropy map."""
    ax.clear()
    h, w = img_shape[:2]
    ent_arr = np.asarray(entropy).astype(np.float32)

    if len(ent_arr) == h * w:
        ent_large = ent_arr.reshape((h, w))
    else:
        patch_h, patch_w = h // 16, w // 16
        ent_small = ent_arr.reshape((patch_h, patch_w))
        ent_large = cv2.resize(ent_small, (w, h), interpolation=cv2.INTER_LINEAR)

    ax.imshow(ent_large, cmap='inferno', aspect='equal')
    ax.set_title("Predictive Entropy", pad=10)
    ax.axis('off')
