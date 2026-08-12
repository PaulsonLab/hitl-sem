"""Human-in-the-loop segmentation and classification of SEM micrographs.

Subpackages
    segmentation   interactive patch/pixel-level segmentation of a sequence of images
    classification hierarchical image-level classification with active learning

Shared modules
    features  DINOv3 patch-embedding extraction
    models    logistic-regression pipeline, tuning, uncertainty
    boxes     bounding boxes -> feature-grid crops -> training arrays
    viz       prediction/entropy overlays used by the segmentation dashboard
"""

__version__ = "1.0.0"
