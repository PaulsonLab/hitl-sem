"""Human-in-the-loop segmentation of a sequence of SEM images.

    loader     SeqLoader — streams images, loads or extracts features, carries
               annotations forward from earlier images as a zero-shot prediction
    dashboard  ActiveSegmentationDashboard — the ipywidgets annotation UI
    replay     headless re-run of a finished session from its saved boxes
"""

from .loader import SeqLoader, load_and_concat_npz

__all__ = ["SeqLoader", "load_and_concat_npz"]
