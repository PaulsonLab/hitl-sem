# DINOv3 (placeholder)

This folder must contain a local clone of Meta's DINOv3 repository. It is not
redistributed here. From the repository root:

```bash
git clone https://github.com/facebookresearch/dinov3.git dinov3
```

The clone is used as a local `torch.hub` source, so `dinov3/hubconf.py` must sit
directly inside this folder:

```
dinov3/
├── hubconf.py
├── dinov3/
└── ...
```

`hitl_sem.features.Dinov3FeatureExtractor` then loads the backbone with

```python
torch.hub.load("dinov3", "dinov3_vitl16", source="local", weights=<checkpoint>)
```

The checkpoint itself goes in `../dinov3_weights/` — see the README there.

DINOv3 is distributed under its own license; please read
`dinov3/LICENSE.md` after cloning.
