# DINOv3 weights (placeholder)

This folder must contain the pretrained DINOv3 checkpoint. The weights are
license-gated by Meta and are not redistributed here.

1. Go to <https://github.com/facebookresearch/dinov3> and follow the link to
   request access to the pretrained backbones.
2. Download the ViT-L/16 checkpoint pretrained on LVD-1689M:

   ```
   dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
   ```

3. Place the file in this folder, so that the path is

   ```
   dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
   ```

That exact filename is the default in `hitl_sem/features.py`. To use a different
checkpoint, pass `weights_path=` (and a matching `model_name=`) to
`Dinov3FeatureExtractor`.
