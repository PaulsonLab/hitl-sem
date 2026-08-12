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

That exact filename is the default in `hitl_sem/features.py`.

## Using a checkpoint somewhere else

If you already have the weights, leave this folder empty and point the package at
them:

```bash
export DINOV3_WEIGHTS=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

The lookup order is the `weights_path=` argument, then `$DINOV3_WEIGHTS`, then
this folder. To use a different backbone, pass a matching `model_name=` to
`Dinov3FeatureExtractor` as well.
