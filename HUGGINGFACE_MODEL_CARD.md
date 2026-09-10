---
library_name: pytorch
pipeline_tag: image-to-image
tags:
- astronomy
- astrophotography
- image-denoising
- star-removal
- pytorch
---

# AstroStarDetect

PyTorch checkpoints and deployment code for astronomical image denoising and star removal.

Source repository: <https://github.com/CHMimilanlan/AstroColorProcess>

## Included models

| Model | Checkpoint | Purpose |
| --- | --- | --- |
| Cosmic Clarity AI 3.6 wrapper | `StarDenoise/models/deep_denoise_cnn_AI3_6.pth` | Astronomical image denoising |
| StarNet RGB generator | `StarNetPyTorch/models/weights_G_RGB.pth` | Star removal from RGB TIFF images |
| StarNet greyscale generator | `StarNetPyTorch/models/weights_G_Greyscale.pth` | Star removal from greyscale TIFF images |

## Quick start

```bash
git clone https://github.com/CHMimilanlan/AstroColorProcess.git
cd AstroColorProcess
python -m pip install -r requirements.txt
python download_models.py
```

Denoise an image:

```bash
python StarDenoise/denoise.py input.tif output.tif --device auto --strength 0.85
```

Remove stars from an RGB TIFF image:

```bash
python StarNetPyTorch/starnet_v1_pytorch_transform.py input.tif output_starless.tif --mode RGB
```

The download helper places every checkpoint at the default location expected by the inference scripts.
See the [GitHub README](https://github.com/CHMimilanlan/AstroColorProcess#readme) for Windows/Conda instructions, parameters, checksums, offline deployment, and licensing notes.

## Attribution and licensing

`StarDenoise` wraps the Cosmic Clarity AI 3.6 PyTorch model and retains its upstream MIT license text.
`StarNetPyTorch` is a PyTorch translation and inference example for the StarNet architecture.
Third-party checkpoints and examples remain subject to their original upstream terms; verify those terms before commercial use.
