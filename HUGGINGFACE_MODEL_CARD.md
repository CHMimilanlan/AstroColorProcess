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

| Model | Parameters | Parameter count (B) | Checkpoint size | Purpose |
| --- | ---: | ---: | ---: | --- |
| Cosmic Clarity AI 3.6 wrapper | 2,945,283 | 0.002945B | 33.77 MiB | Astronomical image denoising |
| StarNet RGB generator | 54,420,483 | 0.054420B | 207.67 MiB | Star removal from RGB TIFF images |
| StarNet greyscale generator | 54,414,337 | 0.054414B | 207.64 MiB | Star removal from greyscale TIFF images |
| **All checkpoints** | **111,780,103** | **0.111780B** | **449.08 MiB** | Stored models; RGB and greyscale variants are not normally loaded together |

FP32 parameter memory is 11.24 MiB for the denoiser, 207.60 MiB for StarNet RGB, and
207.57 MiB for StarNet greyscale. At batch 1 and 256×256 input, profiled FP32 tensor-memory
peaks are approximately 361.67 MiB, 356.33 MiB, and 355.31 MiB respectively. These are
compute-graph estimates and exclude the CUDA context, cuDNN workspace, and allocator cache.
The measurements were collected in a CPU-only PyTorch environment, so they are not direct CUDA
`max_memory_allocated` measurements. See the GitHub README for sizing methodology,
512×512 estimates, and recommended GPU capacities.

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
