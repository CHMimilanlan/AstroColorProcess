from pathlib import Path

import numpy as np
import tifffile as tiff
import torch

from model import create_generator


WINDOW_SIZE = 256
DEFAULT_STRIDE = 64


def _read_tiff_image(image_name):
    data = tiff.imread(image_name)
    if len(data.shape) > 3:
        layer = input("Tiff has %d layers, please enter layer to process: " % data.shape[0])
        data = data[int(layer)]

    input_dtype = data.dtype
    if input_dtype == np.dtype("uint16"):
        image = (data / 65535.0).astype("float32")
    elif input_dtype == np.dtype("uint8"):
        image = (data / 255.0).astype("float32")
    else:
        raise ValueError("Unknown image dtype: %s" % data.dtype)

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Input image must be an RGB/RGBA TIFF image")

    if image.shape[2] == 4:
        print("Input image has 4 channels. Removing Alpha-Channel")
        image = image[:, :, [0, 1, 2]]

    return image, input_dtype


def _default_weights_path():
    candidates = [
        Path.cwd() / "starnet_generator.pth",
        Path(__file__).resolve().parent / "starnet_generator.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _save_tiff(path, array):
    if hasattr(tiff, "imwrite"):
        tiff.imwrite(path, array)
    else:
        tiff.imsave(path, array)


def transform(imageName, stride=DEFAULT_STRIDE, weights_path=None, device=None):
    weights_path = Path(weights_path) if weights_path else _default_weights_path()
    if not weights_path.exists():
        raise FileNotFoundError(
            "PyTorch weights were not found at %s. Convert a TensorFlow checkpoint first "
            "with: python convert_tf_checkpoint.py --checkpoint ../starnet/model.ckpt "
            "--output starnet_generator.pth" % weights_path
        )

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Loading PyTorch model from %s..." % weights_path)
    net = create_generator(weights_path=weights_path, device=device)
    print("Done!")

    print("Opening input image...")
    image, input_dtype = _read_tiff_image(imageName)

    offset = int((WINDOW_SIZE - stride) / 2)
    h, w, _ = image.shape
    ith = int(h / stride) + 1
    itw = int(w / stride) + 1
    dh = ith * stride - h
    dw = itw * stride - w

    image = np.concatenate((image, image[(h - dh):, :, :]), axis=0)
    image = np.concatenate((image, image[:, (w - dw):, :]), axis=1)

    h, w, _ = image.shape
    image = np.concatenate((image, image[(h - offset):, :, :]), axis=0)
    image = np.concatenate((image[:offset, :, :], image), axis=0)
    image = np.concatenate((image, image[:, (w - offset):, :]), axis=1)
    image = np.concatenate((image[:, :offset, :], image), axis=1)

    output = np.copy(image)

    with torch.inference_mode():
        for i in range(ith):
            for j in range(itw):
                progress = int((itw * i + j + 1) * 100 / (ith * itw))
                print("Transforming input image... %d%%\r" % progress, end="")

                x = stride * i
                y = stride * j
                patch = image[x:x + WINDOW_SIZE, y:y + WINDOW_SIZE, :]
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(device)

                result = net(patch_tensor).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                output[
                    x + offset:x + stride + offset,
                    y + offset:y + stride + offset,
                    :,
                ] = result[offset:stride + offset, offset:stride + offset, :]

    print("Transforming input image... Done!")

    output = np.clip(output, 0, 1)
    output = output[offset:-(offset + dh), offset:-(offset + dw), :]

    output_path = Path(str(imageName) + "_starless.tif")
    print("Saving output image...")
    if input_dtype == np.dtype("uint8"):
        _save_tiff(output_path, (output * 255).astype("uint8"))
    else:
        _save_tiff(output_path, (output * 65535).astype("uint16"))
    print("Done!")
    return output_path
