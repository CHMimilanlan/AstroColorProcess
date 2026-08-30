import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from model import Generator


def _decode_names(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _read_layer_weights(layer_group):
    weight_names = _decode_names(layer_group.attrs.get("weight_names", []))
    weights = {}
    for weight_name in weight_names:
        short_name = weight_name.rsplit("/", 1)[-1].split(":", 1)[0]
        try:
            weights[short_name] = np.asarray(layer_group[weight_name])
        except KeyError:
            relative_name = weight_name.split("/", 1)[-1]
            weights[short_name] = np.asarray(layer_group[relative_name])
    return weights


def _load_keras_layers(h5_path):
    layers = []
    with h5py.File(h5_path, "r") as handle:
        root = handle
        if "layer_names" not in root.attrs and "model_weights" in root:
            root = root["model_weights"]

        layer_names = _decode_names(root.attrs.get("layer_names", []))
        if not layer_names:
            raise ValueError(
                "Unsupported H5 layout: no Keras layer_names attribute was found. "
                "Use a TensorFlow/Keras save_weights H5 file."
            )

        for layer_name in layer_names:
            layer_group = root[layer_name]
            weights = _read_layer_weights(layer_group)
            if weights:
                layers.append((layer_name, weights))
    return layers


def _copy_conv(state, torch_name, weights, transpose=False):
    required = {"kernel", "bias"}
    if not required.issubset(weights):
        raise ValueError("Layer %s is missing kernel or bias" % torch_name)

    kernel = torch.from_numpy(weights["kernel"]).permute(3, 2, 0, 1).contiguous()
    if transpose:
        state[torch_name + ".weight"] = kernel
    else:
        state[torch_name + ".weight"] = kernel
    state[torch_name + ".bias"] = torch.from_numpy(weights["bias"])


def _copy_batch_norm(state, torch_name, weights):
    names = {
        "weight": "gamma",
        "bias": "beta",
        "running_mean": "moving_mean",
        "running_var": "moving_variance",
    }
    missing = [source for source in names.values() if source not in weights]
    if missing:
        raise ValueError("Layer %s is missing: %s" % (torch_name, ", ".join(missing)))
    for target, source in names.items():
        state[torch_name + "." + target] = torch.from_numpy(weights[source])


def convert(input_path, output_path, mode):
    input_channels = 3 if mode == "RGB" else 1
    layers = _load_keras_layers(input_path)

    conv_layers = [weights for name, weights in layers if name.startswith("conv2d") and "transpose" not in name]
    deconv_layers = [weights for name, weights in layers if name.startswith("conv2d_transpose")]
    batch_norm_layers = [weights for name, weights in layers if name.startswith("batch_normalization")]

    if len(conv_layers) != 8 or len(deconv_layers) != 8 or len(batch_norm_layers) != 14:
        raise ValueError(
            "Unexpected generator layout: found %d Conv2D, %d Conv2DTranspose, and "
            "%d BatchNormalization layers; expected 8, 8, and 14."
            % (len(conv_layers), len(deconv_layers), len(batch_norm_layers))
        )

    generator = Generator(input_channels=input_channels, batch_norm_eps=1e-3)
    state = generator.state_dict()

    for index, weights in enumerate(conv_layers):
        _copy_conv(state, "down_convs.%d" % index, weights)
    for index, weights in enumerate(deconv_layers):
        _copy_conv(state, "up_convs.%d" % index, weights, transpose=True)
    for index, weights in enumerate(batch_norm_layers[:7]):
        _copy_batch_norm(state, "down_bns.%d" % index, weights)
    for index, weights in enumerate(batch_norm_layers[7:]):
        _copy_batch_norm(state, "up_bns.%d" % index, weights)

    generator.load_state_dict(state)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": generator.state_dict(),
            "mode": mode,
            "source": str(Path(input_path)),
        },
        output_path,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert StarNet TF2 Keras generator H5 weights")
    parser.add_argument("input", help="Keras generator H5 file, such as weights_G_Greyscale.h5")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output .pth file; defaults to the input filename with a .pth suffix",
    )
    parser.add_argument("--mode", choices=("RGB", "Greyscale"), default="Greyscale")
    args = parser.parse_args()

    output = args.output or str(Path(args.input).with_suffix(".pth"))
    output_path = convert(args.input, output, args.mode)
    print("Saved PyTorch weights: %s" % output_path)


if __name__ == "__main__":
    main()
