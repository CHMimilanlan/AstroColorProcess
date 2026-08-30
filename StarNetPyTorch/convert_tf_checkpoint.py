import argparse
from pathlib import Path

import torch

from model import Generator


def _read_tf_checkpoint(checkpoint_path):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required only for checkpoint conversion.") from exc

    return tf.train.load_checkpoint(str(checkpoint_path))


def _copy_conv(reader, state, tf_name, torch_name):
    state[torch_name + ".weight"] = torch.from_numpy(reader.get_tensor(tf_name + "/kernel")).permute(3, 2, 0, 1)
    state[torch_name + ".bias"] = torch.from_numpy(reader.get_tensor(tf_name + "/bias"))


def _copy_deconv(reader, state, tf_name, torch_name):
    state[torch_name + ".weight"] = torch.from_numpy(reader.get_tensor(tf_name + "/kernel")).permute(3, 2, 0, 1)
    state[torch_name + ".bias"] = torch.from_numpy(reader.get_tensor(tf_name + "/bias"))


def _copy_bn(reader, state, tf_name, torch_name):
    state[torch_name + ".weight"] = torch.from_numpy(reader.get_tensor(tf_name + "/gamma"))
    state[torch_name + ".bias"] = torch.from_numpy(reader.get_tensor(tf_name + "/beta"))
    state[torch_name + ".running_mean"] = torch.from_numpy(reader.get_tensor(tf_name + "/moving_mean"))
    state[torch_name + ".running_var"] = torch.from_numpy(reader.get_tensor(tf_name + "/moving_variance"))


def convert(checkpoint_path, output_path):
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    reader = _read_tf_checkpoint(checkpoint_path)
    generator = Generator()
    state = generator.state_dict()

    _copy_conv(reader, state, "generator/g_conv0/conv2d", "down_convs.0")
    for i in range(1, 8):
        _copy_conv(reader, state, "generator/g_conv%d/conv2d" % i, "down_convs.%d" % i)
        _copy_bn(
            reader,
            state,
            "generator/g_conv%d/batch_normalization" % i,
            "down_bns.%d" % (i - 1),
        )

    for i in range(8):
        _copy_deconv(reader, state, "generator/g_deconv%d/conv2d_transpose" % i, "up_convs.%d" % i)
        if i < 7:
            _copy_bn(
                reader,
                state,
                "generator/g_deconv%d/batch_normalization" % i,
                "up_bns.%d" % i,
            )

    generator.load_state_dict(state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": generator.state_dict()}, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert StarNet TF1 generator checkpoint to PyTorch")
    parser.add_argument("--checkpoint", default="model.ckpt", help="TensorFlow checkpoint prefix")
    parser.add_argument("--output", default="starnet_generator.pth", help="Output PyTorch weights")
    args = parser.parse_args()

    output_path = convert(args.checkpoint, args.output)
    print("Saved PyTorch weights to %s" % output_path)


if __name__ == "__main__":
    main()
