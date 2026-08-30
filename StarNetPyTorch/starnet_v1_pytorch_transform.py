import argparse
from pathlib import Path

from starnet_v1_pytorch import StarNet


def main():

    input_default  = "rgb_test5.tif"
    output_default = "rgb_test5_starless.tif"
    weights_default = "../starnet_weights2/weights"
    mode_default = "RGB" 

    parser = argparse.ArgumentParser(description="PyTorch version of the TF2 StarNet transform demo")
    parser.add_argument("input", nargs="?", default=input_default, help="Input TIFF image")
    parser.add_argument(
        "output",
        nargs="?",
        default=output_default,
        help="Output TIFF image",
    )
    parser.add_argument(
        "--weights",
        default=weights_default,
        help="Weight prefix or a direct .pth file path",
    )
    parser.add_argument("--mode", choices=("RGB", "Greyscale"), default=mode_default)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--device", default="cuda", help="Torch device such as cuda, cuda:0, or cpu")
    args = parser.parse_args()

    starnet = StarNet(
        mode=args.mode,
        window_size=args.window_size,
        stride=args.stride,
        device=args.device,
    )
    weights_path = starnet.load_model(args.weights)
    print("Loaded weights: %s" % weights_path)

    output_path = starnet.transform(args.input, args.output)
    print("Saved result: %s" % Path(output_path).resolve())


if __name__ == "__main__":
    main()
