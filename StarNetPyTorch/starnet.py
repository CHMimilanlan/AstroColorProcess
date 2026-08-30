import argparse
import time

import transform as transform_module


epochs = 10000
batch = 1
steps = 1000
output_freq = 100
verbose = True
images = True
log_freq = 50
gen_plots = True
learning_rates = [0.000002, 0.0000005]
stride = 64


def main():
    parser = argparse.ArgumentParser(description="PyTorch StarNet transform entry point")
    subparsers = parser.add_subparsers(dest="command")

    transform_parser = subparsers.add_parser("transform", help="Remove stars from a TIFF image")
    transform_parser.add_argument("image", help="Input TIFF image")
    transform_parser.add_argument("--stride", type=int, default=stride, help="Sliding window stride")
    transform_parser.add_argument("--weights", default=None, help="Path to starnet_generator.pth")
    transform_parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu")

    args = parser.parse_args()
    if args.command == "transform":
        start = time.time()
        transform_module.transform(
            imageName=args.image,
            stride=args.stride,
            weights_path=args.weights,
            device=args.device,
        )
        stop = time.time()
        minutes = float((stop - start) / 60)
        if minutes > 60.0:
            print("Total time taken: %.1f hours" % (minutes / 60))
        else:
            print("Total time taken: %.1f minutes" % minutes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
