import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F

from model import Discriminator, Generator


class StarNet:
    def __init__(
        self,
        mode,
        window_size=512,
        stride=256,
        lr=1e-4,
        train_folder="./train/",
        batch_size=1,
        device=None,
    ):
        if mode not in ("RGB", "Greyscale"):
            raise ValueError("Mode should be either RGB or Greyscale")
        if window_size % 256 != 0:
            raise ValueError("window_size must be divisible by 256")
        if stride <= 0 or stride > window_size:
            raise ValueError("stride must be greater than zero and no larger than window_size")
        if (window_size - stride) % 2 != 0:
            raise ValueError("window_size - stride must be even")

        self.mode = mode
        self.input_channels = 3 if mode == "RGB" else 1
        self.window_size = window_size
        self.stride = stride
        self.lr = lr
        self.train_folder = Path(train_folder)
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.history = {}
        self._ema = 0.9999
        self.weights = []
        self.original = []
        self.starless = []
        self.iters_per_epoch = 0

        self.G = None
        self.D = None
        self.gen_optimizer = None
        self.dis_optimizer = None

    def __str__(self):
        return "StarNet PyTorch instance"

    def load_training_dataset(self):
        original_dir = self.train_folder / "original"
        starless_dir = self.train_folder / "starless"
        original_files = sorted(path.name for path in original_dir.glob("*.tif"))
        starless_files = sorted(path.name for path in starless_dir.glob("*.tif"))

        if not original_files or not starless_files:
            raise ValueError("No training data found in %s" % self.train_folder)
        if original_files != starless_files:
            raise ValueError(
                "Corresponding names in the original and starless folders must be equal"
            )

        print("Total training images found: %d" % len(original_files))
        self.original = [tiff.imread(original_dir / name) for name in original_files]
        self.starless = [tiff.imread(starless_dir / name) for name in starless_files]
        self.weights = []
        total_pixels = 0

        for name, original, starless in zip(original_files, self.original, self.starless):
            if original.shape != starless.shape:
                raise ValueError("Image sizes are not equal for %s" % name)
            if original.shape[0] < self.window_size or original.shape[1] < self.window_size:
                raise ValueError(
                    "%s is smaller than the configured window size %d"
                    % (name, self.window_size)
                )
            pixels = original.shape[0] * original.shape[1]
            total_pixels += pixels
            self.weights.append(pixels)

        self.iters_per_epoch = max(1, total_pixels // (self.window_size * self.window_size))
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.weights /= self.weights.sum()

        print("Total size of training images: %.2f MP" % (total_pixels / 1e6))
        print("One epoch is set to %d iterations" % self.iters_per_epoch)
        print("Training dataset has been successfully loaded!")

    def _generator(self, m=64):
        if m != 64:
            raise ValueError("The translated architecture supports m=64")
        return Generator(input_channels=self.input_channels, batch_norm_eps=1e-3)

    def _discriminator(self):
        return Discriminator(input_channels=self.input_channels, batch_norm_eps=1e-3)

    def _weight_paths(self, weights):
        path = Path(weights)
        
        if path.suffix.lower() in (".pth", ".pt"):
            return path, None
        return (
            Path(str(path) + "_G_" + self.mode + ".pth"),
            Path(str(path) + "_D_" + self.mode + ".pth"),
        )

    @staticmethod
    def _load_state(module, path, device):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        module.load_state_dict(state_dict)

    def load_model(self, weights=None, history=None):
        self.G = self._generator().to(self.device)
        self.D = self._discriminator().to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        self.dis_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.lr / 4)

        if weights:
            generator_path, discriminator_path = self._weight_paths(weights)
            if not generator_path.exists():
                raise FileNotFoundError("Generator weights were not found at %s" % generator_path)
            self._load_state(self.G, generator_path, self.device)

            if discriminator_path and discriminator_path.exists():
                self._load_state(self.D, discriminator_path, self.device)
            elif discriminator_path:
                print("Discriminator weights were not found; using initialized weights.")

        if history:
            history_path = Path(str(history) + "_" + self.mode + ".pkl")
            with history_path.open("rb") as handle:
                self.history = pickle.load(handle)

        self.G.eval()
        self.D.eval()
        return self.G

    def initialize_model(self):
        return self.load_model()

    @staticmethod
    def _ramp(x):
        if torch.is_tensor(x):
            return torch.clamp(x, 0.0, 1.0)
        return np.clip(x, 0.0, 1.0)

    @staticmethod
    def _image_scale(image):
        if image.dtype == np.uint8:
            return 255.0
        if image.dtype == np.uint16:
            return 65535.0
        if np.issubdtype(image.dtype, np.floating):
            return 1.0
        raise ValueError("Unsupported training image dtype: %s" % image.dtype)

    def _get_sample(self, r, h, w, type):
        if type not in ("original", "starless"):
            raise ValueError("type must be original or starless")
        source = self.original if type == "original" else self.starless
        sample = source[r][h:h + self.window_size, w:w + self.window_size]
        return sample.astype(np.float32) / self._image_scale(sample)

    def _augmentator(self, original, starless):
        original = np.array(original, copy=True)
        starless = np.array(starless, copy=True)

        if np.random.rand() < 0.5:
            original = np.flip(original, axis=1)
            starless = np.flip(starless, axis=1)
        if np.random.rand() < 0.5:
            original = np.flip(original, axis=0)
            starless = np.flip(starless, axis=0)
        if np.random.rand() < 0.5:
            rotations = np.random.randint(1, 4)
            original = np.rot90(original, rotations, axes=(1, 0))
            starless = np.rot90(starless, rotations, axes=(1, 0))

        if self.mode == "RGB":
            if original.ndim != 3 or original.shape[2] < 3:
                raise ValueError("RGB training images must contain at least three channels")
            original = original[:, :, :3]
            starless = starless[:, :, :3]
            if np.random.rand() < 0.7:
                channel = np.random.randint(3)
                minimum = min(float(original.min()), float(starless.min()))
                offset = np.random.rand() * 0.25 - np.random.rand() * minimum
                original[:, :, channel] += offset * (1.0 - original[:, :, channel])
                starless[:, :, channel] += offset * (1.0 - starless[:, :, channel])
            if np.random.rand() < 0.7:
                sequence = np.random.permutation(3)
                original = original[:, :, sequence]
                starless = starless[:, :, sequence]
        else:
            if np.random.rand() < 0.7:
                minimum = min(float(original.min()), float(starless.min()))
                offset = np.random.rand() * 0.25 - np.random.rand() * minimum
                original += offset * (1.0 - original)
                starless += offset * (1.0 - starless)
            if original.ndim == 3:
                channel = np.random.randint(min(3, original.shape[2]))
                original = original[:, :, channel]
                starless = starless[:, :, channel]
            original = original[:, :, None]
            starless = starless[:, :, None]

        return (
            np.ascontiguousarray(np.clip(original, 0.0, 1.0)),
            np.ascontiguousarray(np.clip(starless, 0.0, 1.0)),
        )

    def generate_input(self, iterations=1, augmentation=False):
        if not self.original:
            raise RuntimeError("Training dataset was not loaded")

        original_batch = None
        starless_batch = None
        for _ in range(iterations):
            original_batch = np.zeros(
                (
                    self.batch_size,
                    self.window_size,
                    self.window_size,
                    self.input_channels,
                ),
                dtype=np.float32,
            )
            starless_batch = np.zeros_like(original_batch)

            for index in range(self.batch_size):
                image_index = int(
                    np.random.choice(len(self.original), p=np.asarray(self.weights))
                )
                image_h, image_w = self.original[image_index].shape[:2]
                top = np.random.randint(0, image_h - self.window_size + 1)
                left = np.random.randint(0, image_w - self.window_size + 1)
                original = self._get_sample(image_index, top, left, "original")
                starless = self._get_sample(image_index, top, left, "starless")

                if augmentation:
                    original, starless = self._augmentator(original, starless)
                elif self.mode == "Greyscale":
                    if original.ndim == 3:
                        channel = np.random.randint(min(3, original.shape[2]))
                        original = original[:, :, channel]
                        starless = starless[:, :, channel]
                    original = original[:, :, None]
                    starless = starless[:, :, None]
                else:
                    original = original[:, :, :3]
                    starless = starless[:, :, :3]

                original_batch[index] = original
                starless_batch[index] = starless

        return original_batch, starless_batch

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _losses(self, original, target):
        generated = self.G(original)
        real_outputs = self.D(target)
        fake_outputs = self.D(generated)

        real_features = real_outputs[:-1]
        fake_features = fake_outputs[:-1]
        predict_real = real_outputs[-1]
        predict_fake = fake_outputs[-1]

        dis_loss = -(
            torch.log(predict_real + 1e-8) + torch.log(1.0 - predict_fake + 1e-8)
        ).mean()
        gen_loss_gan = -torch.log(predict_fake + 1e-8).mean()
        perceptual = [
            torch.mean(torch.abs(fake - real))
            for fake, real in zip(fake_features, real_features)
        ]
        gen_l1 = F.l1_loss(generated, target)
        gen_loss = (
            gen_loss_gan * 0.1
            + perceptual[0] * 0.1
            + sum(loss * 10.0 for loss in perceptual[1:])
            + gen_l1 * 100.0
        )

        metrics = {
            "dis_loss": dis_loss,
            "gen_loss_GAN": gen_loss_gan,
            "gen_p1": perceptual[0],
            "gen_p2": perceptual[1],
            "gen_p3": perceptual[2],
            "gen_p4": perceptual[3],
            "gen_p5": perceptual[4],
            "gen_p6": perceptual[5],
            "gen_p7": perceptual[6],
            "gen_p8": perceptual[7],
            "gen_L1": gen_l1 * 100.0,
            "total": gen_loss,
        }
        return gen_loss, dis_loss, metrics

    def _train_step(self, original, target):
        self.G.train()
        self.D.train()
        self.gen_optimizer.zero_grad(set_to_none=True)
        self.dis_optimizer.zero_grad(set_to_none=True)

        gen_loss, dis_loss, metrics = self._losses(original, target)
        generator_gradients = torch.autograd.grad(
            gen_loss,
            tuple(self.G.parameters()),
            retain_graph=True,
        )
        discriminator_gradients = torch.autograd.grad(
            dis_loss,
            tuple(self.D.parameters()),
        )

        for parameter, gradient in zip(self.G.parameters(), generator_gradients):
            parameter.grad = gradient
        for parameter, gradient in zip(self.D.parameters(), discriminator_gradients):
            parameter.grad = gradient

        self.gen_optimizer.step()
        self.dis_optimizer.step()
        return {name: float(value.detach().cpu()) for name, value in metrics.items()}

    def _update_history(self, metrics):
        for name, value in metrics.items():
            if name in self.history and self.history[name]:
                value = value * (1.0 - self._ema) + self.history[name][-1] * self._ema
                self.history[name].append(value)
            else:
                self.history[name] = [value]

    def _plot_progress(self, original, target):
        self.G.eval()
        with torch.inference_mode():
            generated = self.G(original)

        original = ((original[0] + 1.0) / 2.0).detach().cpu().numpy()
        generated = ((generated[0] + 1.0) / 2.0).detach().cpu().numpy()
        target = ((target[0] + 1.0) / 2.0).detach().cpu().numpy()

        plt.close()
        _, axes = plt.subplots(1, 3, sharex=True, figsize=(16.5, 5.5))
        titles = ("Original", "Starless", "Target")
        for axis, title, image in zip(axes, titles, (original, generated, target)):
            if self.mode == "RGB":
                axis.imshow(np.transpose(image, (1, 2, 0)))
            else:
                axis.imshow(image[0], cmap="gray", vmin=0, vmax=1)
            axis.set_title(title)
        plt.pause(0.001)
        self.G.train()

    def train(
        self,
        epochs,
        augmentation=True,
        plot_progress=False,
        plot_interval=50,
        save_backups=True,
        warm_up=False,
    ):
        if not self.original:
            raise RuntimeError("Training dataset was not loaded; call load_training_dataset()")
        if self.G is None or self.D is None:
            self.initialize_model()

        for epoch in range(epochs):
            for iteration in range(self.iters_per_epoch):
                original, target = self.generate_input(augmentation=augmentation)
                original = torch.from_numpy(original.transpose(0, 3, 1, 2)).to(self.device)
                target = torch.from_numpy(target.transpose(0, 3, 1, 2)).to(self.device)
                original = original * 2.0 - 1.0
                target = target * 2.0 - 1.0
                if warm_up:
                    target = original

                if plot_progress and iteration % plot_interval == 0:
                    self._plot_progress(original, target)

                metrics = self._train_step(original, target)
                self._update_history(metrics)
                print(
                    "\rEpoch: %d. Iteration %d / %d Loss %f    "
                    % (epoch, iteration, self.iters_per_epoch, self.history["total"][-1]),
                    end="",
                )

            print()
            if save_backups:
                suffix = "even" if epoch % 2 == 0 else "odd"
                self._save_network(self.G, "./starnet_backup_G_%s.pth" % suffix)
                self._save_network(self.D, "./starnet_backup_D_%s.pth" % suffix)

        if plot_progress:
            plt.close()
        self.G.eval()
        self.D.eval()
        return self.history

    def plot_history(self, last=None):
        if not self.history:
            raise RuntimeError("Empty training history, nothing to plot")

        keys = list(self.history.keys())
        columns = 3
        rows = int(np.ceil(len(keys) / columns))
        figure, axes = plt.subplots(rows, columns, sharex=True, figsize=(16, 3.5 * rows))
        axes = np.atleast_1d(axes).reshape(-1)
        for axis, key in zip(axes, keys):
            values = self.history[key][-last:] if last else self.history[key]
            axis.plot(values)
            axis.set_title(key)
        for axis in axes[len(keys):]:
            axis.set_visible(False)
        figure.tight_layout()
        return figure, axes

    @staticmethod
    def _save_network(network, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": network.state_dict()}, path)

    def save_model(self, weights_filename, history_filename=None):
        if self.G is None or self.D is None:
            raise RuntimeError("Model has not been initialized")

        generator_path = Path(str(weights_filename) + "_G_" + self.mode + ".pth")
        discriminator_path = Path(str(weights_filename) + "_D_" + self.mode + ".pth")
        self._save_network(self.G, generator_path)
        self._save_network(self.D, discriminator_path)

        if history_filename:
            history_path = Path(str(history_filename) + "_" + self.mode + ".pkl")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("wb") as handle:
                pickle.dump(self.history, handle)
        return generator_path, discriminator_path

    def _read_image(self, input_path):
        data = tiff.imread(input_path)
        if data.ndim > 3:
            layer = input("Tiff has %d layers, please enter layer to process: " % data.shape[0])
            data = data[int(layer)]

        if data.dtype == np.dtype("uint16"):
            scale = 65535.0
        elif data.dtype == np.dtype("uint8"):
            scale = 255.0
        else:
            raise ValueError("Unknown image dtype: %s" % data.dtype)

        image = (data / scale).astype(np.float32)
        if self.mode == "Greyscale":
            if image.ndim != 2:
                raise ValueError("You loaded Greyscale model, but the image is RGB")
            image = image[:, :, None]
        else:
            if image.ndim != 3:
                raise ValueError("You loaded RGB model, but the image is Greyscale")
            if image.shape[2] == 4:
                print("Input image has 4 channels. Removing Alpha-Channel")
                image = image[:, :, :3]
            if image.shape[2] != 3:
                raise ValueError("RGB input must have 3 or 4 channels")
        return image, data.dtype, scale

    @staticmethod
    def _imwrite(path, image):
        if hasattr(tiff, "imwrite"):
            tiff.imwrite(path, image)
        else:
            tiff.imsave(path, image)

    def transform(self, in_name, out_name):
        if self.G is None:
            raise RuntimeError("Call load_model() before transform()")

        self.G.eval()
        image, input_dtype, scale = self._read_image(in_name)
        offset = (self.window_size - self.stride) // 2
        h, w, _ = image.shape
        ith = h // self.stride + 1
        itw = w // self.stride + 1
        dh = ith * self.stride - h
        dw = itw * self.stride - w

        image = np.concatenate((image, image[(h - dh):, :, :]), axis=0)
        image = np.concatenate((image, image[:, (w - dw):, :]), axis=1)
        padded_h, padded_w, _ = image.shape
        image = np.concatenate((image, image[(padded_h - offset):, :, :]), axis=0)
        image = np.concatenate((image[:offset, :, :], image), axis=0)
        image = np.concatenate((image, image[:, (padded_w - offset):, :]), axis=1)
        image = np.concatenate((image[:, :offset, :], image), axis=1)
        image = image * 2.0 - 1.0
        output = np.empty_like(image)

        with torch.inference_mode():
            for i in range(ith):
                for j in range(itw):
                    x = self.stride * i
                    y = self.stride * j
                    tile = image[
                        x:x + self.window_size,
                        y:y + self.window_size,
                        :,
                    ]
                    tile = (
                        torch.from_numpy(tile.transpose(2, 0, 1))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    result = self.G(tile)
                    result = ((result[0] + 1.0) / 2.0).cpu().numpy().transpose(1, 2, 0)
                    output[
                        x + offset:x + offset + self.stride,
                        y + offset:y + offset + self.stride,
                        :,
                    ] = result[
                        offset:offset + self.stride,
                        offset:offset + self.stride,
                        :,
                    ]
                    progress = int((itw * i + j + 1) * 100 / (ith * itw))
                    print("Transforming input image... %d%%\r" % progress, end="")

        print("Transforming input image... Done!")
        output = np.clip(output, 0.0, 1.0)
        output = output[offset:offset + h, offset:offset + w, :]
        if self.mode == "Greyscale":
            output = output[:, :, 0]

        output_path = Path(out_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._imwrite(output_path, (output * scale).astype(input_dtype))
        return output_path
