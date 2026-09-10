# AstroColorProcess

面向天文摄影后期处理的 Python/PyTorch 示例项目。目前包含 AI 星图降噪、AI 去星，
以及背景提取、星点反卷积和星点缩小等实验模块。

- GitHub：<https://github.com/CHMimilanlan/AstroColorProcess>
- 模型权重：<https://huggingface.co/MilanMagik/AstroStarDetect>

## AI 模块

| 目录 | 功能 | 推理权重 | 大小（约） |
| --- | --- | --- | ---: |
| `StarDenoise` | 星图/深空图像降噪 | `deep_denoise_cnn_AI3_6.pth` | 33.8 MB |
| `StarNetPyTorch` | RGB 或灰度天文图像去星 | `weights_G_RGB.pth`、`weights_G_Greyscale.pth` | 各 208 MB |

权重没有提交到 GitHub，请从 Hugging Face 下载。仓库内的测试图和输出仅用于演示，
实际部署时可以只保留代码、依赖和所需权重。

## 1. 环境安装

本项目在 Windows 的 Conda `study` 环境中验证：

```powershell
conda activate study
cd D:\Workman\MyProject\ImageStack\AstroColorProcess
python -m pip install -r requirements.txt
```

如需 NVIDIA GPU，请根据本机 CUDA 版本先从
[PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)安装对应的 CUDA 版
PyTorch，再安装其余依赖。CPU 也能运行，但处理高分辨率图像会慢很多。

## 2. 下载模型权重

推荐使用仓库自带脚本。它会把权重放到两个程序默认读取的位置：

```powershell
conda activate study
python download_models.py
```

也可以只下载其中一类模型：

```powershell
python download_models.py --only denoise
python download_models.py --only starnet
```

等价的 Hugging Face CLI 命令：

```powershell
hf download MilanMagik/AstroStarDetect `
  StarDenoise/models/deep_denoise_cnn_AI3_6.pth `
  StarNetPyTorch/models/weights_G_RGB.pth `
  StarNetPyTorch/models/weights_G_Greyscale.pth `
  --local-dir .
```

下载后目录应为：

```text
AstroColorProcess/
├── StarDenoise/models/deep_denoise_cnn_AI3_6.pth
└── StarNetPyTorch/models/
    ├── weights_G_RGB.pth
    └── weights_G_Greyscale.pth
```

权重 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `deep_denoise_cnn_AI3_6.pth` | `f860e234dc6a3ab1acacbea399c235179c86cb8b287f2ab9010bc18c8af08671` |
| `weights_G_RGB.pth` | `e69bef14cfe0363f2df79752cae7deca5f8ea3e73685e58998a39409ce9a8a46` |
| `weights_G_Greyscale.pth` | `ed4df34a9f655274c9e985a3d1742e649bb0bb3c21ec63f20f42a593e002dca4` |

## 3. 星图降噪

支持常见位图、TIFF 和 FITS。默认会自动选择 CUDA 或 CPU，并使用分块推理降低显存占用：

```powershell
cd StarDenoise
python denoise.py input.tif output.tif --device auto --strength 0.85
```

密集星场建议从较低强度开始，避免把真实的小星点误判为噪声：

```powershell
python denoise.py input.tif output.tif --device auto --strength 0.30 --color-strength 0.25
```

常用参数：

- `--strength`：亮度降噪强度，通常为 `0.25–0.90`。
- `--color-strength`：色彩降噪强度。
- `--tile-size` / `--overlap`：分块尺寸和重叠宽度。
- `--linear-stretch never`：对已经拉伸的普通图片关闭线性拉伸。

更完整的模型说明、格式支持和训练建议参见
[`StarDenoise/README.md`](StarDenoise/README.md)。1080P 实测记录见
[`StarDenoise/README_1080P.md`](StarDenoise/README_1080P.md)。

## 4. StarNet PyTorch 去星

推荐使用 TIFF 输入。RGB 图像默认加载 `weights_G_RGB.pth`：

```powershell
cd StarNetPyTorch
python starnet_v1_pytorch_transform.py input.tif output_starless.tif `
  --mode RGB --device cuda
```

灰度图像：

```powershell
python starnet_v1_pytorch_transform.py input.tif output_starless.tif `
  --mode Greyscale --device cpu
```

不传 `--device` 时程序会自动选择 CUDA 或 CPU。显存紧张时可减小窗口或增加步长，
例如 `--window-size 256 --stride 128`；`window-size` 必须能被 256 整除。

## 5. 离线部署

在有网络的机器上先执行：

```powershell
python download_models.py
python -m pip download -r requirements.txt -d wheels
```

把整个项目和 `wheels` 目录复制到离线机器，然后执行：

```powershell
python -m pip install --no-index --find-links wheels -r requirements.txt
```

## 6. 权重与许可证说明

`StarDenoise` 是对 Cosmic Clarity AI 3.6 PyTorch 模型的轻量部署封装；仓库保留了
上游 MIT 许可证文本和实现说明。`StarNetPyTorch` 是 StarNet 架构的 PyTorch 转写与
权重转换/推理示例。第三方代码、样例和权重仍受各自上游许可证或使用条款约束；
用于商业产品前请自行核对对应上游授权。本仓库上传权重的目的仅为复现与部署便利，
不改变其原始权利归属。

## 7. 快速自检

```powershell
conda activate study
python -m unittest discover -s StarDenoise/tests -v
python -m compileall -q download_models.py StarDenoise StarNetPyTorch
```
