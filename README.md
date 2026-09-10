# AstroColorProcess

面向天文摄影后期处理的 Python/PyTorch 示例项目。目前包含 AI 星图降噪、AI 去星，
以及背景提取、星点反卷积和星点缩小等实验模块。

- GitHub：<https://github.com/CHMimilanlan/AstroColorProcess>
- 模型权重：<https://huggingface.co/MilanMagik/AstroStarDetect>

## AI 模块

| 目录 | 功能 | 推理权重 |
| --- | --- | --- |
| `StarDenoise` | 星图/深空图像降噪 | `deep_denoise_cnn_AI3_6.pth` |
| `StarNetPyTorch` | RGB 或灰度天文图像去星 | `weights_G_RGB.pth`、`weights_G_Greyscale.pth` |

权重没有提交到 GitHub，请从 Hugging Face 下载。仓库内的测试图和输出仅用于演示，
实际部署时可以只保留代码、依赖和所需权重。

## 模型规模与显存

下面统计的是 Hugging Face 中三个可部署的 PyTorch 推理 checkpoint。`B` 表示十亿参数，
文件大小使用二进制单位 MiB（`1 MiB = 1024² bytes`）。

| 推理模型 | 参数量 | 参数量（B） | checkpoint 文件 | 仅参数 FP32 | 仅参数 FP16 | 256×256 FP32 张量峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cosmic Clarity AI 3.6 降噪 | 2,945,283（2.945M） | 0.002945B | 33.77 MiB（35.42 MB） | 11.24 MiB | 5.62 MiB | 361.67 MiB |
| StarNet RGB 生成器 | 54,420,483（54.420M） | 0.054420B | 207.67 MiB（217.75 MB） | 207.60 MiB | 103.80 MiB | 356.33 MiB |
| StarNet 灰度生成器 | 54,414,337（54.414M） | 0.054414B | 207.64 MiB（217.73 MB） | 207.57 MiB | 103.79 MiB | 355.31 MiB |
| **全部三个 checkpoint** | **111,780,103（111.780M）** | **0.111780B** | **449.08 MiB（470.90 MB）** | **426.41 MiB** | **213.20 MiB** | 不会同时执行 |

普通 RGB 处理流水线只加载“降噪 + StarNet RGB”，不是同时加载 RGB 和灰度去星模型：

| 组合 | 参数量 | 参数量（B） | checkpoint 合计 | FP32 参数合计 |
| --- | ---: | ---: | ---: | ---: |
| 降噪 + StarNet RGB | 57,365,766 | 0.057366B | 241.44 MiB | 218.83 MiB |
| 降噪 + StarNet 灰度 | 57,359,620 | 0.057360B | 241.42 MiB | 218.81 MiB |

### 显存口径

- “仅参数”是由参数量乘以每个参数的字节数得到的确定下限，不包含输入、激活、CUDA
  context、cuDNN workspace 和 PyTorch 缓存。
- “256×256 FP32 张量峰值”是在 `torch.inference_mode()`、`batch=1` 下使用 PyTorch
  内存时间线测得的 CPU 张量峰值，用来估算相同计算图的显存主体，不是 `nvidia-smi`
  的 CUDA 实测值。本机有 RTX 4060 Laptop 8 GB，但当前 `study` 环境是
  `torch 2.13.0+cpu`，无法直接调用 CUDA 峰值统计接口。
- 降噪程序默认 `tile-size=256` 且在 CUDA 上使用混合精度。按激活减半、FP32 主权重
  常驻估算，张量主体约 **186 MiB**；加上 CUDA context、算子 workspace 和缓存后，
  建议至少 **2 GB 显存**。
- StarNet 当前按 FP32 推理。`window-size=256` 的生成器张量峰值约 **356 MiB**；默认
  `window-size=512` 按空间面积推算约 **803 MiB（RGB）/ 799 MiB（灰度）**。
  算上 CUDA 运行时后建议至少 **3 GB 显存**，**4 GB 或以上更稳妥**。
- 当前 `StarNet.load_model()` 还会实例化训练用判别器，虽然推理时不执行它，但会额外
  常驻约 **1.77M 参数 / 6.77 MiB FP32**。上表的模型参数量按真正参与推理的生成器统计。
- 1080P 或更大图像采用分块处理时，显存主要由 `tile-size` / `window-size` 和
  `batch-size` 决定，而不是整张图的像素总数；提高 batch 时，激活显存近似线性增加。
- 降噪与去星顺序执行时，进程应在阶段间释放前一个模型；流水线峰值取两者较大值，
  不需要把两张峰值表简单相加。8 GB 显存可以宽裕运行这里的默认推理配置。

本地 `starnet_weights2` 中还保留了格式转换和训练辅助文件：两个生成器 `.h5` 各约
207.75 MiB、两个判别器 `.h5` 各约 6.85 MiB、两个训练历史 `.pkl` 各约 4.69 MiB。
它们不是额外的推理模型；生成器 `.h5` 与已发布 `.pth` 表示同一组模型，部署无需下载。

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
