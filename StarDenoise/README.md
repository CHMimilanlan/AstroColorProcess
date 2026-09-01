# StarDenoise：天文图像 AI 降噪 Demo

这是一个可直接运行的 PyTorch 天文图像降噪 demo。复现对象是 **SetiAstro Cosmic Clarity Denoise AI 3.6** 的网络与官方预训练权重，并将原项目中面向桌面 GUI 的推理逻辑整理成适合工程集成的命令行模块。

本目录已经包含：

- PyTorch 模型结构与官方 AI 3.6 权重；
- 从 AstroNoiseNet 官方仓库下载的 M51 短曝光/长曝光测试对；
- JPG、PNG、TIFF、单色或 RGB FITS 的推理支持；
- 自动 GPU/CPU 选择、重叠分块、边缘融合、降噪强度与模式控制；
- FITS 线性数据的自动拉伸、网络推理、反拉伸，以及 FITS header 保留；
- 下载校验脚本、简单的参考图 PSNR/MAE 评估和单元测试。
- 三张 ESO 1920×1080 在线图片的完整测试、对比图与指标，见 [`README_1080P.md`](README_1080P.md)。

> 注意：“星图降噪”与“去星”“星点缩小”“反卷积”不同。降噪要压制随机/结构噪声，同时尽量保持星点 PSF、亮度、星云微弱结构和色彩；它不能替代 bias/dark/flat 校准、配准和叠加。

## 1. 快速运行

工作目录：

```powershell
cd D:\Workman\MyProject\ImageStack\AstroColorProcess\StarDenoise
conda activate study
```

环境依赖已经在 `study` 中安装过；在新机器上可执行：

```powershell
# 根据显卡/CUDA 选择 PyTorch 官方安装命令，然后：
python -m pip install -r requirements.txt
```

当前已验证环境是 Python 3.10.20、PyTorch 2.13.0+cpu。由于本次网络下载 CUDA wheel 过慢，先安装 CPU 版保证 demo 即开即用；代码本身已支持 CUDA。要启用本机 RTX 4060，可在网络条件合适时按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)重装 CUDA wheel，例如：

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision `
  --index-url https://download.pytorch.org/whl/cu126
```

运行内置 M51 样例：

```powershell
python denoise.py samples\M51_short.jpg outputs\M51_short_denoised.jpg `
  --device auto --strength 0.9 --mode full --tile-size 256 --overlap 64
```

运行自己的 FITS：

```powershell
python denoise.py D:\path\input.fits outputs\input_denoised.fits `
  --device cuda --strength 0.8 --mode luminance
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--device` | `auto` | 自动选 CUDA，否则 CPU；也可指定 `cuda:0` |
| `--strength` | `0.9` | 神经网络亮度降噪结果与原图的混合比例，范围 0～1 |
| `--color-strength` | 同 `strength` | `full` 模式下的色度降噪强度 |
| `--mode` | `full` | `luminance` 只处理亮度；`full` 亮度 AI + 引导色度滤波；`separate` 三通道各自 AI |
| `--tile-size` | `256` | 官方模型训练/推理块大小；一般无需修改 |
| `--overlap` | `64` | tile 重叠宽度，减小接缝 |
| `--batch-size` | `1` | 8 GB 显存建议 1～2；显存足够再增加 |
| `--precision` | `fp16` | CUDA 上用半精度；CPU 会自动使用 FP32 |
| `--linear-stretch` | `auto` | 线性天文图自动 MTF 拉伸后推理并反拉伸，可设 `always/never` |

比较短曝光输入、我们的结果与仓库给出的长曝光参考：

```powershell
python evaluate.py samples\M51_long_reference.jpg samples\M51_short.jpg
python evaluate.py samples\M51_long_reference.jpg outputs\M51_short_denoised.jpg
```

长曝光参考并不是严格像素级 ground truth（曝光、配准、亮度都可能存在差异），因此该指标只能用于 demo 的相对观察，不能当成论文级结论。更可靠的验证见“训练与评估建议”。

本机实测结果（CPU、400×400、FP32、strength=0.9、full）：

| 项目 | 相对 M51 长曝光参考 |
|---|---:|
| 短曝光输入 | PSNR 30.588 dB；MAE 0.022448 |
| 本 demo 输出 | **PSNR 35.322 dB；MAE 0.013325** |
| 推理时间 | 3.414 秒 |

官方权重严格加载成功，4 个单元测试（结构/参数量、颜色空间往返、零强度恒等、FITS channel-first 往返）全部通过。

如需重新下载资产：

```powershell
python download_assets.py
```

权重下载后会检查官方公布的 SHA-256：`f860e234dc6a3ab1acacbea399c235179c86cb8b287f2ab9010bc18c8af08671`。

## 2. 天文图像噪声与降噪原理

天文图像的信号极弱，观测值可简化为：

```text
观测 = 天体光子 + 天空背景 + 暗电流 + 偏置 + 读出噪声 + 固定图样 + 离群点
```

主要成分如下：

- **光子散粒噪声**：光子到达服从泊松统计，标准差约为 `sqrt(N)`；它是信号相关噪声，不是固定方差高斯噪声。
- **天空背景噪声**：月光、光污染和大气辉光也产生光子与散粒噪声。
- **读出噪声**：传感器读出电路在每个子曝光中引入，常近似高斯分布。
- **暗电流及热像素**：热产生电子；其平均图样可由 dark frame 校正，但暗电流自身仍有随机性。
- **偏置、PRNU、渐晕和尘埃**：分别通过 bias、flat 等校准处理；校准残差会表现为结构噪声。
- **宇宙射线、卫星、坏点**：属于稀疏离群污染，通常应在叠加/专门剔除阶段处理，而非交给普通降噪器“猜”。

[Astropy CCD reduction guide](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-01-astronomical-CCD-image-components.html) 对 bias、dark、read noise、天空背景和 flat correction 有系统说明；NASA 的[传感器噪声与平方根定律说明](https://science.nasa.gov/wp-content/uploads/2023/09/Astrophotography_Guide.pdf)解释了泊松计数的 `sqrt(N)` 关系。2026 年的 [Denoising the Deep Sky](https://arxiv.org/abs/2601.23276) 进一步把光子散粒、PRNU、暗电流、读出效应、宇宙射线和热像素统一到物理 CCD 噪声合成管线中。

传统流程首先做校准、配准和叠加。对独立随机噪声，平均 `N` 帧后噪声标准差大致降低到 `1/sqrt(N)`；深度模型学习的是一个条件先验：根据大量“短曝光/带噪输入 → 长曝光堆栈/低噪目标”样本，预测给定结构更可能对应真实星点和星云，哪些更可能是噪声。

常见网络有两种输出形式：

1. 直接输出干净图 `x_hat = f(y)`；
2. 预测噪声残差 `n_hat = f(y)`，再计算 `x_hat = y - n_hat`。

天文降噪比普通照片更怕“看起来漂亮但不真实”：很小的点状结构既可能是恒星，也可能是热像素/噪声；过强感知损失或 GAN 还可能生成不存在的结构。因此不仅要看 PSNR/SSIM，还要检查星点检出率、孔径测光偏差、PSF/FWHM、背景 NMAD、假阳性和弱结构保真度。

## 3. 模型调研与选型

调研日期：2026-08-31。这里把“学术主流”和“深空摄影社区实际可用”分开考虑。

| 工作 | 领域匹配 | 开源/权重 | 规模与部署 | 结论 |
|---|---|---|---|---|
| [DnCNN](https://github.com/cszn/DnCNN) | 通用 AWGN | 官方代码和 PyTorch 权重；经典论文，社区最成熟 | 约 0.56M 参数，极轻 | 很适合基线，但高斯噪声假设与天文泊松/读噪/固定图样不完全匹配 |
| [SCUNet](https://github.com/cszn/SCUNet) | 通用真实图盲降噪 | 官方 PyTorch；Swin + Conv，研究认可度高 | 约 17.95M 参数、FP32 约 72 MB，速度/显存成本较高 | 通用盲降噪强，但不是天文数据训练，弱星点有域偏移风险 |
| [Astro U-net](https://github.com/Sponka/Astro_U-net) | HST WFC3 科学图像 | 论文、代码和补充数据开放 | U-Net；论文报告 SNR 平均提高 1.63×、恒星恢复率 95.9%、平均 flux error 2.26% | 科学依据强，但针对特定 HST 波段；直接用于消费级 RGB 深空摄影不够即插即用 |
| [AstroNoiseNet](https://github.com/Steffenhir/AstroNoiseNet) | 真实短/长曝光深空摄影 | MIT，代码开放；PRIDNet + 判别器 | 较复杂，仓库没有便于独立部署的现成 PyTorch checkpoint | 数据构造思路很好；本 demo 的 M51 测试对来自该仓库 |
| [GraXpert](https://github.com/Steffenhir/GraXpert) | 深空摄影 | GPL-3.0 应用开放，发布 ONNX 模型 | 产品成熟、约 300+ GitHub stars、支持 CLI/PixInsight/Siril 工作流 | 实际认可度最高之一，但模型以 ONNX 应用形式分发，独立 PyTorch 复现和再训练不如本方案直接 |
| [Cosmic Clarity AI 3.6](https://github.com/setiastro/cosmicclarity) | 深空摄影 | **MIT，官方 PyTorch `.pth` + ONNX** | **2,945,283 参数；纯 FP32 参数约 11.24 MiB；官方 checkpoint 33.8 MB** | **本 demo 选择：领域匹配、体积、开放程度和部署成本最均衡** |
| [Denoising the Deep Sky](https://github.com/ShuhongLL/Denoising-Deep-Sky) | 科研级 raw CCD | 代码、MuSCAT 数据、PyTorch 训练流程开放 | 物理噪声模型 + U-Net/PMN-U-Net，数据约 124 GB | 2026 年的新工作，科学训练路线很强，但还不够“轻量即用”，适合下一阶段自训 |

选择 Cosmic Clarity 的理由：

- 是针对实际深空摄影训练的模型，不是用自然图像 AWGN 权重硬套；
- 官方直接发布 PyTorch checkpoint 和网络源代码，许可证为 MIT；
- 约 2.94M 参数，显著小于 SCUNet，256 tile 可在 8 GB 显卡上轻松运行；
- 结构是全分辨率残差 CNN，含 dilation 和多层 skip concatenation，适合保存点源与小尺度纹理；
- 上游工具已形成社区使用，并有 Siril 相关集成；
- 相比只发布 ONNX 的应用模型，更容易审阅、修改、继续训练和嵌入 Python 工程。

局限也要明确：该模型没有正式 model card 公布训练集规模、相机分布和严格 benchmark；MIT 是仓库整体许可证，权重发布页没有更细的单独数据许可声明。它适合“工程 demo / 深空摄影后期”，若目标是可发表的测光或形态学分析，必须在自己的仪器和目标分布上重新评估或训练。

## 4. 本复现与上游的对应关系

网络完全对应官方 AI 3.6 PyTorch 定义：

- 输入/输出：3 通道，范围 `[0,1]`；单色图复制为三通道，取输出第一通道；
- 编码通道：`3 → 16 → 32 → 64 → 128 → 256`；
- 64 和 256 通道入口使用 dilation=2；
- 每级包含两层 3×3 卷积的 residual block；
- 解码端逐级拼接 encoder 特征，最终 sigmoid 输出；
- 模型不做 pooling，空间尺寸始终不变。

本 demo 保留上游的 256 tile、64 overlap、MTF 线性拉伸/反拉伸、Y/Cb/Cr 亮度处理和引导色度降噪；工程化改动包括：

- Hann 权重融合取代简单覆盖/平均，减少 tile 接缝；
- `torch.inference_mode()`、CUDA FP16 和可调 batch；
- `torch.load(..., weights_only=True)` 限制反序列化范围；
- 独立、可测试的 I/O，支持 FITS header 与原始动态范围回写；
- CLI 参数和 SHA-256 校验。

官方来源和许可证副本保存在 `third_party/`。测试样例来自 AstroNoiseNet 的 `examples/M51_short.jpg` 与 `examples/M51_long.jpg`，均由本项目的下载脚本给出可复现 URL。

## 5. 推荐处理顺序

对消费级深空摄影数据，建议：

```text
raw 子帧
  → bias/dark/flat 校准
  → 星点配准
  → 异常值剔除与叠加
  → 背景/梯度校正、颜色校准
  → 在线性数据上温和降噪（或按训练域选择拉伸后降噪）
  → 反卷积/星点处理
  → 非线性拉伸与最终调色
```

若模型明确在 stretched 图上训练（Cosmic Clarity 属于这一类工程工具），本 demo 的 `auto` 会在线性 FITS 中位数很低时先做可逆 MTF 拉伸。强度建议从 `0.6～0.8` 开始，放大 100% 检查弱星与星云边缘；不要只看缩略图。

## 6. 如何获取自己的训练数据

### 路线 A：自己的相机，最推荐

对每个目标保存校准后的单帧，并生成高 SNR reference：

1. 同一目标、相同相机/增益/温度/滤镜拍摄 20～100+ 个 dithered subframes；
2. 每个子帧完成 bias/dark/flat 校准；
3. 亚像素配准到同一坐标；
4. 从其中一帧或少量帧构造 noisy input；其余帧用 sigma-clipping/加权平均生成 clean target；
5. 避免把 input 本身同时放进 target stack，否则 target 噪声与输入相关，会高估效果；
6. 不同观测目标划分 train/val/test，不能把同一张大图随机 patch 后分到不同集合；
7. 保存 32-bit float linear FITS，以及 gain、read noise、曝光、温度、滤镜等 metadata。

如果没有足够干净 target，可用 **Noise2Noise**：把同一视场奇数帧和偶数帧分别叠加成两张独立噪声图作为 pair。两边天体信号一致、随机噪声独立，使用 L1/L2 训练仍能学习期望的干净信号。也可用 Noise2Void/Self2Self 做单图自监督，但在密集星场要谨慎设计 blind-spot，避免把点源当噪声。

### 路线 B：公开科学数据

- [MuSCAT Raw Image dataset](https://huggingface.co/datasets/ToferFish/MuSCAT-RawImage)：约 124 GB，CC BY-NC 4.0；包含 MuSCAT-3/4 的 raw、overscan-corrected `OS`、calibrated、stacked `MEAN`、mask、bias/dark/flat。可构造 `OS → CALIB` 或 `synthetic → MEAN`，是当前最完整的物理 CCD 降噪数据之一。非商业限制必须注意。
- [Denoising the Deep Sky 官方代码](https://github.com/ShuhongLL/Denoising-Deep-Sky)：包含 FITS→NPY、数据划分、U-Net/PMN-U-Net 训练、PSF 注入和 PSNR/SSIM/NMAD/FLUX-SNR 评估脚本。
- [MAST](https://archive.stsci.edu/)：HST 等任务的公开观测档案。可按 Astro U-net 方法选择同一目标/滤镜的短曝光与深曝光或 drizzle 产品；务必读取仪器校准与 data-use 说明。
- [Astro U-net 论文与代码](https://github.com/Sponka/Astro_U-net)：适合复现 HST WFC3 F555W/F606W 的科学图像训练设计。
- [AstroNoiseNet](https://github.com/Steffenhir/AstroNoiseNet)：仓库展示真实短/长积分 pair 与数据预处理思路，但没有发布完整训练数据；可借鉴采集协议。

公开数据和自己相机的数据不要直接混合后随机切分。不同相机的 PSF、像元、Bayer/mono、增益、读噪、滤镜和后处理差异很大，应先按 instrument 做域划分和归一化，并保留独立的跨仪器测试集。

## 7. 自训练架构建议

### 最低风险、最快落地

直接继续训练本 demo 的 `CosmicClarityDenoiseCNN`：

- 2.95M 参数，8 GB RTX 4060 可用 `256×256` crop、batch 2～4、AMP；
- 单色科学 FITS 最好把首尾层改为 1 通道，不要无意义复制 RGB；
- 输入预测 clean image，或者改为 residual noise prediction 并加全局 skip；
- 在输入中追加 noise-level/gain/read-noise map，提升不同相机和曝光参数的泛化。

### 追求更高上限

使用小型 **Residual U-Net / NAFNet-small**，4 层尺度、base width 32，约 8～20M 参数。下采样扩大感受野，skip 保留星点定位；用 tile inference 控制显存。科研 raw CCD 可参考 PMN-U-Net，把物理噪声参数或固定图样作为条件输入。

不建议第一版就用扩散模型或 GAN：推理慢、训练成本高，而且感知上“锐利”的结果可能改变光度或生成假结构。天文图像首先应优化可测量的真实性。

建议训练配置：

| 项目 | 建议 |
|---|---|
| 数据域 | 线性、校准后、未锐化 FITS；每个 filter/instrument 记录 metadata |
| crop | 256×256；至少一部分 patch 强制含星点/星云，另一部分纯背景 |
| augmentation | 90° 旋转、翻转；Bayer raw 不能任意旋转/翻转后破坏 CFA 排列 |
| optimizer | AdamW，初始 LR `1e-4`，cosine decay，50～150 epochs |
| primary loss | Charbonnier 或 L1，比纯 MSE 更抗离群点 |
| structure loss | `0.1～0.2 × (1-SSIM)` + 小权重 gradient loss |
| astronomy loss | aperture-flux L1、PSF/FWHM loss 或 source-mask 加权；背景与源区分别统计 |
| validation | 按目标/夜晚/仪器分组，绝不能 patch 泄漏 |
| checkpoint | 以 flux bias + 背景 NMAD + SSIM 综合选模，而非只看训练 loss |

物理噪声合成可在干净 stack 上加入：

```text
electrons = clean_ADU × gain
shot      ~ Poisson(electrons)
dark      ~ Poisson(dark_current × exposure)
read      ~ Normal(0, read_noise_e)
PRNU      = signal × pixel_response_map
outliers  = hot pixels + cosmic ray events
noisy_ADU = (shot + dark + read + PRNU + outliers) / gain
```

参数应从真实 bias/dark/flat 与相机 specification 估计，不要只用固定 `Gaussian(σ=25)`。合成数据用于扩充覆盖，最终仍应以真实独立短/长曝光 pair 微调和验证。

## 8. 科学验证清单

在把模型用于正式处理前，至少做以下检查：

1. 背景：均值/中位数偏移、标准差、MAD/NMAD、空间相关噪声功率谱；
2. 星点：检测 completeness、false positive、centroid 偏差、FWHM/椭率变化；
3. 测光：多个亮度区间的 aperture flux ratio 与 magnitude bias；
4. 扩展源：表面亮度、颜色、弱结构差值图；
5. 泛化：未见过的目标、夜晚、相机、滤镜、增益和曝光；
6. 伪影：星周黑环、彩斑、tile seam、背景塑料感、被抹掉的弱星；
7. 与传统基线比较：stacking、BM3D/小波、DnCNN/SCUNet，以及不降噪输入。

论文级评估可参考 Astro U-net 的星点恢复率与 flux error，以及 Denoising the Deep Sky 的 PSNR、SSIM、NMAD、source injection 与 FLUX-SNR。保存原始图、输出图、模型版本、权重 hash、参数和环境版本，保证可追溯。

## 9. 目录结构与许可证

```text
StarDenoise/
├── denoise.py                 # CLI 与推理管线
├── model.py                   # AI 3.6 PyTorch 网络
├── image_io.py                # JPG/PNG/TIFF/FITS I/O
├── evaluate.py                # 样例全参考指标
├── download_assets.py         # 官方资产下载与 hash 校验
├── models/                    # 官方 checkpoint
├── samples/                   # 官方 M51 短/长曝光样例
├── outputs/                   # demo 输出
├── tests/                     # 单元测试
└── third_party/               # 上游许可证/审阅源码
```

Cosmic Clarity 和 AstroNoiseNet 均使用 MIT License；上游许可证副本位于 `third_party/`。本复现应保留这些版权与许可声明。测试样例的使用遵循其来源仓库条款；用于商业产品前，仍建议向权重作者确认模型训练数据与权重的商业授权范围。

## 10. 主要来源

- [Cosmic Clarity 官方仓库（MIT、PyTorch 源码）](https://github.com/setiastro/cosmicclarity)
- [Cosmic Clarity 官方 Release（AI 3.6 PTH/ONNX 与 SHA-256）](https://github.com/setiastro/cosmicclarity/releases/tag/Linux)
- [AstroNoiseNet 官方仓库与 M51 样例](https://github.com/Steffenhir/AstroNoiseNet)
- [GraXpert 官方仓库与 CLI](https://github.com/Steffenhir/GraXpert)
- [Learning to denoise astronomical images with U-nets（MNRAS）](https://academic.oup.com/mnras/article/503/3/3204/5989737)
- [Denoising the Deep Sky（论文）](https://arxiv.org/abs/2601.23276)
- [Denoising the Deep Sky（代码与数据说明）](https://github.com/ShuhongLL/Denoising-Deep-Sky)
- [DnCNN 官方仓库](https://github.com/cszn/DnCNN)
- [SCUNet 官方仓库](https://github.com/cszn/SCUNet)
- [Astropy CCD reduction guide](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-01-astronomical-CCD-image-components.html)
