# 星点反卷积修复 Demo（效果优先版）

这是一个可解释的星点引导反卷积实现。它从孤立星点自动估计全局 PSF，也支持输入标定 PSF；
质量模式对 RGB 三通道分别执行带阻尼 Richardson–Lucy（RL）迭代，并恢复到有限目标 PSF。
它借鉴 BlurXTerminator（BXT）公开的成像模型与工程思想，但不含、不复刻其专有神经网络。

## 1. 原理与 ML5 调研结论

天文成像可以近似写为：

```text
observed = latent * source_PSF + noise
restored = latent * target_PSF
```

反卷积不是普通锐化。普通锐化只增强高频；反卷积则使用已知或估计的 PSF 逆转具体的成像模糊。
由于噪声会使零宽度目标成为严重病态问题，真实数据通常应恢复到一个有限、可控的 target PSF。

BXT ML5 的公开说明揭示了几个重要设计方向：

- 它是针对天文数据训练的 semi-blind 模型，会在图内识别合适的参考星并推断 PSF；
- 空间变化像差需要局部 PSF 与重叠 tile，单个全局卷积无法完整表达；
- stellar 与 nonstellar 分量可以使用不同的恢复目标；
- ML5 约 4500 万参数、接近每像素 100 万次运算；训练损失是对已知 ground truth 的纯重建损失，
  没有 perceptual、SSIM 或 adversarial loss；
- 恒星颜色、致密星场、衍射尖刺和暗结构保护都需要专门训练数据，而不是靠更猛烈的锐化替代。

本 demo 因此采用“物理 PSF + 纯重建验证”的传统路径：从星点估计经验 PSF，或读取标定 PSF，
再用正则化求解器恢复。它能验证反卷积引擎和数据流，但自动 PSF 仍是全局估计，达不到 ML5 在复杂、
空间变化像差上的泛化能力。

## 2. 本轮效果优先改动

旧版输出接近 blur 的主要原因不是 RL 无效，而是只解一次 luminance，并把非星结构强度乘在稀疏
边缘蒙版上，绝大部分图像最终仍取原始模糊像素。本版已改为：

```text
输入 float32 图像
  -> 星点检测、筛选与亚像素配准
  -> robust median 经验 PSF（或外部标定 PSF）
  -> quality: RGB 三通道分别做阻尼 RL
  -> balanced: 单次 luminance RL 后回写 RGB
  -> fast: 正则化 Wiener PSF-to-target
  -> stellar / nonstellar 全视场连续强度融合
  -> 16-bit TIFF 或 float32 FITS/NPY 输出
```

质量模式的计算量约为旧版 luminance 路径的三倍，但恢复差异清晰可见。`balanced` 适合预览，
`fast` 适合快速搜索参数；最终输出推荐 `quality`。

## 3. 环境与运行

```powershell
conda activate study
cd D:\Workman\MyProject\ImageStack\AstroColorProcess\StarDeconvolution
python demo.py --overwrite
```

demo 会同时生成两种结果：

- `03_restored_auto_psf.tif`：仅从退化图像自动估计 PSF，代表实际自动流程；
- `04_restored_known_psf.tif`：使用 demo 注入的精确退化 kernel，代表求解器上限，不冒充自动结果；
- `comparison_hd.png`：4000 × 1000 原生像素四联图，可放大检查；
- `metrics.json`：两条路径各自的质量、耗时与配置。

按项目现有流程“先拉伸 FITS，再做星点反卷积”：

```powershell
python stretch_then_deconvolve.py `
  --input input.fits `
  --output-dir stretched_outputs `
  --prefix target_name `
  --target-fwhm 2.0 `
  --iterations 20 `
  --overwrite
```

该脚本直接调用
`D:\Workman\MyProject\ImageStack\RASPAstroStacker\ImageStretch` 的 MTF 实现，默认参数为
`shadow_co=-2.0`、`color_calib=False`。实际反卷积输入是该函数返回的 uint8 拉伸图，经 RGB 通道转换
并归一化到 `[0, 1]`；不是仅为预览而进行的显示拉伸。输出包含拉伸输入 TIFF、反卷积后的 TIFF/FITS、
经验/目标 PSF、星点蒙版、原生尺寸对比图和 JSON 报告。

处理自己的线性 FITS（自动 PSF）：

```powershell
python star_deconvolution.py `
  --input input.fits `
  --output restored.fits `
  --solver quality `
  --iterations 40 `
  --target-fwhm 1.35 `
  --stellar-strength 0.82 `
  --nonstellar-strength 0.65 `
  --diagnostics-dir diagnostics `
  --overwrite
```

若已从同一光学系统的实测星点、波前或标定帧得到 PSF，可输入椭圆 Gaussian 近似参数：

```powershell
python star_deconvolution.py `
  --input input.fits `
  --output restored.fits `
  --solver quality `
  --iterations 60 `
  --psf-fwhm 4.2 `
  --psf-axis-ratio 0.62 `
  --psf-angle 28 `
  --target-fwhm 1.3 `
  --noise-sigma 0.0018 `
  --overwrite
```

`--target-fwhm 0` 只应用于“提供的是额外退化 kernel，目标是其卷积前图像”的校准实验。对未知真实
光学 PSF 不建议使用零宽度目标，因为它更容易放大噪声、振铃和黑环。

Python API 也可直接传入任意归一化、奇数方形 PSF：

```python
result, report, diagnostics = star_deconvolution(
    image_float32,
    DeconvolutionConfig(solver="quality", iterations=50, target_fwhm_px=1.3),
    source_psf=measured_psf,
)
```

## 4. 当前 benchmark

测试图为从 CDS 下载的 1000 × 1000 M45 DSS2 星场。demo 注入 FWHM 4.2 px、轴比 0.62、
28° 的椭圆 Gaussian 模糊和 sigma=0.0018 固定随机噪声。当前 `study` 环境实测：

| 路径 | PSNR | 相对 blur | SSIM | 相对 blur | CPU 时间 |
|---|---:|---:|---:|---:|---:|
| Synthetic blur | 33.980 dB | — | 0.90537 | — | — |
| Auto PSF / quality | 36.828 dB | **+2.847 dB** | 0.93667 | **+0.03130** | 5.46 s |
| Known PSF / solver ceiling | 38.085 dB | **+4.105 dB** | 0.94479 | **+0.03942** | 9.15 s |

两条路径连续执行时，Python `tracemalloc` 记录的峰值分配约 118.8 MB。数值以
`outputs/metrics.json` 为准；硬件、OpenCV 构建和后台负载会影响耗时。

已知 PSF 路径用于证明求解器确实能逆转模糊。自动路径与它之间约 1.26 dB 的差距主要来自经验 PSF
同时包含原图本身的光学 PSF、JPEG/拉伸影响和本次额外退化，因此不是精确的增量 kernel。

## 5. 参数与风险

| 档位 | solver | iterations | target FWHM | stellar | nonstellar | 用途 |
|---|---|---:|---:|---:|---:|---|
| 快速预览 | fast | — | 1.6–2.0 | 0.55 | 0.35 | 秒级找方向 |
| 平衡 | balanced | 20–35 | 1.4–1.8 | 0.75 | 0.45 | 较低内存/开销 |
| 效果优先 | quality | 40–70 | 1.2–1.5 | 0.75–0.9 | 0.55–0.75 | 最终输出 |

使用线性 32-bit FITS 会比已拉伸 JPEG 更可靠。出现黑环、噪声颗粒或亮星棋盘状伪影时，应降低
iterations / strength，增大 target FWHM 或 damping。饱和核心没有可恢复信息；本实现不会凭空重建
被裁掉的星核。

## 6. 已知边界与下一阶段

- 自动路径目前使用单个全局 PSF；严重场曲、彗差和传感器倾斜应增加重叠 tile + 局部 PSF 插值；
- 密集星团、强星云或少星视场会污染经验 PSF，最好提供标定 PSF；
- 传统 RL 对 PSF 偏差敏感；要进一步接近 BXT ML5，需要成对的 clean/degraded 线性天文训练集、
  空间变化退化生成器、独立 stellar/nonstellar 目标和纯 ground-truth 重建训练；
- 当前 demo 是效果可验证的传统基线，不声称具备 ML5 的 4500 万参数模型能力。

## 7. 资料与样例来源

- [BlurXTerminator ML5](https://www.rc-astro.com/blurxterminator-ml5/)
- [The Mathematics of BlurXTerminator](https://www.rc-astro.com/the-mathematics-of-blurxterminator/)
- [Richardson 1972](https://doi.org/10.1364/JOSA.62.000055)
- [Lucy 1974](https://doi.org/10.1086/111605)
- [Lauer 2002: Deconvolution With a Spatially-Variant PSF](https://arxiv.org/abs/astro-ph/0208247)
- [STARRED: two-channel deconvolution](https://arxiv.org/abs/2305.18526)
- [CDS HiPS / DSS2](https://aladin.cds.unistra.fr/hips/)
- [NASA Hubble M51](https://science.nasa.gov/asset/hubble/hubble-acs-visible-image-of-m51/)

样例的精确下载 URL、署名、日期与 SHA-256 记录在 `samples/SOURCE.md`。
