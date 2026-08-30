# 基于 Siril 缩星算法的 Python 实现

本项目将 Siril 社区脚本 `DSA-Star_Reduction.py` 使用的 PixelMath 缩星方法改写为独立的 NumPy 实现，并预留接口调用你自己的去星算法。

## 1. 算法原理

Siril 的自动缩星流程需要两张同尺寸图像：

- `original`：已经拉伸的原图；
- `starless`：去星算法输出的无星图。

Siril 脚本使用的 PixelMath 表达式为：

```text
~((~mtf(~v, original) / ~mtf(~v, starless)) * ~starless)
```

其中：

- `~x` 表示像素反相，即 `1 - x`；
- `v` 是缩星参数；
- `mtf(m, x)` 是 Midtones Transfer Function；
- 图像像素应归一化到 `[0, 1]`。

MTF 定义为：

```text
MTF(m, x) = ((m - 1) * x) / (((2*m - 1) * x) - m)
```

参数规律：

- `v = 0.5`：结果严格等于原图；
- `v < 0.5`：缩星，越接近 0 越强；
- `v > 0.5`：放大星点；
- Siril 脚本默认使用 `v = 0.2`。

这不是形态学腐蚀算法。它利用原图与无星图之间的非线性亮度关系，压低星点外围和中等亮度区域。因为星点的外翼首先变暗，其可见半径随之减小，所以视觉上表现为缩星。背景区域满足 `original ≈ starless`，因此基本保持不变。

代码默认使用与原 PixelMath 表达式代数等价的稳定形式：

```text
m = 1 - v
result = 1 - (1 - original) *
         (m - (2*m - 1)*starless) /
         (m - (2*m - 1)*original)
```

该形式避免了饱和像素附近的 `0/0` 数值问题。代码也保留了 `literal` 模式，用于逐项复现 Siril 表达式并进行结果核对。

## 2. 环境安装

最低依赖：

```bash
pip install numpy
```

处理 FITS：

```bash
pip install astropy
```

处理 TIFF、PNG、JPEG：

```bash
pip install imageio pillow tifffile
```

推荐一次安装：

```bash
pip install numpy astropy imageio pillow tifffile
```

## 3. 输入要求

为了与 Siril 的算法一致，输入应满足：

1. 图像已经完成拉伸，是非线性图像；
2. 原图和无星图尺寸、通道数、通道顺序完全一致；
3. 浮点像素范围为 `[0, 1]`；
4. 去星图应尽量保留原始背景、星云和星系结构；
5. 不建议直接对未经拉伸的线性堆栈图使用该算法。

支持的文件格式：

- FITS：`.fit`、`.fits`、`.fts`；
- NumPy：`.npy`；
- TIFF：`.tif`、`.tiff`；
- PNG、JPEG。

FITS 和 NPY 浮点输入默认必须已经处于 `[0, 1]`。整数图像会根据整数类型的最大值自动归一化。

## 4. 使用预先生成的无星图

假设：

- 原图：`input.fits`
- 无星图：`starless.fits`
- 输出：`reduced.fits`

运行：

```bash
python siril_star_reduction.py \
  --input input.fits \
  --starless starless.fits \
  --output reduced.fits \
  --value 0.2
```

覆盖已有输出：

```bash
python siril_star_reduction.py \
  --input input.fits \
  --starless starless.fits \
  --output reduced.fits \
  --value 0.2 \
  --overwrite
```

## 5. 接入自己的去星算法

### 5.1 Python API 接口

你的去星函数需要满足：

```python
import numpy as np


def my_star_remover(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    输入：float NumPy 图像，范围 [0, 1]
    输出：相同 shape 的无星图，范围 [0, 1]
    """
    starless = your_model_or_algorithm(image, **kwargs)
    return starless
```

在项目中调用：

```python
from siril_star_reduction import siril_star_reduction

reduced = siril_star_reduction(
    original_image,
    value=0.2,
    star_remover=my_star_remover,
    star_remover_kwargs={
        "tile_size": 512,
    },
)
```

也可以直接传入已经计算好的无星图：

```python
reduced = siril_star_reduction(
    original_image,
    value=0.2,
    starless=starless_image,
)
```

`starless` 和 `star_remover` 二选一，不能同时传入。

### 5.2 CLI 动态加载接口

新建 `my_star_remover.py`：

```python
import numpy as np

_MODEL = None


def remove_stars(image: np.ndarray, model_path=None, tile_size=512) -> np.ndarray:
    global _MODEL

    if _MODEL is None:
        _MODEL = load_your_model(model_path)

    starless = run_your_model(
        _MODEL,
        image,
        tile_size=tile_size,
    )
    return np.asarray(starless, dtype=np.float32)
```

通过文件路径调用：

```bash
python siril_star_reduction.py \
  --input input.fits \
  --star-remover ./my_star_remover.py:remove_stars \
  --star-remover-kwargs '{"model_path":"weights/model.pth","tile_size":512}' \
  --output reduced.fits \
  --value 0.2
```

也可以使用可导入模块：

```bash
python siril_star_reduction.py \
  --input input.fits \
  --star-remover my_package.star_removal:remove_stars \
  --output reduced.fits \
  --value 0.2
```

## 6. 参数说明

### `--value`

建议从以下值开始测试：

| 数值 | 效果 |
|---:|---|
| `0.40` | 轻微缩星 |
| `0.30` | 中等缩星 |
| `0.20` | Siril 脚本默认值，较明显 |
| `0.10` | 强缩星，容易暴露去星伪影 |
| `0.50` | 不改变图像 |
| `>0.50` | 放大星点 |

参数必须严格位于 `(0, 1)`。

### `--implementation`

```bash
--implementation stable
```

默认模式。使用代数等价的稳定公式，推荐实际处理时使用。

```bash
--implementation literal
```

逐项计算反相、MTF、比值和重建，主要用于与 Siril PixelMath 逻辑对照。

### `--clip-input`

默认情况下，浮点输入超出 `[0, 1]` 会直接报错，因为自动裁剪可能掩盖输入归一化问题。

确实需要裁剪时使用：

```bash
--clip-input
```

## 7. 数值自检

执行：

```bash
python siril_star_reduction.py --self-test
```

测试内容包括：

- `value=0.5` 时输出等于原图；
- 原图等于无星图时输出不变；
- 稳定公式与 Siril 逐项公式数值一致；
- 合法缩星结果位于原图和无星图之间；
- 输出范围保持在 `[0, 1]`。

## 8. 重要注意事项

### 8.1 去星质量决定最终效果

该算法依赖 `starless` 作为背景估计。如果去星结果存在以下问题，缩星后可能被放大：

- 星点中心残留；
- 大星周围出现黑洞或亮环；
- 星云细节被误删；
- 无星图局部明显亮于原图；
- 原图和无星图没有严格对齐。

代码会在无星图大量像素亮于原图时给出警告，但不会擅自修改你的无星结果。

### 8.2 饱和星点不会被完全消除

该变换对星点外围的压缩最明显。完全饱和的中心像素接近 1 时仍会保持接近 1，因此强缩星后可能留下较小但较硬的亮核。这与 Siril 方法本身的性质一致。

### 8.3 它减少的是可见半径

算法并没有直接移动像素，也没有显式估计星点半径。所谓“缩星”来自星点亮度轮廓被压缩，特别是外翼低于可见阈值，因此视觉半径减小。

### 8.4 彩色图像按通道逐像素处理

RGB 图像会对每个通道应用相同公式。如果去星算法在各通道产生不同残差，大星边缘可能出现色环。遇到这种情况，应优先改进去星结果，也可以在缩星后增加颜色保护或亮度域处理。

### 8.5 高精度格式

PNG/JPEG 输出为了兼容性保存为 8-bit。天文图像处理建议优先保存为：

- 32-bit 浮点 FITS；
- 32-bit 浮点 NPY；
- 16-bit TIFF。

## 9. 代码中的核心调用

最小调用示例：

```python
import numpy as np
from siril_star_reduction import siril_star_reduction

original = np.load("original.npy")
starless = np.load("starless.npy")

result = siril_star_reduction(
    original,
    starless=starless,
    value=0.2,
)

np.save("reduced.npy", result)
```

接入自研去星模型：

```python
import numpy as np
from siril_star_reduction import siril_star_reduction


def remove_stars(image: np.ndarray) -> np.ndarray:
    return model.inference(image)


result = siril_star_reduction(
    original,
    star_remover=remove_stars,
    value=0.2,
)
```

## 10. 星点图层提取与重新合成

项目中的 `star_layer.py` 可以把星点保存成独立的加法图层，并在之后将其叠加到另一张已经对齐的无星图上。该脚本直接复用 `siril_star_reduction.py` 的读写函数和缩星算法。

星点层和最终合成使用以下关系：

```text
原始星点层 = original - starless
缩星星点层 = reduced - starless
最终图像   = clip(target_starless + stars, 0, 1)
```

这里保存的是黑底的“加法星点层”，不是普通 Alpha 透明图层。在 Photoshop 中手工合成时应使用“线性减淡（添加）”，在 GIMP 中应使用“Addition/相加”混合模式。

### 10.1 提取原始大小的星点

```bash
python star_layer.py extract \
  --input original.tif \
  --starless starless.tif \
  --output stars.tif
```

默认的 `--mode original` 会保存完整的原始星点。将该图层重新加到原来的无星图上，可以恢复原图；将其加到经过其他处理的无星图上，可以保留原始大小的星点。

输出文件已经存在时，使用：

```bash
python star_layer.py extract \
  --input original.tif \
  --starless starless.tif \
  --output stars.tif \
  --overwrite
```

### 10.2 提取缩星后的星点

使用 `--mode reduced` 可以先执行 Siril 兼容缩星，再把缩小后的星点单独保存：

```bash
python star_layer.py extract \
  --input original.tif \
  --starless starless.tif \
  --output reduced_stars.tif \
  --mode reduced \
  --value 0.2
```

此时缩星效果已经固化在 `reduced_stars.tif` 中。重新合成时不需要再次指定 `--value`。

`--mode reduced` 还支持：

```text
--implementation stable|literal
--clip-input
```

含义与主缩星脚本中的同名参数一致。

### 10.3 将星点层粘贴到无星图

```bash
python star_layer.py compose \
  --starless new_starless.tif \
  --stars stars.tif \
  --output recombined.tif
```

目标无星图必须与星点层满足以下条件：

- 图像宽度、高度和通道数完全一致；
- 星点位置严格对齐；
- 使用相同的拉伸状态、通道顺序和色彩空间；
- 如果无星背景被大幅调亮，重新加入星点后可能发生高光裁剪。

合成结果默认裁剪到 `[0, 1]`。对于需要保留超范围浮点数据的 FITS/NPY，可以使用：

```bash
python star_layer.py compose \
  --starless new_starless.fits \
  --stars stars.fits \
  --output recombined.fits \
  --no-clip-output
```

### 10.4 Python API

```python
import numpy as np
from star_layer import compose_star_layer, extract_star_layer

original = np.load("original.npy")
starless = np.load("starless.npy")

# 原始大小的星点
stars = extract_star_layer(
    original,
    starless,
    mode="original",
)

# 或者提取 value=0.2 的缩星星点
reduced_stars = extract_star_layer(
    original,
    starless,
    mode="reduced",
    value=0.2,
)

result = compose_star_layer(starless, reduced_stars)
```

`extract_star_layer` 和 `compose_star_layer` 默认返回 `float32` 数组。

### 10.5 输出格式和负残差

推荐的星点层格式：

- 16-bit TIFF：适合在图像编辑软件中作为黑底加法图层使用；
- 32-bit FITS：保留浮点精度、FITS 头信息和负残差；
- 32-bit NPY：适合继续通过 Python 处理；
- PNG/JPEG：当前读写代码只保存为 8-bit，不推荐用于高精度星点层。

如果无星图局部比原图或缩星结果更亮，差值中会出现负残差。脚本不会把带负值的星点层静默保存成 TIFF、PNG 或 JPEG，因为这些格式在当前流程中会裁掉负数。此时可以：

1. 使用 FITS 或 NPY 保存，以获得可逆的有符号浮点星点层；
2. 使用 `--clip-negative` 把负残差裁剪为零，得到常规黑底图层。

例如：

```bash
python star_layer.py extract \
  --input original.tif \
  --starless starless.tif \
  --output stars.tif \
  --clip-negative
```

裁剪负残差后，发生裁剪的位置将不能再严格恢复原图。星点差值层还可能包含星晕和去星算法产生的局部差异，因此它是“星点残差层”，并不等同于只包含离散星核的检测蒙版。

## 11. 参考来源

- Siril 官方教程：Automated Star Reduction Script  
  https://siril.org/tutorials/star-reduction/
- Siril 社区脚本：`DSA-Star_Reduction.py`  
  https://gitlab.com/free-astro/siril-scripts/-/raw/main/processing/DSA-Star_Reduction.py
- Siril 官方文档：Image Stretching / MTF  
  https://siril.readthedocs.io/en/stable/processing/stretching.html
- Siril 官方 PixelMath 文档  
  https://siril.readthedocs.io/en/latest/processing/pixelmath.html

本实现是独立的 NumPy 重写版本，不依赖 Siril 程序本体，但算法行为以 Siril 社区缩星脚本的 PixelMath 表达式为依据。
