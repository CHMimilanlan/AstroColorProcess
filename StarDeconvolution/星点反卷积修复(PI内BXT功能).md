# 天文图像星点反卷积修复功能研究技术路线文档
## 1. 研究背景
在深空天文图像处理中，恒星可以近似看作点光源，但实际成像过程中，星点会受到大气湍流、望远镜光学系统、对焦误差、导星误差、相机采样和后期叠加等因素影响，从理想点源扩散为具有一定宽度和形状的亮斑。该扩散过程通常由点扩散函数，即 PSF，Point Spread Function 描述。

实际观测图像可以简化表示为：

[  
h = f * g + n  
]

其中，(f) 表示理想天文图像，(g) 表示成像系统和大气共同造成的 PSF，(n) 表示噪声，(h) 表示实际观测图像。星点反卷积修复的目标，是在不凭空生成虚假细节的前提下，尽可能减弱 PSF 模糊带来的影响，使星点更小、更圆、更自然，同时恢复星云、星系等非星点结构中已经存在但被模糊压低的局部对比度。

PixInsight 中的 BlurXTerminator，以下简称 BXT，是一个具有代表性的 AI 辅助反卷积工具。其核心思想不是将图像恢复到完全无模糊的理想状态，而是将原始较差 PSF (g) 转换为更小、更干净、更可控的目标 PSF (g')。因此，其思想可以抽象为：

[  
f * g + n \rightarrow f * g' + n  
]

该思想对于本项目具有重要参考价值。本项目拟从两个层面开展研究：第一阶段构建快速、可控、效果良好的传统算法 demo；第二阶段在传统算法和数据构造能力成熟后，进一步研发自研 AI 星点反卷积修复模型。

---

## 2. 研究目标
本项目的总体目标是实现一个面向天文图像的星点反卷积修复功能，能够对输入天文图像中的星点模糊、星点膨胀、轻微拖线、轻微椭圆化、边角局部变形等问题进行修复，并在一定程度上提升星云、星系等非星点结构的细节表现。

具体目标包括：

1. 构建传统算法 demo，实现星点检测、PSF 估计、PSF kernel 构建、反卷积、mask 融合和结果评价。
2. 构建星点质量评估模块，能够评估星点 FWHM、偏心率、圆度、峰值集中度、残差和反卷积前后变化。
3. 构建可扩展的局部 PSF 估计流程，使算法能够处理图像不同区域 PSF 不一致的问题。
4. 构建星点区域和非星点区域的分离处理策略，实现 stellar 与 nonstellar 的不同强度反卷积。
5. 在未来阶段构建自研 AI 数据集生成流程，利用模拟退化 PSF 构造输入和 ground truth。
6. 训练自研 AI 模型，实现从坏 PSF 图像到目标 PSF 图像的学习型反卷积修复。
7. 建立客观评价指标和伪影检测机制，避免过锐化、黑圈、振铃、噪声放大和虚假细节生成。

---

## 3. 总体技术路线
本项目分为两条路线并行推进：

```plain
路线 A：快速 demo 实现
目标：短期内实现可展示、可调试、效果较好的星点反卷积修复功能
方法：星点检测 + PSF 估计 + Richardson-Lucy / Wiener 反卷积 + mask 融合

路线 B：自研 AI 模型实现
目标：长期实现类似 BXT 思想的自动化 AI 反卷积模型
方法：构造 synthetic paired dataset，训练模型学习 f*g_bad+n → f*g_target+n
```

两条路线共用以下基础模块：

```plain
图像读取与归一化
背景估计与预处理
星点检测
星点候选验证
PSF 参数拟合
PSF kernel 构建
局部 tile 划分与融合
star mask / halo mask / background mask
评价指标
伪影检测
```

这样，快速 demo 阶段开发出的星点检测、PSF 估计和评价模块，可以直接服务于后续 AI 数据集构建、模型条件输入和推理质量控制。

---

# 第一部分：快速 demo 技术路线
## 4. 快速 demo 的定位
快速 demo 阶段不追求完全复现 BXT，而是先实现一个可解释、可调试、可控的传统反卷积系统。

该阶段的功能目标是：

```plain
输入一张天文图像
  ↓
自动检测星点
  ↓
筛选高质量星点
  ↓
估计全局或局部 PSF
  ↓
构建 PSF kernel
  ↓
执行反卷积
  ↓
通过 soft mask 融合结果
  ↓
输出星点更小、细节更清楚、伪影可控的结果
```

该阶段的优点是实现速度快、可解释性强、便于做 ablation 和参数调试，也有利于后续建立 AI 训练数据构造流程。

---

## 5. 输入图像与预处理
建议 demo 阶段优先处理线性阶段的天文图像，即完成基础校准、叠加、背景校正和颜色校准，但尚未进行强拉伸、强降噪和强锐化的图像。

输入格式可支持：

```plain
FITS
TIFF
PNG/JPEG，仅用于可视化 demo，不建议作为高精度反卷积输入
```

预处理步骤包括：

```plain
读取图像
  ↓
转为 float32
  ↓
提取 luminance 或灰度图
  ↓
背景估计
  ↓
背景扣除或背景归一化
  ↓
负值裁剪
  ↓
归一化到 [0, 1]
```

背景估计可以采用大尺度 median filter、morphological opening、rolling ball 或低阶多项式拟合。该步骤的目的不是增强图像，而是让星点检测和 PSF 拟合不受背景梯度干扰。

---

## 6. 星点检测与候选筛选
星点检测是整个系统的基础。建议采用如下流程：

```plain
灰度图 / luminance 图
  ↓
背景估计与扣除
  ↓
DoG / LoG / top-hat 增强星点
  ↓
阈值分割
  ↓
连通域分析
  ↓
面积、峰值、SNR、圆度初筛
  ↓
高斯模板匹配验证
  ↓
输出高质量星点候选
```

已有的 `PSF_check()` 函数可以作为星点候选验证模块。该函数的作用是判断候选区域是否接近固定圆形高斯星点，其定位应为：

```plain
star candidate filtering
```

而不是最终 PSF 估计模块。

该函数适合过滤：

```plain
热噪声点
坏点
非星点小结构
星云局部亮斑
形状不合理的候选区域
```

但它目前只使用固定 sigma 的圆形高斯模板，因此无法描述真实星点的椭圆形、Moffat 长尾、旋转角度和局部像差。后续需要在该函数后加入真正的 PSF fitting 模块。

---

## 7. PSF 参数拟合
在保留高质量星点后，需要对每颗星点进行参数拟合。

### 7.1 椭圆 Gaussian 拟合
第一版可使用椭圆 Gaussian 模型：

[  
I(x,y)=B+A\exp\left(-\frac{1}{2}  
\left[  
\frac{x'^2}{\sigma_x^2}+  
\frac{y'^2}{\sigma_y^2}  
\right]\right)  
]

其中：

```plain
B：背景亮度
A：星点峰值
x0, y0：亚像素中心
sigma_x, sigma_y：两个主轴方向的宽度
theta：旋转角度
```

拟合后可计算：

[  
FWHM_x = 2.355\sigma_x  
]

[  
FWHM_y = 2.355\sigma_y  
]

以及偏心率、圆度和拟合残差。

### 7.2 Moffat 拟合
真实天文星点通常具有比 Gaussian 更长的尾部，因此第二版建议加入 Moffat 模型：

[  
I(r)=B+A\left[1+\left(\frac{r}{\alpha}\right)^2\right]^{-\beta}  
]

Moffat 模型能够更好描述大气 seeing 和星点 halo，因此更适合生成用于反卷积的 PSF kernel。

### 7.3 星点评估与过滤
每颗星点拟合后，输出如下参数：

```plain
x0, y0
amplitude
background
FWHM_x
FWHM_y
theta
eccentricity
roundness
SNR
residual
saturation flag
neighbor contamination flag
```

剔除以下星点：

```plain
饱和星
太暗星
粘连星
残差过大的星
偏心率过高的异常星
落在强星云结构上的星
靠近边缘导致 patch 不完整的星
```

最终保留稳定星点用于 PSF 估计。

---

## 8. 全局 PSF 与局部 PSF
### 8.1 全局 PSF
Demo v1 使用全局 PSF。对所有高质量星点的 FWHM、theta、eccentricity、Moffat beta 等参数取 median，得到整张图的代表性 PSF。

流程如下：

```plain
高质量星点集合
  ↓
逐星拟合 PSF 参数
  ↓
剔除异常值
  ↓
统计 median FWHM / eccentricity / theta / beta
  ↓
生成全局 PSF kernel
```

优点是实现简单、稳定、计算成本低。缺点是无法处理图像边角 PSF 与中心 PSF 不一致的问题。

### 8.2 局部 PSF
Demo v2 实现局部 PSF。将图像分成多个重叠 tile，每个 tile 独立估计 PSF：

```plain
输入图像
  ↓
划分 overlapping tiles
  ↓
每个 tile 内检测星点
  ↓
每个 tile 拟合局部 PSF
  ↓
星点不足 tile 使用邻域插值或全局 PSF
  ↓
每个 tile 生成局部 PSF kernel
```

局部 PSF 的意义在于处理：

```plain
边角彗差
像散
场曲
相机倾斜
局部拖线
不同区域 seeing 或散射变化
```

该方案更接近 BXT 的局部 PSF 思想，也更适合处理真实天文图像中空间变化 PSF 的问题。

---

## 9. PSF kernel 构建
根据拟合参数生成二维 PSF kernel。第一版可以支持：

```plain
圆形 Gaussian PSF
椭圆 Gaussian PSF
圆形 Moffat PSF
椭圆 Moffat PSF
运动模糊 kernel
```

kernel 需要满足：

```plain
尺寸足够覆盖星点尾部
中心对齐
总和归一化为 1
支持奇数尺寸，例如 15x15, 21x21, 31x31
```

如果使用局部 PSF，则每个 tile 对应一个 kernel。

---

## 10. 反卷积算法
### 10.1 Richardson-Lucy 反卷积
第一版推荐使用 Richardson-Lucy，原因是实现简单、效果直观、适合快速验证。

基础形式：

```plain
image + psf kernel
  ↓
Richardson-Lucy iterations
  ↓
deconvolved image
```

需要控制：

```plain
迭代次数
反卷积强度
是否 clip
是否加入正则化
是否只在 mask 区域融合结果
```

常见问题：

```plain
迭代过多导致噪声放大
星点周围出现黑圈
亮星出现 ringing
背景颗粒变强
星云边缘产生伪结构
```

因此 demo 阶段不建议直接将反卷积结果全图替换原图，而应采用 mask 融合。

### 10.2 Wiener 反卷积
第二版可以加入 Wiener 反卷积作为对比。Wiener 方法对噪声有更明确的频域抑制机制，但需要估计噪声水平，参数敏感。其优势是可作为 Richardson-Lucy 的补充 baseline。

### 10.3 正则化反卷积
后续可以加入 TV regularization、wavelet regularization 或 starlet regularization，用于抑制噪声放大和振铃伪影。

---

## 11. Mask 控制与结果融合
反卷积不能无差别作用于全图，否则容易放大背景噪声。因此需要构建多个 soft mask：

```plain
star core mask
star halo mask
nonstellar structure mask
background mask
```

融合策略：

[  
output = M \cdot deconv + (1-M)\cdot image  
]

其中 (M) 是 soft mask，而不是二值硬 mask。

推荐流程：

```plain
星点检测结果
  ↓
生成 star core mask
  ↓
dilate 得到 halo mask
  ↓
Gaussian blur 羽化
  ↓
得到 soft star mask
```

对于星点区域：

```plain
星点核心：中等反卷积，目标是缩小星点
星点 halo：弱反卷积或保护，避免黑圈
背景区域：基本不处理，避免噪声放大
```

对于非星点区域：

```plain
星云、星系结构：适度反卷积或局部锐化
背景平坦区域：弱处理或不处理
```

---

## 12. Stellar / Nonstellar 分离处理
Demo v3 可以引入星点层与非星点层分离思想：

```plain
输入图像
  ↓
星点检测 / star mask
  ↓
生成 starless 图像
  ↓
stellar layer = image - starless
  ↓
stellar layer 做星点反卷积或缩星
  ↓
starless layer 做非星点细节增强
  ↓
重组输出
```

如果暂时没有自研 starless 模型，可以使用传统 inpainting、morphological filtering 或外部去星工具作为 demo 辅助。长期来看，stellar / nonstellar 分离可以与自研 AI 模型结合，形成双分支网络结构。

---

## 13. 快速 demo 阶段版本规划
### Demo v1：全局 PSF + Richardson-Lucy
目标：快速实现可展示效果。

```plain
输入图像
  ↓
星点检测
  ↓
高质量星点筛选
  ↓
全局 PSF 拟合
  ↓
Richardson-Lucy 反卷积
  ↓
star mask 融合
  ↓
FWHM 前后对比
```

交付物：

```plain
原图
星点检测图
PSF kernel 可视化
反卷积结果
FWHM before/after 统计
```

### Demo v2：局部 tile PSF + 局部反卷积
目标：提升边角星点修复效果。

```plain
输入图像
  ↓
overlapping tile 划分
  ↓
每个 tile 估计局部 PSF
  ↓
每个 tile 独立反卷积
  ↓
Hann / Gaussian window 融合
  ↓
输出完整图像
```

交付物：

```plain
局部 PSF map
各 tile FWHM 分布
局部反卷积结果
边角 before/after 对比
```

### Demo v3：stellar / nonstellar 分离处理
目标：提升视觉自然度，避免星点和星云使用同一强度处理。

```plain
输入图像
  ↓
星点层 / 非星点层分离
  ↓
stellar branch：星点缩小和形状修复
  ↓
nonstellar branch：星云细节增强
  ↓
结果重组
```

交付物：

```plain
star mask
stellar layer
starless layer
recombined result
星点与星云分别处理的 ablation
```

---

# 第二部分：自研 AI 模型技术路线
## 14. AI 路线总体思想
自研 AI 模型不应简单训练成普通锐化网络，而应以物理成像模型为基础，学习从坏 PSF 图像到目标 PSF 图像的映射。

训练输入：

[  
x = f * g_{bad} + n  
]

训练标签：

[  
y = f * g_{target} + n  
]

其中：

```plain
f：相对清晰的高质量天文图像
g_bad：模拟退化 PSF
g_target：更小、更圆、更理想的目标 PSF
n：噪声
```

该设计的核心是：ground truth 不是完全无 PSF 的理想图，而是目标 PSF 下的可实现图像。这可以避免模型学习到不自然的无限锐化结果。

---

## 15. AI 数据集构建
### 15.1 Pseudo-clean 图像收集
收集高质量天文图像作为 pseudo-clean 图像 (f)。图像应尽量满足：

```plain
信噪比较高
星点较小
细节清楚
无严重拖线
无明显过锐化伪影
背景较干净
线性或接近线性数据优先
```

数据来源可以包括：

```plain
高质量公开天文图像
自采高质量叠加图
经过人工筛选的优秀样本
合成星点和星云图
```

### 15.2 退化 PSF 采样
对 pseudo-clean 图像随机采样坏 PSF：

```plain
Gaussian PSF
Moffat PSF
椭圆 Gaussian
椭圆 Moffat
motion blur kernel
defocus kernel
coma-like asymmetric kernel
astigmatism-like kernel
RGB channel-dependent kernel
spatially varying PSF
```

退化参数包括：

```plain
FWHM_x
FWHM_y
ellipticity
theta
Moffat beta
motion length
motion angle
defocus radius
chromatic shift
noise level
background gradient
```

### 15.3 训练输入生成
输入图像：

[  
x = f * g_{bad} + n  
]

其中噪声可以包括：

```plain
Poisson noise
Gaussian read noise
background noise
hot pixels，可选
compression artifacts，可选
```

### 15.4 Ground truth 生成
标签图像：

[  
y = f * g_{target} + n  
]

目标 PSF (g_{target}) 应满足：

```plain
比 bad PSF 更小
比 bad PSF 更圆
无明显拖线
无明显彗差
保留合理有限宽度
不追求零宽度 PSF
```

例如：

```plain
bad FWHM = 5.0 px, ellipticity = 0.5
target FWHM = 2.0 px, ellipticity = 0.05
```

### 15.5 Stellar / Nonstellar 分离标签
进一步可以将图像拆分为：

[  
f = f_s + f_{ns}  
]

其中：

```plain
f_s：星点层
f_ns：星云、星系、尘埃等非星点层
```

训练目标变为：

[  
y = f_s * g_s' + f_{ns} * g_{ns}' + n  
]

这样可以让模型学习：

```plain
星点：适度缩小、变圆、减少畸变
非星点：恢复细节对比度，但避免过锐化
背景：尽量保持干净，不放大噪声
```

---

## 16. AI 模型结构设计
### 16.1 AI v1：Residual U-Net Baseline
第一版使用简单稳定的 residual U-Net：

```plain
输入：模糊图像 patch
输出：目标 PSF 图像 patch
```

模型预测 residual：

[  
\hat{y}=x+R(x)  
]

优点：

```plain
实现简单
训练稳定
容易验证 AI 路线可行性
不容易大幅改坏原图
```

### 16.2 AI v2：PSF-conditioned Network
第二版加入 PSF 条件输入，让模型可控。

输入包括：

```plain
image patch
star mask
estimated bad PSF parameters
target PSF parameters
noise level map
position encoding
```

PSF 条件可以包括：

```plain
bad FWHM
target FWHM
ellipticity
theta
noise level
stellar strength
nonstellar strength
```

条件注入方式可以采用：

```plain
FiLM
AdaIN
condition embedding
cross-attention
feature concatenation
```

该阶段目标是让模型不只是做固定锐化，而是根据当前 PSF 和目标 PSF 进行可控修复。

### 16.3 AI v3：Stellar / Nonstellar 双分支模型
第三版构建双分支模型：

```plain
输入图像
  ↓
shared encoder
  ↓
stellar branch
nonstellar branch
  ↓
feature fusion
  ↓
输出修复图像
```

或者采用显式分层：

```plain
输入图像
  ↓
star mask / starless 分离
  ↓
stellar network 处理星点层
  ↓
nonstellar network 处理非星点层
  ↓
重组输出
```

该结构更接近 BXT-like 思路，可以分别控制星点和非星点结构的修复强度。

---

## 17. AI Loss 设计
基础重建损失：

[  
\mathcal{L}_{rec}=|\hat{y}-y|_1  
]

推荐使用 L1 或 Charbonnier loss，而不是单纯 MSE，因为 MSE 容易使结果过平滑。

### 17.1 梯度损失
用于增强结构恢复：

# [  
\mathcal{L}_{grad}
|\nabla \hat{y}-\nabla y|_1  
]

### 17.2 Mask 加权损失
针对星点、非星点和背景分别设置权重：

# [  
\mathcal{L}_{mask}
\lambda_s |M_s(\hat{y}-y)|_1  
__+  
__\lambda_{ns} |M_{ns}(\hat{y}-y)|_1  
+  
\lambda_b |M_b(\hat{y}-y)|_1  
]

其中：

```plain
M_s：星点区域 mask
M_ns：非星点结构 mask
M_b：背景 mask
```

背景权重应较低，避免模型在背景噪声中生成细节。

### 17.3 Flux Consistency Loss
用于约束星点总亮度不发生明显漂移：

# [  
\mathcal{L}_{flux}
## \left|  
\sum_{M_s}\hat{y}
\sum_{M_s}y  
\right|  
]

该损失可以防止模型在缩星时过度改变星点亮度。

### 17.4 Forward Consistency Loss
将输出重新退化回输入空间：

[  
\hat{x} = \hat{y} * g_{target\rightarrow bad}  
]

要求：

# [  
\mathcal{L}_{forward}
|\hat{x}-x|_1  
]

该损失用于约束模型不要生成无法解释原始输入的结构。

### 17.5 总损失
最终可以组合为：

# [  
\mathcal{L}
\lambda_1\mathcal{L}_{rec}  
__+  
__\lambda_2\mathcal{L}_{grad}  
+  
\lambda_3\mathcal{L}_{mask}  
__+  
__\lambda_4\mathcal{L}_{flux}  
+  
\lambda_5\mathcal{L}_{forward}  
]

早期不建议使用 GAN loss 或强 perceptual loss，因为这类损失可能鼓励模型生成看似锐利但不一定真实存在的结构。

---

## 18. AI 推理流程
AI 推理阶段建议采用 patch-based 或 tile-based 处理：

```plain
输入大图
  ↓
切分 overlapping patches
  ↓
每个 patch 估计局部 PSF / noise
  ↓
模型推理
  ↓
weighted blending
  ↓
伪影检测
  ↓
输出完整图像
```

为了避免模型过度修改图像，可以加入 strength 参数：

[  
output = \alpha \cdot model_output + (1-\alpha) \cdot input  
]

其中 (\alpha) 控制修复强度。

---

# 第三部分：评价指标与实验设计
## 19. 星点评价指标
星点修复效果不能只依赖肉眼判断，需要量化评估。

建议指标包括：

```plain
FWHM before / after
FWHM reduction ratio
eccentricity before / after
roundness improvement
star flux preservation
star residual after PSF fitting
halo intensity change
black ring score
```

理想结果是：

```plain
FWHM 适度下降
偏心率下降
星点更圆
星点总 flux 基本保持
无明显黑圈
无明显 ringing
```

## 20. 背景与噪声评价指标
反卷积容易放大背景噪声，因此需要评估：

```plain
background std before / after
high-frequency noise ratio
background smoothness
noise amplification factor
tile boundary score
```

理想结果是：

```plain
背景噪声不明显增加
无棋盘格
无 tile 边界
无颗粒化增强
```

## 21. 非星点结构评价指标
对星云、星系、尘埃结构评估：

```plain
local contrast improvement
edge sharpness
gradient consistency
structure preservation
ringing around bright structures
```

目标是增强真实细节对比度，而不是制造虚假纹理。

## 22. Forward Consistency 检查
对输出结果重新卷积，检查是否能解释输入：

```plain
output
  ↓
用估计 PSF 重新模糊
  ↓
reblurred output
  ↓
与 input 比较
```

如果重新模糊后与输入差异过大，说明模型或算法可能生成了输入中不支持的结构。

---

# 第四部分：项目阶段计划
## 23. 阶段一：传统 demo 最小闭环
目标：实现从输入图像到反卷积结果的最小可用系统。

任务：

```plain
图像读取与归一化
星点检测
PSF_check 候选过滤
椭圆 Gaussian 拟合
全局 PSF 统计
Richardson-Lucy 反卷积
star mask 融合
FWHM 前后对比
```

交付：

```plain
demo_global_rl.py
星点检测可视化
PSF kernel 可视化
反卷积 before/after
FWHM 统计表
```

## 24. 阶段二：局部 PSF 与伪影控制
目标：提升真实复杂图像中的效果。

任务：

```plain
tile-based PSF estimation
局部 PSF map
overlapping tile deconvolution
weighted blending
黑圈检测
噪声放大检测
参数自动推荐
```

交付：

```plain
demo_tile_rl.py
局部 PSF 可视化
边角修复对比
伪影检测结果
```

## 25. 阶段三：stellar / nonstellar 分离处理
目标：实现星点和非星点不同强度处理。

任务：

```plain
star mask 优化
halo mask 构建
starless 层生成
stellar layer 修复
nonstellar layer 细节增强
重组输出
```

交付：

```plain
demo_stellar_nonstellar.py
star layer / starless layer 可视化
分支处理 ablation
```

## 26. 阶段四：AI 数据集构建
目标：构建自研 AI 训练所需 paired dataset。

任务：

```plain
收集 pseudo-clean 天文图像
构建 PSF degradation library
模拟 Gaussian / Moffat / elliptical / motion / defocus / chromatic PSF
生成 x = f*g_bad+n
生成 y = f*g_target+n
保存 metadata
可视化数据质量
```

交付：

```plain
dataset_builder.py
psf_degradation.py
paired patch dataset
metadata json
数据集可视化报告
```

## 27. 阶段五：AI baseline 模型
目标：训练第一个可用 AI 反卷积模型。

任务：

```plain
Residual U-Net
L1 / Charbonnier loss
star mask weighted loss
灰度 luminance 输入
patch-based training
patch-based inference
```

交付：

```plain
train_unet.py
infer_unet.py
AI baseline checkpoint
AI vs RL 对比实验
```

## 28. 阶段六：BXT-like 自研模型
目标：实现可控、局部、自适应的 AI 星点反卷积模型。

任务：

```plain
PSF-conditioned network
stellar / nonstellar dual branch
局部 PSF map 条件输入
forward consistency loss
flux consistency loss
tile 推理融合
强度参数控制
```

交付：

```plain
train_psf_conditioned.py
infer_bxt_like.py
局部 AI 修复结果
星点/非星点可控修复结果
完整技术报告
```

---

# 第五部分：风险与解决策略
## 29. 风险一：PSF 估计不稳定
问题：

```plain
星点数量不足
星点饱和
星点粘连
背景复杂
边角 PSF 过度异常
```

解决：

```plain
增加星点筛选规则
引入 Moffat 拟合
使用多星点 median 统计
局部 tile 星点不足时使用邻域插值
加入拟合残差过滤
```

## 30. 风险二：反卷积产生黑圈和 ringing
问题：

```plain
迭代过多
PSF kernel 不准确
亮星 halo 处理过强
mask 边界太硬
```

解决：

```plain
限制迭代次数
引入 soft mask
halo 区域弱处理
使用正则化反卷积
加入 black ring score 自动检测
```

## 31. 风险三：背景噪声被放大
问题：

```plain
反卷积作用于背景
高频噪声被当成细节
AI 模型过拟合噪声
```

解决：

```plain
背景 mask 保护
低背景权重 loss
反卷积前后噪声检测
加入噪声模型训练
输出强度融合
```

## 32. 风险四：AI 模型变成普通锐化器
问题：

```plain
ground truth 设计不合理
直接用 f 作为标签
loss 鼓励图像越锐越好
缺少 PSF 条件约束
```

解决：

```plain
使用 f*g_target+n 作为 ground truth
加入 PSF metadata
加入 forward consistency
避免早期使用 GAN loss
使用星点 FWHM 和 flux 指标约束
```

## 33. 风险五：模型生成虚假细节
问题：

```plain
perceptual loss 或 GAN loss 过强
训练数据分布不合理
模型缺少物理约束
```

解决：

```plain
以 reconstruction loss 为主
加入 forward consistency
保留输入输出可解释性
使用重卷积一致性检查
人工检查高风险样本
```

---

# 第六部分：推荐工程结构
建议项目按如下结构组织：

```plain
astro_deconv/
  io/
    read_fits.py
    write_fits.py

  preprocessing/
    background.py
    normalize.py

  stars/
    detect_stars.py
    psf_check.py
    fit_gaussian.py
    fit_moffat.py
    star_mask.py

  psf/
    global_psf.py
    local_psf.py
    psf_kernel.py

  classical/
    richardson_lucy.py
    wiener.py
    tile_deconv.py

  ai/
    dataset.py
    psf_degradation.py
    model_unet.py
    model_psf_conditioned.py
    train.py
    infer.py

  metrics/
    fwhm.py
    eccentricity.py
    ringing.py
    noise.py
    forward_consistency.py

  demos/
    demo_global_rl.py
    demo_tile_rl.py
    demo_stellar_nonstellar.py
    demo_ai_infer.py
```

---

## 34. 总结
本项目应采用“传统算法快速验证 + 自研 AI 长期升级”的双路线策略。

短期内，优先通过星点检测、PSF 拟合、PSF kernel 构建、Richardson-Lucy 反卷积和 soft mask 融合，完成一个可解释、可调试、可展示的星点反卷积修复 demo。该阶段的重点是建立基础模块和评价体系，而不是追求完全自动化。

中期进一步引入局部 PSF、tile 反卷积、stellar / nonstellar 分离处理，提高真实天文图像中的鲁棒性和视觉自然度。

长期则构建自研 AI 模型。AI 训练目标不应是简单的“模糊图到清晰图”，而应是更符合天文成像物理的：

[  
f_g_{bad}+n \rightarrow f_g_{target}+n  
]

在此基础上，引入 PSF 条件输入、星点/非星点双分支、mask 加权损失、flux consistency 和 forward consistency，使模型既具备自动化修复能力，又尽量避免过锐化、噪声放大和虚假细节生成。

最终，该技术路线能够形成一个从传统反卷积 demo 到 BXT-like 自研 AI 模型的完整研究与工程实现路径。

