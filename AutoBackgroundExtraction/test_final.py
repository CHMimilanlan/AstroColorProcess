"""
Deep Sky Image Background Extraction (ABE) - Final Version
============================================================
天文深空图像背景提取与 MTF 拉伸处理流程

主要功能：
    1. 结构检测（恒星/星云）用于采样点过滤
    2. ABE 自动背景提取（消除背景梯度）
    3. MTF 拉伸（增强图像细节）

Author: GC3
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys
import cv2
current_dir = os.path.dirname(os.path.abspath(__file__))
astro_color_process_dir = os.path.dirname(current_dir)       # .../AstroColorProcess
image_stack_root = os.path.dirname(astro_color_process_dir)  # .../ImageStack

if image_stack_root not in sys.path:
    sys.path.insert(0, image_stack_root)

from astropy.io import fits
from abe_algorithm_v2 import automatic_background_extraction
from structure_detection import detect_structures, StructureDetectionConfig

# ============================================================
# 添加 ImageStretch 模块路径
# ============================================================
try:

    from RASPAstroStacker.ImageStretch.stretch import ImageStretchAlgo, ImageDebayerAndStretch, SCNR_Average_Neutral
    HAS_IMAGE_STRETCH = True
except ImportError as e:
    print(f"⚠️  ImageStretch 模块加载失败: {e}")
    HAS_IMAGE_STRETCH = False



def visualize(img, filename, tmpdir="results"):
    if img.dtype == np.uint16:
        img = img >> 8
        img = img.astype(np.uint8)
        # compute_rgb_histogram(img, True)
        img = SCNR_Average_Neutral(img, mode="normal",show_hist=False)

    # compute_rgb_histogram(img, True)
    os.makedirs(tmpdir, exist_ok=True)
    dst_path = os.path.join(tmpdir, f"{filename}.jpg")
    cv2.imwrite(dst_path,img)

# ============================================================
# 用户可调参数配置区
# ============================================================

# -------------------- 1. 输入输出配置 --------------------

INPUT_IMAGE_PATH = r'C:\Workman02\python\OtherProject\pic\ABESamples\DWARF3-stacked-16_IC 1805_60s60_Duo-Band_20251126-013524041.fits'  # 输入 FITS 图像路径
# INPUT_IMAGE_PATH = r'C:\Workman02\python\OtherProject\pic\ABESamples\DWARF3-stacked-16_M 31_30s60_Astro_20251125-212604245.fits'  # 输入 FITS 图像路径
# INPUT_IMAGE_PATH = r'C:\Workman02\python\OtherProject\pic\ABESamples\DWARF3-stacked-16_M 45_30s60_Astro_20251124-203802912.fits'  # 输入 FITS 图像路径
# INPUT_IMAGE_PATH = r'C:\Workman02\python\OtherProject\pic\ABESamples\M20.FTS'  # 输入 FITS 图像路径
# INPUT_IMAGE_PATH = r'C:\Workman02\python\OtherProject\pic\ABESamples\SeestarS30-Stacked_161_IC 1805_60.0s_LP_20251126-053001.fit'  # 输入 FITS 图像路径
# INPUT_IMAGE_PATH = r'C:\Workman02\python\OtherProject\pic\ABESamples\SeestarS30-Stacked_451_M 45_30.0s_IRCUT_20251125-004001.fit'  # 输入 FITS 图像路径
# OUTPUT_DIR = 'results'                               # 输出目录

# -------------------- 2. ABE 算法核心参数 --------------------
"""
【ABE 自动背景提取算法参数说明】

grid_size: int
    网格大小，控制采样点密度。值越小采样点越密，但计算更慢。
    推荐: 32~128，常用 64

threshold_factor: float  
    阈值因子，用于筛选背景采样点。该值越大，允许的阈值越高，保留的采样点越多。
    值过大可能导致结构（恒星/星云）被误判为背景。
    推荐: 1.5~2.5，常用 2.0

degree: int
    多项式拟合次数。次数越高能拟合越复杂的背景梯度，但可能过拟合。
    推荐: 2~4，常用 3

correction_mode: str
    背景校正模式：
    - 'subtraction': 直接减去背景模型，背景压到 0（最彻底）
    - 'neutralize': 去掉背景梯度同时保留全局基线（可能略变亮）
    - 'division': 除法模式，结果为相对值
    推荐: 'neutralize' 或 'subtraction'

fitting_method: str
    背景拟合方法：
    - 'polynomial': 多项式拟合（快速，常用）
    - 'rbf': 径向基函数（更平滑）
    - 'spline': 样条插值（最平滑）

use_multiscale_sampling: bool
    是否启用多尺度采样。在粗/中/细网格分别采样后合并，能更好覆盖不同尺度
    的背景变化，但计算量更大。

multiscale_grid_sizes: tuple
    多尺度采样的各层网格数，如 (32, 64, 128)
"""
ABE_CONFIG = {
    'grid_size': 64,                    # 网格大小
    'threshold_factor': 2.0,           # 阈值因子
    'degree': 3,                       # 多项式次数
    'correction_mode': 'neutralize',   # 校正模式: subtraction / neutralize / division
    'fitting_method': 'polynomial',     # 拟合方法: polynomial / rbf / spline
    'use_multiscale_sampling': False, # 多尺度采样
    'multiscale_grid_sizes': (32, 64, 128),  # 多尺度网格
}

# -------------------- 3. 结构检测参数 --------------------
"""
【结构检测参数说明】

用于检测恒星和星云结构，在 ABE 采样时剔除这些区域，避免污染背景模型。

star_threshold_percentile: float
    恒星检测阈值百分位。越高越保守，只检测最亮的点源。
    推荐: 98.0~99.9

sobel_percentile: float
    边缘检测阈值百分位。用于检测星云边界。
    推荐: 85.0~95.0

structure_dilate_radius: int
    结构掩码膨胀半径。膨胀后结构区域更大，采样时剔除范围更广。
    推荐: 1~5

max_structure_ratio: float
    每个采样 patch 内允许的最大结构像素占比。
    超过该比例的 patch 会被剔除，不作为背景采样点。
    推荐: 0.05~0.2
"""
STRUCTURE_CONFIG = StructureDetectionConfig(
    star_threshold_percentile=98.0,    # 恒星检测阈值
    sobel_percentile=90.0,             # 边缘检测阈值
    structure_dilate_radius=3,         # 结构掩码膨胀半径
)
MAX_STRUCTURE_RATIO = 0.1              # patch 内最大结构占比

# -------------------- 4. 图像拉伸参数 --------------------
"""
【MTF 拉伸参数说明】

shadow_co: float
    阴影系数。负值提升暗部，绝对值越大提升越多。
    推荐: -1.0 ~ -3.0，常用 -2.0
"""
STRETCH_TYPE = 'MTF'                   # 拉伸类型: MTF / Gamma / Arcsinh
STRETCH_SHADOW_CO = -2.0               # 阴影系数


# ============================================================
# 核心函数
# ============================================================

def simple_stretch(image, stretch_type='MTF', shadow_co=-2.0):
    """
    简化版 MTF 拉伸函数
    
    使用 ImageStretch 库对图像进行非线性拉伸，增强暗部细节。
    
    参数:
        image: 输入图像 (float64, 0-1 范围)
        stretch_type: 拉伸类型 ('MTF', 'Gamma', 'Arcsinh')
        shadow_co: 阴影系数，负值提升暗部
        
    返回:
        拉伸后的 uint8 图像 (0-255)
    """
    # 1. 确保图像在合理范围内
    img = np.clip(image, 0, 1)
    
    # 2. 转换为 uint16（ImageStretch 内部处理需要）
    img_uint16 = (img * 65535).astype(np.uint16)
    
    # 3. 判断是彩色还是灰度图像
    rgb_flag = img_uint16.ndim == 3
    
    # 4. 彩色图像转换为 BGR 顺序（ImageStretch 期望 BGR）
    img_input = img_uint16
    
    # 5. 执行拉伸
    stretched = ImageStretchAlgo(
        resize_img=img_input,
        rgb_flag=rgb_flag,
        do_jpg=True,          # 输出 uint8 (0-255)
        shadow_co=shadow_co,
        s_type=stretch_type,
        do_debug=False
    )
    
    # 6. 如果转换过 BGR，再转回 RGB
    if rgb_flag and img_uint16.shape[2] == 3:
        stretched = stretched[:, :, [2, 1, 0]]  # BGR -> RGB
    
    return stretched


def normalize_for_save(img):
    """
    归一化图像到 [0,1] 范围
    
    用于保存时避免动态范围过窄导致图像看起来全黑。
    """
    img_f = img.astype(np.float32)
    vmin, vmax = img_f.min(), img_f.max()
    if vmax > vmin:
        return (img_f - vmin) / (vmax - vmin)
    return np.zeros_like(img_f, dtype=np.float32)


def ensure_uint8(img):
    """
    确保图像为 uint8 类型
    
    matplotlib.imsave 要求输入为 uint8 或 float。
    """
    if img.dtype != np.uint8:
        return img.astype(np.uint8)
    return img


def load_fits_image(path):
    """
    加载 FITS 图像文件
    
    参数:
        path: FITS 文件路径
        
    返回:
        float64 图像，范围归一化到 0-1
    """
    print(f"📂 读取图像: {os.path.basename(path)}")
    
    with fits.open(path) as hdul:
        image_data = hdul[0].data
    
    # 调整维度顺序（如果通道在第一维）
    if (image_data.ndim == 3 and 
        image_data.shape[0] < image_data.shape[1] and 
        image_data.shape[0] < image_data.shape[2]):
        image_data = np.transpose(image_data, (1, 2, 0))
        print("↔️  已调整维度顺序")
    
    # 转换为 float64 并归一化
    image_float = image_data.astype(np.float32)
    if np.max(image_float) > 0:
        image_float = image_float / np.max(image_float)
    
    print(f"📊 图像信息: {image_float.shape}, "
          f"范围: {image_float.min():.3f} - {image_float.max():.3f}")
    
    return image_float


def run_abe(image,structure_mask, abe_config, max_struct_ratio):
    """
    运行 ABE 自动背景提取算法
    
    参数:
        image: 输入图像 (float, 0-1)
        abe_config: ABE 配置字典
        max_struct_ratio: 最大结构占比
        
    返回:
        corrected: 校正后的图像
        background: 背景模型
    """
    print("\n🔄 正在执行 ABE 背景提取...")
    
    corrected, background, (samples_x, samples_y) = automatic_background_extraction(
        image=image,
        structure_mask=structure_mask,
        grid_size=abe_config['grid_size'],
        threshold_factor=abe_config['threshold_factor'],
        degree=abe_config['degree'],
        correction_mode=abe_config['correction_mode'],
        fitting_method=abe_config['fitting_method'],
        use_multiscale_sampling=abe_config['use_multiscale_sampling'],
        multiscale_grid_sizes=abe_config['multiscale_grid_sizes'],
        return_background=True,
        return_samples=True,
        max_structure_ratio=max_struct_ratio,
        profile_time=True,
    )
    
    # 裁剪负值到 0
    corrected = np.clip(corrected, 0, 1)
    
    print(f"✅ ABE 完成: 校正后范围 {corrected.min():.3f} - {corrected.max():.3f}")
    
    return corrected, background, samples_x, samples_y


def get_sample_points(image, structure_mask, abe_config, structure_cfg, max_struct_ratio):
    """
    获取背景采样点坐标
    
    用于可视化显示采样点分布。
    """
    # 转灰度用于采样点计算
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    
    _, (samples_x, samples_y) = automatic_background_extraction(
        image=gray,
        structure_mask=structure_mask,  # 采样点计算时不使用结构掩码，获取所有候选点
        grid_size=abe_config['grid_size'],
        threshold_factor=abe_config['threshold_factor'],
        degree=abe_config['degree'],
        correction_mode=abe_config['correction_mode'],
        fitting_method=abe_config['fitting_method'],
        use_multiscale_sampling=abe_config['use_multiscale_sampling'],
        multiscale_grid_sizes=abe_config['multiscale_grid_sizes'],
        return_background=False,
        return_samples=True,
        use_structure_filter=True,
        structure_config=structure_cfg,
        max_structure_ratio=max_struct_ratio,
    )
    
    return samples_x, samples_y


def apply_stretch(image, background, corrected, stretch_type, shadow_co):
    """
    对图像进行 MTF 拉伸
    """
    print("\n🎨 正在执行 MTF 拉伸...")
    
    original_stretched = simple_stretch(image, stretch_type, shadow_co)
    background_stretched = simple_stretch(background, stretch_type, shadow_co)
    corrected_stretched = simple_stretch(corrected, stretch_type, shadow_co)
    
    print("✅ 拉伸完成")
    
    return original_stretched, background_stretched, corrected_stretched


def _compute_visual_grid_centers(length, grid_size, margin):
    if grid_size <= 1:
        return np.array([length // 2], dtype=int)

    start = min(max(margin, 0), max(length - 1, 0))
    end = max(start, length - 1 - max(margin, 0))
    if end <= start:
        return np.linspace(0, length - 1, grid_size, dtype=int)
    return np.linspace(start, end, grid_size, dtype=int)


def _compute_visual_grid_edges(centers, length):
    centers = np.asarray(centers, dtype=np.float32)
    if centers.size == 0:
        return np.array([0.0, float(length)], dtype=np.float32)
    if centers.size == 1:
        return np.array([0.0, float(length)], dtype=np.float32)

    midpoints = (centers[:-1] + centers[1:]) / 2.0
    edges = np.concatenate(([0.0], midpoints, [float(length)]))
    return edges


def _draw_selected_sampling_grids(ax, image_shape, samples_x, samples_y, abe_config):
    h, w = image_shape[:2]
    patch_size = int(np.clip(round(min(h, w) / 64), 7, 31))
    if patch_size % 2 == 0:
        patch_size += 1
    half_patch = patch_size // 2

    samples_x = np.asarray(samples_x, dtype=np.float32)
    samples_y = np.asarray(samples_y, dtype=np.float32)
    if samples_x.size == 0 or samples_y.size == 0:
        return

    use_multiscale = abe_config.get('use_multiscale_sampling', False)
    if not use_multiscale:
        grid_size = int(abe_config['grid_size'])
        margin = max(2, half_patch)
        x_centers = _compute_visual_grid_centers(w, grid_size, margin)
        y_centers = _compute_visual_grid_centers(h, grid_size, margin)
        x_edges = _compute_visual_grid_edges(x_centers, w)
        y_edges = _compute_visual_grid_edges(y_centers, h)

        for x_edge in x_edges[1:-1]:
            ax.axvline(x_edge, color='deepskyblue', linewidth=0.4, alpha=0.18)
        for y_edge in y_edges[1:-1]:
            ax.axhline(y_edge, color='deepskyblue', linewidth=0.4, alpha=0.18)

        for x, y in zip(samples_x, samples_y):
            x_idx = int(np.argmin(np.abs(x_centers - x)))
            y_idx = int(np.argmin(np.abs(y_centers - y)))
            cell_x0 = x_edges[x_idx]
            cell_x1 = x_edges[x_idx + 1]
            cell_y0 = y_edges[y_idx]
            cell_y1 = y_edges[y_idx + 1]
            ax.add_patch(Rectangle(
                (cell_x0, cell_y0),
                cell_x1 - cell_x0,
                cell_y1 - cell_y0,
                fill=False,
                edgecolor='yellow',
                linewidth=0.55,
                alpha=0.75,
            ))

    for x, y in zip(samples_x, samples_y):
        x0 = max(0, x - half_patch)
        y0 = max(0, y - half_patch)
        x1 = min(w, x + half_patch + 1)
        y1 = min(h, y + half_patch + 1)
        ax.add_patch(Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor='lime',
            linewidth=0.7,
            alpha=0.55,
        ))

def visualize_results(image, background, corrected, 
                     original_stretched, background_stretched, corrected_stretched,
                     samples_x, samples_y, structure_result, base_name, debug_tmp,
                     abe_config):
    """
    可视化并保存结果
    """
    print("\n📈 正在生成可视化结果...")
    
    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ========== 第一行: 原始图像、背景模型、校正后图像 ==========
    # ========== 第二行: 拉伸后的图像（叠加采样点）==========
    titles_row2 = ['Original (MTF)', 'Background (MTF)', 'Corrected (MTF)']
    images_row2 = [original_stretched, background_stretched, corrected_stretched]
    
    for i, (ax, title, img) in enumerate(zip(axes, titles_row2, images_row2)):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        
        # 仅在第一张图叠加采样点
        if i == 0:
            _draw_selected_sampling_grids(ax, image.shape, samples_x, samples_y, abe_config)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Deep Sky Image ABE + MTF Stretch Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存整体对比图
    output_path = os.path.join(debug_tmp, f'{base_name}_abe_comparison.jpg')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 可视化已保存: {output_path}")
    plt.close()
    
    # ========== 保存单独的图像 ==========
    print("\n📁 正在保存单独图像...")
    
    # 辅助函数：保存单张图像
    def save_image(img, filename, cmap=None):
        img_save = ensure_uint8(img)
        plt.imsave(os.path.join(debug_tmp, filename), img_save, cmap=cmap)
        print(f"  ✓ {filename}")
    
    def save_image_with_samples(img, filename):
        img_h, img_w = img.shape[:2]
        dpi = 100
        fig = plt.figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        _draw_selected_sampling_grids(ax, image.shape, samples_x, samples_y, abe_config)
        ax.set_xlim(-0.5, img_w - 0.5)
        ax.set_ylim(img_h - 0.5, -0.5)
        ax.axis('off')
        fig.savefig(
            os.path.join(debug_tmp, filename),
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0,
            facecolor='black',
            edgecolor='none',
            # quality=95,
        )
        plt.close(fig)
        print(f"  saved {filename}")

    base_name_lower = base_name.lower()
    save_image(original_stretched, f'{base_name_lower}_original_mtf.jpg')
    save_image_with_samples(original_stretched, f'{base_name_lower}_original_mtf_samples.jpg')
    save_image(background_stretched, f'{base_name_lower}_background_mtf.jpg', cmap='hot')
    save_image(corrected_stretched, f'{base_name_lower}_corrected_mtf.jpg')


# ============================================================
# 主程序
# ============================================================

# def main():
#     """
#     主处理流程:
#         1. 加载 FITS 图像
#         2. 结构检测（恒星/星云）
#         3. ABE 背景提取
#         4. MTF 拉伸
#         5. 可视化与保存
#     """
#     print("=" * 50)
#     print("🚀 深空图像 ABE 背景提取处理")
#     print("=" * 50)
    
#     # 1. 创建输出目录
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # 2. 加载图像
#     from pathlib import Path
#     path = Path(INPUT_IMAGE_PATH)
#     debayer_img, stretch_img, header_configs, rgb_flag = ImageDebayerAndStretch(path, shadow_co=-2.0, color_calib=True, do_debug=True)
#     visualize(stretch_img, "stretch", tmpdir="results")
#     visualize(debayer_img, "debayer", tmpdir="results")

#     image_float = load_fits_image(INPUT_IMAGE_PATH)
#     base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    
#     # 3. 结构检测
#     print("\n🔍 正在进行结构检测...")
#     structure_result = detect_structures(image_float, STRUCTURE_CONFIG)
#     print("✅ 结构检测完成")
    
#     # 可选：保存结构掩码可视化
#     visualize_structure(image_float, structure_result)

#     star_mask = structure_result["star_mask"]
#     nebula_mask = structure_result["nebula_mask"]
#     structure_mask = structure_result["structure_mask"]

#     star_mask_img = star_mask.astype(np.uint8)
#     nebula_mask_img = nebula_mask.astype(np.uint8)
#     structure_mask_img = structure_mask.astype(np.uint8)

#     if stretch_img.ndim == 3:
#         star_mask_img = star_mask_img[:, :, np.newaxis]
#         nebula_mask_img = nebula_mask_img[:, :, np.newaxis]
#         structure_mask_img = structure_mask_img[:, :, np.newaxis]

#     visualize(stretch_img * star_mask_img, "star_mask_mult", tmpdir="results")
#     visualize(stretch_img * nebula_mask_img, "nebula_mask_mult", tmpdir="results")
#     visualize(stretch_img * structure_mask_img, "structure_mask_mult", tmpdir="results")

#     merged_mask_img = (
#         star_mask.astype(np.uint8)
#         + nebula_mask.astype(np.uint8)
#         + structure_mask.astype(np.uint8)
#     )
#     if stretch_img.ndim == 3:
#         merged_mask_img = merged_mask_img[:, :, np.newaxis]
#     visualize(stretch_img * merged_mask_img, "merged_structure_mask_mult", tmpdir="results")


#     # sys.exit(0)  # 先退出，后续代码暂不执行，待调试结构检测结果
#     # 4. ABE 背景提取
#     corrected, background = run_abe(
#         image_float, 
#         ABE_CONFIG, 
#         STRUCTURE_CONFIG, 
#         MAX_STRUCTURE_RATIO
#     )
    
#     # 5. 获取采样点（用于可视化）
#     samples_x, samples_y = get_sample_points(
#         image_float, 
#         ABE_CONFIG, 
#         STRUCTURE_CONFIG, 
#         MAX_STRUCTURE_RATIO
#     )
    
#     # 6. MTF 拉伸
#     original_stretched, background_stretched, corrected_stretched = apply_stretch(
#         image_float, background, corrected,
#         STRETCH_TYPE, STRETCH_SHADOW_CO
#     )
    
#     # 7. 可视化与保存
#     visualize_results(
#         image_float, background, corrected,
#         original_stretched, background_stretched, corrected_stretched,
#         samples_x, samples_y, structure_result, base_name
#     )
    
#     print("\n✨ 处理完成!")
#     print(f"📁 结果保存在: {os.path.abspath(OUTPUT_DIR)}")


# if __name__ == "__main__":
#     main()
