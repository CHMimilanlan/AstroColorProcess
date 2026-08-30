import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from scipy import ndimage
from skimage import filters, morphology, measure


@dataclass
class StructureDetectionConfig:
    """
    星点/星云结构检测配置。

    目标：输出结构掩码（True=结构区域，需要避开），用于采样点剔除或可视化。
    """

    # 预处理
    clip_low_pct: float = 5.0
    clip_high_pct: float = 95.0
    gamma: float = 0.7  # <1 提升暗部，利于弥散星云

    # 恒星检测（点源）
    star_threshold_percentile: float = 99.0  # 越高越保守
    star_open_radius: int = 2
    star_min_area: int = 5
    star_max_area: int = 300

    # 星云/扩展结构检测
    nebula_thresholds: tuple = (0.7, 0.5, 0.3)  # 相对 max(gray_norm) 的分层阈值
    nebula_close_radius: int = 3
    nebula_min_area: int = 200

    # 边缘补充（适合有明显边界结构）
    sobel_percentile: float = 90.0
    edge_close_radius: int = 5
    edge_min_area: int = 500

    # 后处理
    nebula_dilate_radius: int = 1
    structure_dilate_radius: int = 2


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return np.mean(image, axis=2).astype(np.float32)
    return image.astype(np.float32)


def _preprocess(gray: np.ndarray, cfg: StructureDetectionConfig) -> np.ndarray:
    lo = np.percentile(gray, cfg.clip_low_pct)
    hi = np.percentile(gray, cfg.clip_high_pct)
    if hi > lo:
        scaled = np.clip((gray - lo) / (hi - lo + 1e-10), 0, 1)
    else:
        scaled = np.zeros_like(gray, dtype=np.float32)
    return np.power(scaled, cfg.gamma)


def detect_structures(image: np.ndarray, config: Optional[StructureDetectionConfig] = None) -> Dict[str, np.ndarray]:
    """
    检测星点与星云结构。

    Returns:
        dict:
          - 'star_mask': bool(H,W)
          - 'nebula_mask': bool(H,W)
          - 'structure_mask': bool(H,W) = star|nebula（膨胀后）
          - 'gray_norm': float(H,W) 预处理后灰度（便于调参/可视化）
    """
    cfg = config or StructureDetectionConfig()
    gray = _to_gray(image)
    gray_norm = _preprocess(gray, cfg)

    # 1) 恒星：亮点 + 连通域面积过滤
    thr_star = np.percentile(gray_norm, cfg.star_threshold_percentile)
    bright = gray_norm > thr_star
    if cfg.star_open_radius > 0:
        bright = morphology.opening(bright, morphology.disk(cfg.star_open_radius))

    labels = measure.label(bright)
    star_mask = np.zeros_like(bright, dtype=bool)
    if labels.max() > 0:
        for region in measure.regionprops(labels):
            if cfg.star_min_area < region.area < cfg.star_max_area:
                star_mask[tuple(region.coords.T)] = True

    # 2) 星云：多阈值分层 + 形态学连接 + 面积过滤
    nebula_mask = np.zeros_like(bright, dtype=bool)
    maxv = float(np.max(gray_norm)) if np.max(gray_norm) > 0 else 1.0
    for t in cfg.nebula_thresholds:
        layer = gray_norm > (t * maxv)
        if not np.any(layer):
            continue
        if cfg.nebula_close_radius > 0:
            layer = morphology.closing(layer, morphology.disk(cfg.nebula_close_radius))

        lab = measure.label(layer)
        if lab.max() == 0:
            continue
        for region in measure.regionprops(lab):
            if region.area > cfg.nebula_min_area:
                nebula_mask[tuple(region.coords.T)] = True

    # 3) 边缘补充：Sobel 强边缘 -> 填洞 -> 面积过滤
    sob = filters.sobel(gray_norm)
    thr_edge = np.percentile(sob, cfg.sobel_percentile)
    edge = sob > thr_edge
    if np.any(edge):
        edge_filled = ndimage.binary_fill_holes(edge)
        if cfg.edge_close_radius > 0:
            edge_filled = morphology.closing(edge_filled, morphology.disk(cfg.edge_close_radius))

        lab = measure.label(edge_filled)
        if lab.max() > 0:
            for region in measure.regionprops(lab):
                if region.area > cfg.edge_min_area:
                    nebula_mask[tuple(region.coords.T)] = True

    if cfg.nebula_dilate_radius > 0 and np.any(nebula_mask):
        nebula_mask = morphology.dilation(nebula_mask, morphology.disk(cfg.nebula_dilate_radius))

    structure_mask = star_mask | nebula_mask
    if cfg.structure_dilate_radius > 0 and np.any(structure_mask):
        structure_mask = morphology.dilation(structure_mask, morphology.disk(cfg.structure_dilate_radius))

    return {
        "star_mask": star_mask,
        "nebula_mask": nebula_mask,
        "structure_mask": structure_mask,
        "gray_norm": gray_norm,
    }

