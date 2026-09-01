import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from scipy.interpolate import Rbf, griddata
import time

try:
    import cv2
except ImportError:  # OpenCV is an optional acceleration dependency.
    cv2 = None

from structure_detection import detect_structures, StructureDetectionConfig


EPS = 1e-8


def _profile_now():
    return time.time()


def _profile_log(enabled, label, start_time):
    if enabled:
        print(f"[ABE profile] {label}: {time.time() - start_time:.4f}s")


# ============================================================
# 基础工具函数
# ============================================================

def _normalize_xy(x, y, w, h):
    """将像素坐标归一化到 [-1, 1]，提高多项式拟合的数值稳定性。"""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_n = 2.0 * x / max(w - 1, 1) - 1.0
    y_n = 2.0 * y / max(h - 1, 1) - 1.0
    return x_n, y_n


def _polynomial_terms(degree):
    """生成二维多项式项，顺序固定，便于拟合与求值一致。"""
    terms = []
    for total_degree in range(degree + 1):
        for x_degree in range(total_degree + 1):
            y_degree = total_degree - x_degree
            terms.append((x_degree, y_degree))
    return terms


def _patch_bounds(x, y, half_patch, w, h):
    y_min = max(0, int(y) - half_patch)
    y_max = min(h, int(y) + half_patch + 1)
    x_min = max(0, int(x) - half_patch)
    x_max = min(w, int(x) + half_patch + 1)
    return y_min, y_max, x_min, x_max


def _robust_mad(values):
    values = np.asarray(values, dtype=np.float32)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return med, 1.4826 * mad


def _ensure_channel_last(image):
    """返回 (H, W, C) 视图以及是否原始为 CHW。"""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError("image must be 3D for _ensure_channel_last")

    # 常见的 CHW 判据：第一维明显更小，且通常 <= 8
    if image.shape[0] <= 8 and image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        return np.transpose(image, (1, 2, 0)), True
    return image, False


# ============================================================
# 采样点评分与采样
# ============================================================

def _median_filter_2d(image, size):
    """Fast float32 median filter with SciPy-compatible reflect boundaries."""
    if cv2 is not None and size in (3, 5):
        radius = size // 2
        padded = np.pad(image, radius, mode="symmetric")
        try:
            return cv2.medianBlur(padded, size)[
                radius:-radius, radius:-radius
            ]
        except cv2.error:
            # Some older OpenCV builds have narrower dtype support.
            pass
    return median_filter(image, size=size)


def _build_sampling_metrics(image, median_filter_size=5):
    """构建背景采样判别用指标图。"""
    image = np.asarray(image, dtype=np.float32)

    # OpenCV's float32 median implementation is substantially faster than
    # scipy.ndimage on large images. Explicit symmetric padding reproduces
    # SciPy's default reflect boundary exactly; keep SciPy as a fallback.
    smooth = _median_filter_2d(image, median_filter_size)
    abs_residual = np.abs(image - smooth)

    # 局部残差的再平滑，可减少单个坏点/亮星边缘对 patch 判定的破坏
    local_residual = _median_filter_2d(abs_residual, 3)

    residual_med, residual_sigma = _robust_mad(local_residual)
    return {
        "smooth": smooth,
        "abs_residual": abs_residual,
        "local_residual": local_residual,
        "residual_med": residual_med,
        "residual_sigma": max(residual_sigma, EPS),
    }


def _sample_background_from_grid(
    image,
    structure_mask,
    x_coords,
    y_coords,
    local_residual,
    residual_threshold,
    patch_size,
    max_structure_ratio=0.1,
    residual_percentile=95.0,
):
    """
    从规则网格中提取背景采样点。

    判据采用 patch 残差的高分位数，而不是最大值，降低对离群点的敏感性。
    """
    h, w = image.shape
    half_patch = patch_size // 2
    use_structure = structure_mask is not None

    x_coords = np.asarray(x_coords, dtype=np.intp)
    y_coords = np.asarray(y_coords, dtype=np.intp)
    if x_coords.size == 0 or y_coords.size == 0:
        return []

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    y0 = flat_y - half_patch
    x0 = flat_x - half_patch
    y1 = y0 + patch_size
    x1 = x0 + patch_size

    # Normal ABE grids have a margin of half_patch, so all patches are the
    # same size.  Batched percentile/median operations remove thousands of
    # Python/NumPy dispatches while preserving the original statistics.
    full_patches = (
        np.all(y0 >= 0)
        and np.all(x0 >= 0)
        and np.all(y1 <= h)
        and np.all(x1 <= w)
    )
    if not full_patches:
        samples = []
        for y, x in zip(flat_y, flat_x):
            y_min, y_max, x_min, x_max = _patch_bounds(x, y, half_patch, w, h)
            patch = image[y_min:y_max, x_min:x_max]
            resid_patch = local_residual[y_min:y_max, x_min:x_max]
            if patch.size == 0:
                continue
            residual_score = float(np.percentile(resid_patch, residual_percentile))
            struct_ratio = (
                float(np.mean(structure_mask[y_min:y_max, x_min:x_max]))
                if use_structure
                else 0.0
            )
            if residual_score <= residual_threshold and struct_ratio <= max_structure_ratio:
                samples.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "z": float(np.median(patch)),
                        "residual_score": residual_score,
                        "struct_ratio": struct_ratio,
                    }
                )
        return samples

    residual_windows = np.lib.stride_tricks.sliding_window_view(
        local_residual, (patch_size, patch_size)
    )
    residual_patches = residual_windows[y0, x0]
    residual_scores = np.percentile(
        residual_patches, residual_percentile, axis=(-2, -1)
    )

    if use_structure:
        # Integral-image rectangle sums avoid materializing a second patch
        # tensor merely to calculate structure-mask means.
        integral = np.pad(
            np.asarray(structure_mask, dtype=np.int8)
            .cumsum(axis=0, dtype=np.int64)
            .cumsum(axis=1),
            ((1, 0), (1, 0)),
        )
        structure_counts = (
            integral[y1, x1]
            - integral[y0, x1]
            - integral[y1, x0]
            + integral[y0, x0]
        )
        structure_ratios = structure_counts / float(patch_size * patch_size)
    else:
        structure_ratios = np.zeros(flat_x.size, dtype=np.float64)

    selected = (residual_scores <= residual_threshold) & (
        structure_ratios <= max_structure_ratio
    )
    if not np.any(selected):
        return []

    selected_x = flat_x[selected]
    selected_y = flat_y[selected]
    image_windows = np.lib.stride_tricks.sliding_window_view(
        image, (patch_size, patch_size)
    )
    sample_values = np.median(
        image_windows[y0[selected], x0[selected]], axis=(-2, -1)
    )

    return [
        {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "residual_score": float(score),
            "struct_ratio": float(ratio),
        }
        for x, y, z, score, ratio in zip(
            selected_x,
            selected_y,
            sample_values,
            residual_scores[selected],
            structure_ratios[selected],
        )
    ]


def _merge_samples_by_distance(samples, min_distance):
    """
    合并过近的采样点。

    与旧版本不同：这里优先保留质量更高的点（残差更低、结构占比更低），
    而不是简单保留先出现的点。
    """
    if not samples:
        return []
    if min_distance is None or min_distance <= 0:
        return list(samples)

    sorted_samples = sorted(
        samples,
        key=lambda s: (s.get("residual_score", 0.0), s.get("struct_ratio", 0.0)),
    )

    kept = []
    min_distance_sq = float(min_distance) ** 2
    for sample in sorted_samples:
        keep = True
        for existing in kept:
            dx = sample["x"] - existing["x"]
            dy = sample["y"] - existing["y"]
            if dx * dx + dy * dy < min_distance_sq:
                keep = False
                break
        if keep:
            kept.append(sample)
    return kept


def _compute_grid_coordinates(length, grid_size, margin):
    """在边缘留出 margin，避免直接在最外层边界取样。"""
    if grid_size <= 1:
        return np.array([length // 2], dtype=int)

    start = min(max(margin, 0), max(length - 1, 0))
    end = max(start, length - 1 - max(margin, 0))
    if end <= start:
        return np.linspace(0, length - 1, grid_size, dtype=int)
    return np.linspace(start, end, grid_size, dtype=int)


def _sample_background_multiscale_internal(
    image,
    structure_mask,
    grid_sizes=(32, 64, 128),
    threshold_factor=1.5,
    max_structure_ratio=0.1,
    merge_min_distance_ratio=0.5,
    residual_percentile=95.0,
    patch_size=None,
):
    """多尺度采样主流程。"""
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    metrics = _build_sampling_metrics(image)

    if patch_size is None:
        # 比旧版本更大，更符合“背景块统计”而不是“像素点抽样”
        patch_size = int(np.clip(round(min(h, w) / 64), 7, 31))
        if patch_size % 2 == 0:
            patch_size += 1

    residual_med = metrics["residual_med"]
    residual_sigma = metrics["residual_sigma"]

    thresholds = [
        residual_med + threshold_factor * residual_sigma,
        residual_med + max(1.35 * threshold_factor, threshold_factor + 0.75) * residual_sigma,
        residual_med + max(1.8 * threshold_factor, threshold_factor + 1.5) * residual_sigma,
    ]

    all_samples = []
    edge_margin = max(2, patch_size // 2)
    for threshold in thresholds:
        all_samples.clear()
        for grid_size in grid_sizes:
            x_coords = _compute_grid_coordinates(w, grid_size, edge_margin)
            y_coords = _compute_grid_coordinates(h, grid_size, edge_margin)
            scale_samples = _sample_background_from_grid(
                image=image,
                structure_mask=structure_mask,
                x_coords=x_coords,
                y_coords=y_coords,
                local_residual=metrics["local_residual"],
                residual_threshold=threshold,
                patch_size=patch_size,
                max_structure_ratio=max_structure_ratio,
                residual_percentile=residual_percentile,
            )
            all_samples.extend(scale_samples)
        if all_samples:
            break

    min_dist = None
    if merge_min_distance_ratio is not None and merge_min_distance_ratio > 0:
        min_dist = (min(h, w) / max(grid_sizes)) * merge_min_distance_ratio

    merged_samples = _merge_samples_by_distance(all_samples, min_dist)
    return merged_samples, structure_mask, metrics

# ============================================================
# 拟合函数
# ============================================================

def fit_polynomial_2d(x_list, y_list, z_list, degree, image_shape=None, sigma_clip=3.0, max_iter=3):
    """
    鲁棒二维多项式拟合。

    改进点：
    1. 坐标归一化到 [-1,1]，提升数值稳定性。
    2. 迭代 sigma-clipping，减少脏采样点对背景面的影响。
    3. 返回 actual_degree 和 terms，避免求值阶段隐式猜测项顺序。
    """
    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    z = np.asarray(z_list, dtype=np.float32)

    if x.size == 0:
        return {
            "coeffs": np.array([0.0], dtype=np.float32),
            "terms": [(0, 0)],
            "degree": 0,
            "shape": image_shape,
        }

    if image_shape is None:
        h = int(np.max(y)) + 1
        w = int(np.max(x)) + 1
    else:
        h, w = image_shape

    degree = int(max(0, degree))
    while True:
        terms = _polynomial_terms(degree)
        if x.size >= len(terms) or degree == 0:
            break
        degree -= 1

    x_n, y_n = _normalize_xy(x, y, w, h)

    def build_design_matrix(xv, yv, terms_):
        A = np.empty((xv.size, len(terms_)), dtype=np.float32)
        for k, (ix, iy) in enumerate(terms_):
            A[:, k] = (xv ** ix) * (yv ** iy)
        return A

    A = build_design_matrix(x_n, y_n, terms)
    mask = np.ones(x.size, dtype=bool)
    coeffs = None

    for _ in range(max(1, max_iter)):
        coeffs, _, _, _ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
        fitted = A @ coeffs
        residual = z - fitted
        med, sigma = _robust_mad(residual[mask])
        sigma = max(sigma, EPS)
        new_mask = np.abs(residual - med) <= sigma_clip * sigma

        # 至少保留足够样本以支撑拟合
        if np.sum(new_mask) < len(terms):
            break
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    if coeffs is None:
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    return {
        "coeffs": coeffs,
        "terms": terms,
        "degree": degree,
        "shape": (h, w),
    }


def evaluate_polynomial_2d(X, Y, coeffs, degree=None, terms=None, image_shape=None):
    """根据给定项和系数评估二维多项式背景面。"""
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    coeffs = np.asarray(coeffs, dtype=np.float32)

    if terms is None:
        if degree is None:
            raise ValueError("degree or terms must be provided")
        terms = _polynomial_terms(int(degree))
        if len(terms) != len(coeffs):
            terms = terms[: len(coeffs)]

    if image_shape is None:
        h = int(np.max(Y)) + 1
        w = int(np.max(X)) + 1
    else:
        h, w = image_shape

    x_n, y_n = _normalize_xy(X, Y, w, h)
    Z = np.zeros_like(X, dtype=np.float32)
    for (ix, iy), c in zip(terms, coeffs):
        Z += c * (x_n ** ix) * (y_n ** iy)
    return Z


def _evaluate_polynomial_grid(image_shape, coeffs, terms):
    """Evaluate a polynomial on a regular image grid without two meshgrids."""
    h, w = image_shape
    coeffs = np.asarray(coeffs, dtype=np.float32)
    x_n, y_n = _normalize_xy(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        w,
        h,
    )
    max_x_degree = max(ix for ix, _ in terms)
    max_y_degree = max(iy for _, iy in terms)
    x_powers = [x_n ** exponent for exponent in range(max_x_degree + 1)]
    y_powers = [y_n ** exponent for exponent in range(max_y_degree + 1)]

    background = np.zeros((h, w), dtype=np.float32)
    for (ix, iy), coefficient in zip(terms, coeffs):
        background += (coefficient * x_powers[ix][None, :]) * y_powers[iy][:, None]
    return background


def _fit_background_surface(samples, image_shape, degree=3, fitting_method="polynomial"):
    """对采样点进行背景曲面拟合。"""
    h, w = image_shape

    sx = np.asarray([s["x"] for s in samples], dtype=np.float32)
    sy = np.asarray([s["y"] for s in samples], dtype=np.float32)
    sz = np.asarray([s["z"] for s in samples], dtype=np.float32)

    if sx.size == 0:
        return np.zeros((h, w), dtype=np.float32)

    method = str(fitting_method).lower()
    if method == "rbf":
        x_grid, y_grid = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        rbf = Rbf(sx, sy, sz, function="thin_plate", smooth=0.1)
        background = rbf(x_grid, y_grid)
    elif method == "spline":
        x_grid, y_grid = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        points = np.column_stack([sx, sy])
        background = griddata(points, sz, (x_grid, y_grid), method="cubic", fill_value=np.nan)
        if np.isnan(background).any():
            linear_bg = griddata(points, sz, (x_grid, y_grid), method="linear", fill_value=np.nan)
            nearest_bg = griddata(points, sz, (x_grid, y_grid), method="nearest")
            background = np.where(np.isnan(background), linear_bg, background)
            background = np.where(np.isnan(background), nearest_bg, background)
    else:
        poly = fit_polynomial_2d(sx, sy, sz, degree=degree, image_shape=image_shape)
        background = _evaluate_polynomial_grid(
            image_shape=image_shape,
            coeffs=poly["coeffs"],
            terms=poly["terms"],
        )

    return np.asarray(background, dtype=np.float32)


# ============================================================
# 背景校正
# ============================================================

def _apply_background_correction(image, background_model, samples_z, correction_mode="subtraction"):
    image = np.asarray(image, dtype=np.float32)
    background_model = np.asarray(background_model, dtype=np.float32)
    samples_z = np.asarray(samples_z, dtype=np.float32)

    if samples_z.size == 0:
        baseline = float(np.median(image))
    else:
        baseline = float(np.median(samples_z))

    mode = str(correction_mode).lower()
    if mode == "subtraction":
        corrected = image - background_model
        corrected = np.clip(corrected, 0, None)
    elif mode == "neutralize":
        # 保留一个全局背景基线：只去掉背景的空间变化，不额外引入 floor 偏移。
        corrected = image - (background_model - baseline)
        corrected = np.clip(corrected, 0, None)
    elif mode == "division":
        # 乘性校正时应乘回参考基线，避免结果尺度失真。
        scale = max(baseline, EPS)
        safe_bg = np.maximum(background_model, EPS)
        corrected = image * (scale / safe_bg)
        corrected = np.clip(corrected, 0, None)
    else:
        raise ValueError("correction_mode must be 'subtraction', 'division' or 'neutralize'")

    return corrected


def smooth_edge_background(bg_model, sigma=5):
    """保留兼容接口：对背景模型做轻微平滑。"""
    return gaussian_filter(np.asarray(bg_model, dtype=np.float32), sigma=sigma)


# ============================================================
# ABE 主流程
# ============================================================

def _prepare_sampling_luminance(image_2d_or_3d):
    if image_2d_or_3d.ndim == 2:
        return image_2d_or_3d.astype(np.float32)
    if image_2d_or_3d.ndim == 3:
        chw_or_hwc, was_chw = _ensure_channel_last(image_2d_or_3d)
        luminance = np.mean(chw_or_hwc, axis=2)
        return luminance.astype(np.float32)
    raise ValueError("image must be 2D or 3D")


def _sample_patch_medians(image_channel, samples_x, samples_y, patch_size):
    """Return patch medians in one batched operation for regular ABE samples."""
    h, w = image_channel.shape
    half_patch = patch_size // 2
    x = np.asarray(samples_x, dtype=np.intp)
    y = np.asarray(samples_y, dtype=np.intp)
    y0 = y - half_patch
    x0 = x - half_patch
    y1 = y0 + patch_size
    x1 = x0 + patch_size

    if (
        x.size > 0
        and np.all(y0 >= 0)
        and np.all(x0 >= 0)
        and np.all(y1 <= h)
        and np.all(x1 <= w)
    ):
        windows = np.lib.stride_tricks.sliding_window_view(
            image_channel, (patch_size, patch_size)
        )
        return np.median(windows[y0, x0], axis=(-2, -1)).astype(np.float32)

    return None


def _extract_background_for_single_channel(
    image_channel,
    samples_xy,
    degree=3,
    correction_mode="subtraction",
    fitting_method="polynomial",
    smooth_background_sigma=0.0,
):
    h, w = image_channel.shape
    samples_x, samples_y = samples_xy
    patch_size = int(np.clip(round(min(h, w) / 64), 7, 31))
    if patch_size % 2 == 0:
        patch_size += 1
    sample_values = _sample_patch_medians(
        image_channel, samples_x, samples_y, patch_size
    )
    if sample_values is not None:
        samples = [
            {"x": float(x), "y": float(y), "z": float(z)}
            for x, y, z in zip(samples_x, samples_y, sample_values)
        ]
    else:
        half_patch = patch_size // 2
        samples = []
        for x, y in zip(samples_x, samples_y):
            y_min, y_max, x_min, x_max = _patch_bounds(
                x, y, half_patch, w, h
            )
            patch = image_channel[y_min:y_max, x_min:x_max]
            if patch.size:
                samples.append(
                    {"x": float(x), "y": float(y), "z": float(np.median(patch))}
                )

    background = _fit_background_surface(samples, (h, w), degree=degree, fitting_method=fitting_method)
    if smooth_background_sigma and smooth_background_sigma > 0:
        background = gaussian_filter(background, sigma=float(smooth_background_sigma))
    corrected = _apply_background_correction(
        image_channel,
        background,
        samples_z=[s["z"] for s in samples],
        correction_mode=correction_mode,
    )
    return corrected, background


def automatic_background_extraction(
    image,
    structure_mask,
    grid_size=64,
    threshold_factor=1.5,
    degree=3,
    correction_mode="subtraction",
    return_background=False,
    return_samples=False,
    max_structure_ratio=0.1,
    fitting_method: str = "polynomial",
    use_multiscale_sampling=False,
    multiscale_grid_sizes=(32, 64, 128),
    multiscale_merge_min_distance_ratio=0.5,
    profile_time=False,
):
    """
    自动背景提取（ABE）优化版。

    设计原则：
    1. 采样：在规则网格上自动寻找低结构、低残差区域。
    2. 拟合：拟合低频背景模型，而不是追踪高频星点/星云结构。
    3. 校正：根据 subtraction / neutralize / division 模式完成背景消除。

    相比旧版本的改进：
    - 不再用递归处理彩色图，主流程更清晰。
    - 采样点在所有通道共享同一组 (x, y)，避免 RGB 独立采样导致偏色。
    - 使用更稳健的 patch 残差高分位数判别背景点。
    - 多项式拟合加入坐标归一化与 sigma-clipping。
    - neutralize / division 公式重写，更符合 ABE 的背景基线思想。
    """
    total_t0 = _profile_now()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale image or a 3D color image")

    # ---------- 1) 在亮度图上统一确定采样点坐标 ----------
    t0 = _profile_now()
    sampling_image = _prepare_sampling_luminance(image)
    h, w = sampling_image.shape
    _profile_log(profile_time, "prepare_sampling_luminance", t0)

    if use_multiscale_sampling:
        t0 = _profile_now()
        samples, _, _ = _sample_background_multiscale_internal(
            image=sampling_image,
            structure_mask=structure_mask,
            grid_sizes=multiscale_grid_sizes,
            threshold_factor=threshold_factor,
            max_structure_ratio=max_structure_ratio,
            merge_min_distance_ratio=multiscale_merge_min_distance_ratio,
        )
        _profile_log(profile_time, "sample_background_multiscale", t0)
    else:
        t0 = _profile_now()
        samples, _, _ = _sample_background_multiscale_internal(
            image=sampling_image,
            structure_mask=structure_mask,
            grid_sizes=(grid_size,),
            threshold_factor=threshold_factor,
            max_structure_ratio=max_structure_ratio,
            merge_min_distance_ratio=0.0,
        )
        _profile_log(profile_time, "sample_background_single_scale", t0)

    # fallback：若样本过少，则自动启用两层网格补点
    min_required = max(16, (degree + 1) * (degree + 2) // 2 * 2)
    if len(samples) < min_required:
        fallback_grids = tuple(sorted(set([max(8, grid_size // 2), grid_size, grid_size * 2])))
        t0 = _profile_now()
        samples, _, _ = _sample_background_multiscale_internal(
            image=sampling_image,
            structure_mask=structure_mask,
            grid_sizes=fallback_grids,
            threshold_factor=max(threshold_factor, 1.5),
            max_structure_ratio=max_structure_ratio,
            merge_min_distance_ratio=max(multiscale_merge_min_distance_ratio, 0.35),
        )
        _profile_log(profile_time, "sample_background_fallback", t0)

    if not samples:
        # 没有任何采样点时，退化为零背景，避免崩溃
        if image.ndim == 2:
            background = np.zeros_like(image, dtype=np.float32)
            corrected = image.copy()
            outputs = [corrected]
            if return_background:
                outputs.append(background)
            if return_samples:
                outputs.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
            return tuple(outputs) if len(outputs) > 1 else outputs[0]
        else:
            img_hwc, was_chw = _ensure_channel_last(image)
            background = np.zeros_like(img_hwc, dtype=np.float32)
            corrected = img_hwc.copy()
            if was_chw:
                background = np.transpose(background, (2, 0, 1))
                corrected = np.transpose(corrected, (2, 0, 1))
            outputs = [corrected]
            if return_background:
                outputs.append(background)
            if return_samples:
                outputs.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
            return tuple(outputs) if len(outputs) > 1 else outputs[0]

    t0 = _profile_now()
    samples_x = np.asarray([s["x"] for s in samples], dtype=np.float32)
    samples_y = np.asarray([s["y"] for s in samples], dtype=np.float32)
    _profile_log(profile_time, "materialize_sample_coordinates", t0)
    if profile_time:
        print(f"[ABE profile] selected_samples: {len(samples)}")

    # ---------- 2) 灰度图 ----------
    if image.ndim == 2:
        t0 = _profile_now()
        corrected, background = _extract_background_for_single_channel(
            image_channel=image,
            samples_xy=(samples_x, samples_y),
            degree=degree,
            correction_mode=correction_mode,
            fitting_method=fitting_method,
        )
        _profile_log(profile_time, "extract_background_single_channel", t0)
        _profile_log(profile_time, "automatic_background_extraction_total", total_t0)
        outputs = [corrected]
        if return_background:
            outputs.append(background)
        if return_samples:
            outputs.append((samples_x, samples_y))
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    # ---------- 3) 彩色图：共享采样点，逐通道拟合 ----------
    t0 = _profile_now()
    image_hwc, was_chw = _ensure_channel_last(image)
    _profile_log(profile_time, "ensure_channel_last", t0)
    corrected_channels = []
    background_channels = []

    for channel_idx in range(image_hwc.shape[2]):
        channel_t0 = _profile_now()
        corrected_c, background_c = _extract_background_for_single_channel(
            image_channel=image_hwc[:, :, channel_idx],
            samples_xy=(samples_x, samples_y),
            degree=degree,
            correction_mode=correction_mode,
            fitting_method=fitting_method,
        )
        corrected_channels.append(corrected_c)
        background_channels.append(background_c)
        _profile_log(profile_time, f"extract_background_channel_{channel_idx}", channel_t0)

    t0 = _profile_now()
    corrected = np.stack(corrected_channels, axis=2)
    background = np.stack(background_channels, axis=2)
    _profile_log(profile_time, "stack_channels", t0)

    if was_chw:
        t0 = _profile_now()
        corrected = np.transpose(corrected, (2, 0, 1))
        background = np.transpose(background, (2, 0, 1))
        _profile_log(profile_time, "transpose_back_to_chw", t0)

    _profile_log(profile_time, "automatic_background_extraction_total", total_t0)
    outputs = [corrected]
    if return_background:
        outputs.append(background)
    if return_samples:
        outputs.append((samples_x, samples_y))
    return tuple(outputs) if len(outputs) > 1 else outputs[0]
