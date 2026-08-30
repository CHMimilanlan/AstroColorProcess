import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)          # .../RASPAstroStacker
parent_of_project = os.path.dirname(project_root)    # .../ImageStack
parent_of_project = os.path.dirname(parent_of_project)    # .../ImageStack

sys.path.insert(0, parent_of_project)
from os.path import join as pjoin

from RASPAstroStacker.ImageStretch.stretch import ImageDebayerAndStretch, SCNR_Average_Neutral
from RASPAstroStacker.StarExtraction.extract import BackgroundExtractionAPI, PatchifyBackgroundExtractionAPI
# from RASPAstroStacker.ColorProcess.SaturEnhance import (
#     compute_and_plot_histogram,
#     apply_brightness_threshold_mask_and_save,
# )
from pathlib import Path
import argparse
import cv2
import numpy as np
import time
from test_final import run_abe, apply_stretch, visualize_results
from structure_detection import detect_structures, StructureDetectionConfig


def LoadExtractParam(mode):
    from RASPAstroStacker.base import _load_extract_param, LoadYAML, _apply_extract_mode_override
    config = LoadYAML()
    stack_hyper_param = config["hyper_parameters"]
    extract_param = _load_extract_param(stack_hyper_param)
    extract_param = _apply_extract_mode_override(config, extract_param, mode)
    if mode in ["stacker", "solver"]:
        extract_param = _apply_extract_mode_override(config, extract_param, mode)
    else:
        assert False, "mode should be stacker or solver"
    return extract_param


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


def clear_debug_tmp(debug_tmp_path):
    if not os.path.exists(debug_tmp_path):
        return

    for entry_name in os.listdir(debug_tmp_path):
        entry_path = os.path.join(debug_tmp_path, entry_name)
        if os.path.isdir(entry_path):
            for root, dirs, files in os.walk(entry_path, topdown=False):
                for file_name in files:
                    os.remove(os.path.join(root, file_name))
                for dir_name in dirs:
                    os.rmdir(os.path.join(root, dir_name))
            os.rmdir(entry_path)
        else:
            os.remove(entry_path)


def patchify_standardize_and_normalize(stretch_image, patch_num=6):
    image = np.asarray(stretch_image, dtype=np.float32)
    if image.ndim not in (2, 3):
        raise ValueError("stretch_image must be a 2D or 3D array")

    patch_num = int(max(1, patch_num))
    row_splits = np.array_split(np.arange(image.shape[0]), patch_num)
    col_splits = np.array_split(np.arange(image.shape[1]), patch_num)

    stitched_rows = []
    for row_indices in row_splits:
        if row_indices.size == 0:
            continue

        row_start = int(row_indices[0])
        row_end = int(row_indices[-1]) + 1
        row_patches = []

        for col_indices in col_splits:
            if col_indices.size == 0:
                continue

            col_start = int(col_indices[0])
            col_end = int(col_indices[-1]) + 1
            patch = image[row_start:row_end, col_start:col_end].astype(np.float32)

            patch_mean = float(np.mean(patch))
            patch_std = float(np.std(patch))
            if patch_std > 1e-8:
                patch = (patch - patch_mean) / patch_std
            else:
                patch = np.zeros_like(patch, dtype=np.float32)

            row_patches.append(patch)

        if row_patches:
            stitched_rows.append(np.hstack(row_patches))

    if not stitched_rows:
        return np.zeros_like(image, dtype=np.uint8)

    stitched_image = np.vstack(stitched_rows).astype(np.float32)
    stitched_min = float(np.min(stitched_image))
    stitched_max = float(np.max(stitched_image))

    if stitched_max > stitched_min:
        stitched_image = (stitched_image - stitched_min) / (stitched_max - stitched_min)
    else:
        stitched_image = np.zeros_like(stitched_image, dtype=np.float32)

    stitched_image = (stitched_image * 255.0).clip(0, 255).astype(np.uint8)
    return stitched_image


def stitch_patch_multiscale_thresholds(patch_multiscale_results_thres):
    if not patch_multiscale_results_thres or not patch_multiscale_results_thres[0]:
        return []

    first_patch_thres = patch_multiscale_results_thres[0][0]
    if not first_patch_thres:
        return []

    scale_count = len(first_patch_thres)
    stitched_results = []

    for scale_idx in range(scale_count):
        stitched_rows = []
        for row_patches in patch_multiscale_results_thres:
            row_images = []
            for patch_thres in row_patches:
                row_images.append(patch_thres[scale_idx])
            stitched_rows.append(np.hstack(row_images))
        stitched_results.append(np.vstack(stitched_rows))

    return stitched_results

ABE_CONFIG = {
    'grid_size': 64,                    # 网格大小
    'threshold_factor': 2.0,           # 阈值因子
    'degree': 3,                       # 多项式次数
    'correction_mode': 'neutralize',   # 校正模式: subtraction / neutralize / division
    'fitting_method': 'polynomial',     # 拟合方法: polynomial / rbf / spline
    'use_multiscale_sampling': False, # 多尺度采样
    'multiscale_grid_sizes': (32, 64, 128),  # 多尺度网格
}
# STRUCTURE_CONFIG = StructureDetectionConfig(
#     star_threshold_percentile=98.0,    # 恒星检测阈值
#     sobel_percentile=90.0,             # 边缘检测阈值
#     structure_dilate_radius=3,         # 结构掩码膨胀半径
# )
MAX_STRUCTURE_RATIO = 0.1              # patch 内最大结构占比
STRETCH_TYPE = 'MTF'                   # 拉伸类型: MTF / Gamma / Arcsinh
STRETCH_SHADOW_CO = -2.0               # 阴影系数
# OUTPUT_DIR = 'results'                               # 输出目录



def GetPatchifyStructureMask(img, extract_params, filepath, debug_tmp):
    patch_multiscale_results, patch_multiscale_results_thres = PatchifyBackgroundExtractionAPI(img, 
                                                                           filepath.stem,
                                                                           extract_params, 
                                                                           do_debug=True,
                                                                           debug_tmp_path=debug_tmp)
    multiscale_results_thres = stitch_patch_multiscale_thresholds(patch_multiscale_results_thres)
    for i, stitched_thres_img in enumerate(multiscale_results_thres):
        visualize(stitched_thres_img.astype(np.uint8), f"patch_stitched_multiscale_thres_{i}", tmpdir=debug_tmp)

    thres_bool_list = [thres_img.astype(bool) for thres_img in multiscale_results_thres]
    merged_thres = np.sum(thres_bool_list, axis=0).astype(np.int8)
    merged_thres_for_mult = (merged_thres == 0).astype(np.int8)
    structure_mask = (merged_thres_for_mult == 0).astype(np.int8)
    if img.ndim == 3 and merged_thres_for_mult.ndim == 2:
        merged_thres_for_mult = merged_thres_for_mult[:, :, np.newaxis]
    merged_mult_img = img * merged_thres_for_mult
    visualize(merged_mult_img, "multiscale_thres_merged_sum", tmpdir=debug_tmp)
    return structure_mask, multiscale_results_thres


def GetNormStructureMask(img, extract_params, filepath, do_debug, debug_tmp):
    multiscale_results, multiscale_results_thres = BackgroundExtractionAPI(img, 
                                                                           filepath.stem,
                                                                           extract_params, 
                                                                           do_debug=do_debug,
                                                                           debug_tmp_path=debug_tmp)
    # multiscale_results_thres = multiscale_results_thres[:-1]
    thres_bool_list = [thres_img.astype(bool) for thres_img in multiscale_results_thres]
    merged_thres = np.sum(thres_bool_list, axis=0).astype(np.int8)
    merged_thres_for_mult = (merged_thres == 0).astype(np.int8)
    structure_mask = (merged_thres_for_mult == 0).astype(np.int8)
    return structure_mask, merged_thres_for_mult, multiscale_results_thres


def ABEProcess(mode, filepath, do_debug, debug_tmp):
    extract_params = LoadExtractParam(mode)
    filepath = Path(filepath)
    t0 = time.time()
    debayer_img, stretch_img, header_configs, rgb_flag = ImageDebayerAndStretch(filepath, shadow_co=-2.0, color_calib=True, do_debug=True)
    print(f"ImageDebayerAndStretch time: {time.time() - t0:.3f}s")
    if do_debug:
        # origin image
        visualize(stretch_img, "stretch", tmpdir=pjoin(debug_tmp, "image"))
        visualize(debayer_img, "debayer", tmpdir=pjoin(debug_tmp, "image"))

    # 优先使用normalize后的图像进行结构检测，因为其对不同区域的适应性更好，能更准确地提取结构特征。
    # structure_mask, multiscale_results_thres = GetPatchifyStructureMask(stretch_img, extract_params, filepath, debug_tmp)
    t0 = time.time()
    normed_stretch_img = patchify_standardize_and_normalize(stretch_img, patch_num=6)
    print(f"patchify_standardize_and_normalize time: {time.time() - t0:.3f}s")
    t0 = time.time()
    structure_mask, merged_thres_for_mult, multiscale_results_thres = GetNormStructureMask(normed_stretch_img, extract_params, 
                                                                    filepath,False, pjoin(debug_tmp, "multiscale"))
    print(f"GetNormStructureMask time: {time.time() - t0:.3f}s")

    if do_debug:
        if stretch_img.ndim == 3 and merged_thres_for_mult.ndim == 2:
            merged_thres_for_mult = merged_thres_for_mult[:, :, np.newaxis]
        merged_mult_img = stretch_img * merged_thres_for_mult
        visualize(merged_mult_img, "thres_merged_image", tmpdir=debug_tmp)
        visualize(normed_stretch_img, "norm_stretch", tmpdir=pjoin(debug_tmp, "image"))
        visualize(structure_mask*255, "structure_mask", tmpdir=pjoin(debug_tmp, "image"))
        for i, thres_img in enumerate(multiscale_results_thres):
            tmpimg = ((255 - thres_img) / 255).astype(np.uint8)
            if stretch_img.ndim == 3 and tmpimg.ndim == 2:
                tmpimg = tmpimg[:, :, np.newaxis]
            mult_img = stretch_img * tmpimg
            visualize(mult_img, f"merged_thres_{i}", tmpdir=pjoin(debug_tmp, "multiscale"))
        print("done!")

    image_float = debayer_img.astype(np.float32)
    if np.max(image_float) > 0:
        image_float = image_float / np.max(image_float)
    
    print(f"📊 图像信息: {image_float.shape}, "
          f"范围: {image_float.min():.3f} - {image_float.max():.3f}")
    
    t0 = time.time()
    corrected, background, samples_x, samples_y = run_abe(
        image_float, 
        structure_mask,
        ABE_CONFIG, 
        # STRUCTURE_CONFIG, 
        MAX_STRUCTURE_RATIO
    )
    print(f"run_abe time: {time.time() - t0:.3f}s")
    
    # 6. MTF 拉伸
    t0 = time.time()
    original_stretched, background_stretched, corrected_stretched = apply_stretch(
        image_float, background, corrected,
        STRETCH_TYPE, STRETCH_SHADOW_CO
    )
    print(f"apply_stretch time: {time.time() - t0:.3f}s")
    
    # 7. 可视化与保存
    t0 = time.time()
    visualize_results(
        image_float, background, corrected,
        original_stretched, background_stretched, corrected_stretched,
        samples_x, samples_y, structure_mask, base_name="M", debug_tmp=debug_tmp,
        abe_config=ABE_CONFIG
    )
    print(f"visualize_results time: {time.time() - t0:.3f}s")
    
    print("\n✨ 处理完成!")
    print(f"📁 结果保存在: {os.path.abspath(debug_tmp)}")





def main(filepath, debug_tmp_base):
    mode = "solver"
    debug_tmp = os.path.join(debug_tmp_base, Path(filepath).stem)
    os.makedirs(debug_tmp, exist_ok=True)
    clear_debug_tmp(debug_tmp)
    ABEProcess(mode, filepath, do_debug=True, debug_tmp=debug_tmp)


    
if __name__ == "__main__":
    # dir
    inputpath = r"C:\Workman02\python\OtherProject\pic\ABESamples"

    # samples
    # inputpath = r"C:\Workman02\python\OtherProject\pic\ABESamples\M31111.fit"
    # inputpath = r"C:\Workman02\python\OtherProject\pic\ABESamples\SeestarS30-Stacked_451_M 45_30.0s_IRCUT_20251125-004001.fit"
    # filepath = r"C:\Workman02\python\OtherProject\ImageStack\RASPAstroStacker\ColorProcess\AutoBackgroundExtraction\M8.FTS"
    # filepath = r'C:\Workman02\python\OtherProject\pic\ABESamples\DWARF3-stacked-16_IC 1805_60s60_Duo-Band_20251126-013524041.fits'  # 输入 FITS 图像路径
    # filepath = r'C:\Workman02\python\OtherProject\pic\ABESamples\DWARF3-stacked-16_M 31_30s60_Astro_20251125-212604245.fits'  # 输入 FITS 图像路径
    # filepath = r'C:\Workman02\python\OtherProject\pic\ABESamples\DWARF3-stacked-16_M 45_30s60_Astro_20251124-203802912.fits'  # 输入 FITS 图像路径
    # filepath = r'C:\Workman02\python\OtherProject\pic\ABESamples\M20.FTS'  # 输入 FITS 图像路径
    # filepath = r'C:\Workman02\python\OtherProject\pic\ABESamples\SeestarS30-Stacked_161_IC 1805_60.0s_LP_20251126-053001.fit'  # 输入 FITS 图像路径
    # filepath = r'C:\Workman02\python\OtherProject\pic\ABESamples\SeestarS30-Stacked_451_M 45_30.0s_IRCUT_20251125-004001.fit'  # 输入 FITS 图像路径

    debug_tmp_base = r"C:\Workman02\python\OtherProject\ImageStack\RASPAstroStacker\ColorProcess\AutoBackgroundExtraction\debug_tmp"
    clear_debug_tmp(debug_tmp_base)
    if os.path.isfile(inputpath):
        main(inputpath, debug_tmp_base)
    elif os.path.isdir(inputpath):
        for filename in os.listdir(inputpath):
            filepath = os.path.join(inputpath, filename)
            main(filepath, debug_tmp_base)




