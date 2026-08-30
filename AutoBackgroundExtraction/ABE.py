from RASPAstroStacker.ImageStretch.stretch import ImageDebayerAndStretch
from RASPAstroStacker.StarExtraction.StarExtraction import StarExtraction


def ABE(img_path, threshold=0.5):
    debayer_img, stretch_img, header_configs, rgb_flag = ImageDebayerAndStretch(img_path, shadow_co=-2.0, color_calib=True, do_debug=False)
    







