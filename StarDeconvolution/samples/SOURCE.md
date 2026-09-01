# 测试样例来源

## 主测试样例：M45 DSS2 星场

- 文件：`m45_dss2_cds.jpg`
- 天体：M45（Pleiades / 昴星团）
- 数据服务：CDS HiPS2FITS，`CDS/P/DSS2/color`
- 下载地址：https://alasky.cds.unistra.fr/hips-image-services/hips2fits?hips=CDS%2FP%2FDSS2%2Fcolor&width=1000&height=1000&fov=1&projection=TAN&coordsys=icrs&object=M45&format=jpg
- 下载日期：2026-08-30
- 本地尺寸：1000 × 1000，RGB，8-bit JPEG
- SHA-256：`E7D877F5AD5E0A94358741778B47F74516E8FC371B7BD51A5FCE1F8A4E75C6EB`
- 数据署名：Digitized Sky Survey - STScI/NASA；彩色合成与 HiPS：CDS
- 数据集记录：https://alasky.cds.unistra.fr/MocServer/query?ID=CDS%2FP%2FDSS2%2Fcolor&fmt=html&get=record

这张星场图用于默认 benchmark，原因是它比星系特写包含更多可用于经验 PSF 估计的孤立星点。

## 补充视觉样例：M51 Hubble

- 文件：`m51_hubble_nasa.tif`
- 天体：M51（Whirlpool Galaxy / 涡状星系）
- 来源页面：https://science.nasa.gov/asset/hubble/hubble-acs-visible-image-of-m51/
- 下载地址：https://assets.science.nasa.gov/dynamicimage/assets/science/missions/hubble/releases/2011/01/STScI-01EVT4ZDTA6V9ASDFRHWCSK775.tif?w=1000
- 下载日期：2026-08-30
- 本地尺寸：1000 × 1347，RGB，8-bit TIFF
- SHA-256：`0191538473D662A7BBF92AE5549AB9AB7B0D33E3F5A842072D9C70DACDCDF291`
- 页面署名：NASA, ESA, S. Beckwith (STScI), and the Hubble Heritage Team (STScI/AURA)

两个下载文件都是已经拉伸、配色的公开展示图，只用于视觉 demo。默认 `demo.py` 将 M45
作为参考图，施加已知椭圆 PSF 和固定随机种子噪声，生成可重复的退化输入；M51 用于补充
检查星系等扩展结构。科学数据处理应使用线性、32-bit 浮点 FITS。
