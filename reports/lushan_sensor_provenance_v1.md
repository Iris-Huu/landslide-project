# GDCLD Lushan影像来源核查

## 原始论文证据

Fang et al. (2024), Earth System Science Data:
https://essd.copernicus.org/articles/16/4817/2024/

论文第3.1.8节报告芦山区域具有：
- Map World：0.5 m影像
- PlanetScope：3 m影像
- UAV：0.2 m影像，Sony ILCE-5100，2022-06-13获取

论文讨论部分说明Map World可能整合不同卫星来源，
因此Map World不能直接等同于单一卫星传感器。

## 本地GeoTIFF证据

| 文件 | EPSG | 像素间距 | 来源推断 |
|---|---:|---:|---|
| Lushan_1.tif | 4544 | 0.5 m | 可能为Map World |
| Lushan_2.tif | 32648 | 0.5 m | 可能为Map World |
| Lushan_3.tif | 32648 | 3.0 m | 可能为PlanetScope |

三幅影像均有投影和仿射变换，
但没有能够直接识别具体传感器型号的元数据。

## 科研判定

Lushan_1和Lushan_2的0.5米像素间距，
与论文报告的Map World影像分辨率一致。

Lushan_3的3米像素间距，
与论文报告的PlanetScope影像分辨率一致。

这些均为来源推断，而不是文件级传感器确证。

特别注意：
- 不能将Lushan_1/2直接认定为UAV影像
- 不能确定Map World底层卫星传感器型号
- 不能仅凭GeoTIFF像素间距断定原生光学分辨率
- 不能假定全部论文数据源都包含在公开测试包中

## 待解决事项

已向GDCLD作者发送来源确认邮件，尚未收到回复。

仍需核实：
1. Lushan_1/2各自的实际数据提供方与底层传感器
2. Lushan_3是否确为PlanetScope
3. 公开测试包中是否包含论文描述的0.2米UAV影像

当前状态：source_inferred_file_level_confirmation_pending
