# CAS Moxitaidi分辨率与传感器来源核查

## 1. 原始论文证据

来源：
Xu et al., CAS Landslide Dataset, Scientific Data, 2024.
https://doi.org/10.1038/s41597-023-02847-z

论文Table 2报告：
- Moxitaidi (SAT)：Sentinel-2/L2A，0.6 m
- Moxitaidi (UAV-0.6m)：UAV，0.6 m

注意：以上为论文报告值，不是本项目独立实测值。

## 2. 传感器规格证据

Copernicus Sentinel-2 MSI官方文档：
https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/PDFS_ADFS/index.html

MSI原生波段空间分辨率为10、20、60米。
因此，论文中的SAT 0.6米不能直接解释为
Sentinel-2传感器的原生空间分辨率。

## 3. 本地切片核查

SAT001、SAT100、UAV0519、UAV0533：
- CRS：无
- GCP：无
- RPC：无
- TIFF无可确认的地面尺度变换

SAT100至UAV0519/0533的局部仿射尺度约为2。
该值为像素坐标尺度，不能单独当作绝对GSD。

## 4. 科研判定

- SAT论文报告GSD：0.6 m/px
- UAV论文报告GSD：0.6 m/px
- SAT传感器原生分辨率：与论文报告值存在待解释差异
- 实际切片绝对GSD：未独立验证
- 绝对有效分辨率：未确定
- 可能涉及重采样或来源描述问题，但原因未证实
- 不得直接以论文报告的0.6米认定SAT与UAV物理分辨率相等

状态：provenance_conflict_requires_review

## 5. 后续行动

核查CAS原始数据处理代码和说明；
必要时向作者询问Moxitaidi SAT实际影像来源、
0.6米数值含义及重采样处理过程。

第100步生成的对照表须结合本核查记录使用，
后续应更新自动生成脚本中的标称值和来源说明。
