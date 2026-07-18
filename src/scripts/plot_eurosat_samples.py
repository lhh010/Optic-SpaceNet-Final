# -*- coding: utf-8 -*-
"""EuroSAT 10 类样例拼图 + 类别信息表 — 存 docs/figures/eurosat_samples.{png,pdf}
从仓库根目录运行: python src/scripts/plot_eurosat_samples.py"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os, random

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA = 'data/EuroSAT_RGB'
# 文件夹名 -> 友好显示名（与表格“类别名”列一致）
display = {'AnnualCrop': 'Annual Crop', 'Forest': 'Forest',
           'HerbaceousVegetation': 'Herbaceous', 'Highway': 'Highway',
           'Industrial': 'Industrial', 'Pasture': 'Pasture',
           'PermanentCrop': 'Permanent Crop', 'Residential': 'Residential',
           'River': 'River', 'SeaLake': 'Sea Lake'}
folders = sorted(c for c in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, c)))
random.seed(42)

fig = plt.figure(figsize=(13, 9.6), dpi=300)
# 上：图片网格；下：信息表。top 留大给 suptitle 与首行拉开距离。
gs = gridspec.GridSpec(2, 1, height_ratios=[2.7, 1.0], hspace=0.16,
                       left=0.045, right=0.955, top=0.875, bottom=0.075)
gs_img = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[0],
                                          hspace=0.55, wspace=0.08)

for i, c in enumerate(folders):
    r, col = divmod(i, 5)
    ax = fig.add_subplot(gs_img[r, col])
    f = random.choice(sorted(os.listdir(os.path.join(DATA, c))))
    ax.imshow(Image.open(os.path.join(DATA, c, f)))
    ax.set_title(display.get(c, c), fontsize=12, pad=6)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('EuroSAT 数据集 · 10 类地物样例（Sentinel-2 卫星遥感，64×64 RGB）',
             fontsize=16, fontweight='bold', y=0.965)

# ---- 信息表 ----
ax_t = fig.add_subplot(gs[1]); ax_t.axis('off')
columns = ['类别名', '地物含义', '样本张数', '类别名', '地物含义', '样本张数']
rows = [
    ['Annual Crop', '一年生农田', '3,000', 'Pasture',       '开阔牧场',   '2,000'],
    ['Forest',      '森林/林地',  '3,000', 'Permanent Crop', '常年作物',   '2,500'],
    ['Herbaceous',  '草本植被',   '3,000', 'Residential',    '住宅建筑',   '3,000'],
    ['Highway',     '普通公路',   '2,500', 'River',          '天然河流',   '2,500'],
    ['Industrial',  '工业区厂房', '2,500', 'Sea Lake',       '海、淡水湖', '3,000'],
]
tbl = ax_t.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center',
                 colWidths=[0.17, 0.155, 0.115, 0.17, 0.155, 0.115])
tbl.auto_set_font_size(False); tbl.set_fontsize(11.5)
tbl.scale(1, 1.75)

# 表头：蓝底白字加粗
for c in range(6):
    cell = tbl[(0, c)]
    cell.set_facecolor('#0072B2')
    cell.set_text_props(color='white', fontweight='bold')
# 两组底色区分（左组浅蓝 / 右组浅绿），数据行
for r in range(1, 6):
    for c in range(3):
        tbl[(r, c)].set_facecolor('#EAF2FB')
    for c in range(3, 6):
        tbl[(r, c)].set_facecolor('#EAF7EE')
# 细边框
for key, cell in tbl.get_celld().items():
    cell.set_edgecolor('#CCCCCC'); cell.set_linewidth(0.8)

# 汇总行（图底）— 仅保留数据集固有的总样本数（val/test 划分属于下游项目，非数据集本身属性）
fig.text(0.5, 0.028,
         '总样本集大小：27,000 张',
         ha='center', fontsize=13, fontweight='bold')

out = 'docs/figures/eurosat_samples'
plt.savefig(f'{out}.png', dpi=300); plt.savefig(f'{out}.pdf')
print(f'saved: {out}.png / .pdf')
