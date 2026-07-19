"""
下载并解压 EuroSAT_RGB 数据集。

数据来源: ModelScope (https://www.modelscope.cn/datasets/lhh010/EuroSAT_RGB)
使用方式:
  1. 从 ModelScope 手动下载 zip 文件, 放到项目根目录的 data/ 下
  2. 运行本脚本解压并验证数据集结构
     python scripts/download_eurosat.py
"""

import os
import zipfile

# 1. 设定存放数据的本地目录
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(data_dir, exist_ok=True)

zip_path = os.path.join(data_dir, "EuroSAT_RGB.zip")
extract_dir = data_dir  # 解压后的目录

# ---- 如需自动下载 (Zenodo 镜像, 备用) ----
# import urllib.request
# zenodo_url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
# if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
#     os.makedirs(extract_dir, exist_ok=True)
#     if not os.path.exists(zip_path):
#         print("正在从 Zenodo 下载数据集 (约90MB), 请稍候...")
#         try:
#             req = urllib.request.Request(zenodo_url, headers={"User-Agent": "Mozilla/5.0"})
#             with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
#                 out_file.write(response.read())
#             print("下载完成!")
#         except Exception as e:
#             print(f"网络下载失败: {e}")
#             exit()
# -------------------------------------------

if not os.path.exists(zip_path):
    print(f"错误: 找不到 {zip_path}")
    print(f"请先从 ModelScope 下载数据集: https://www.modelscope.cn/datasets/lhh010/EuroSAT_RGB")
    print(f"将下载的 EuroSAT_RGB.zip 放到 {data_dir}/ 下, 然后重新运行本脚本。")
    exit(1)

print("正在解压数据集...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
print("解压完成!")

# 3. 智能寻找解压后的图像目录 (绝对容错版)
dataset_root = extract_dir
for root, dirs, files in os.walk(extract_dir):
    # 只要该文件夹下包含 "Forest" 和 "River" 这两个经典的 EuroSAT 类别, 就绝对是正确的根目录!
    if "Forest" in dirs and "River" in dirs:
        dataset_root = root
        break

print(f"已精准定位到数据集分类目录: {dataset_root}")

# 4. 验证数据集完整性
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder(root=dataset_root)
print("=" * 50)
print("成功加载航空航天卫星数据集!")
print(f"图像总数: {len(train_dataset)} 张")
print(f"图像类别 (10分类): {train_dataset.classes}")
print("=" * 50)
