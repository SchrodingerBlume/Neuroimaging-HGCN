import shutil
from nilearn import datasets

# 自动下载Harvard-Oxford atlas (cortical, probability map)
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_path_src = atlas.filename
print("nilearn下载到临时路径:", ho_path_src)

# 指定目标路径（你的atlas目录下）
dst_path = "./atlas/ho.nii.gz"
shutil.copy(ho_path_src, dst_path)
print("已拷贝到:", dst_path)