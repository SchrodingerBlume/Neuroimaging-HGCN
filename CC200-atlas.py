from nilearn import datasets
import os
import shutil

# 1. 下载 Craddock 2012 atlas（含CC200与CC400）
cc = datasets.fetch_atlas_craddock_2012()
print("全部返回键:", cc.keys())

# 2. 获取CC200模板路径（一般是tcorr_mean_filenames[0]）
cc200_path = cc['tcorr_mean_filenames'][0]
print("CC200 分区模板路径:", cc200_path)

# 3. 拷贝到你的atlas目录下
dst_dir = "./atlas"
os.makedirs(dst_dir, exist_ok=True)
dst_path = os.path.join(dst_dir, "cc200.nii.gz")
shutil.copy(cc200_path, dst_path)
print("已复制到:", dst_path)

# 4. 可选：如需进一步检查
# import nibabel as nib
# img = nib.load(dst_path)
# print("shape:", img.shape)