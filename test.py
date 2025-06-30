import nibabel as nib
import numpy as np

# 路径
infile = './atlas/cc200.nii.gz'
outfile = './atlas/cc200_labels3D.nii.gz'

img = nib.load(infile)
data = img.get_fdata()  # (x, y, z, 200)

# 统计每个ROI的最大值
roi_max = np.max(data, axis=(0,1,2))
print("每个ROI最大值:", roi_max)
print("最小最大值:", roi_max.min(), "最大最大值:", roi_max.max())

# 动态设置极小阈值（比如max中最小的1%作为背景）
# 通常建议 threshold = roi_max.min() * 0.9 或直接 threshold=1e-6
threshold = max(roi_max.min() * 0.9, 1e-6)
print(f"使用阈值: {threshold}")

# 每体素分配给最大响应ROI（最大值小于阈值则为背景）
maxval = np.max(data, axis=3)
argmax = np.argmax(data, axis=3) + 1
label_map = np.where(maxval > threshold, argmax, 0).astype(np.int16)

# 标签分布统计
uniq, cnts = np.unique(label_map, return_counts=True)
print('标签统计:')
for u, c in zip(uniq, cnts):
    print(f'{u:3d}: {c}')

# 确认非零标签数
print('有效分区数(不含背景):', np.sum(np.array(uniq)>0))

# 保存
nib.save(nib.Nifti1Image(label_map, img.affine), outfile)
print('已保存:', outfile)
roi_max = np.max(data, axis=(0,1,2))
print("每个ROI最大值:", roi_max)
print("非零最大值的ROI数:", np.sum(roi_max > 1e-6))
