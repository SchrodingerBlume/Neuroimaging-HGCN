import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

# ========== 1. 读取AAL Atlas ==========
atlas_path = './atlas/aal.nii.gz'
aal_img = nib.load(atlas_path)
atlas_data = aal_img.get_fdata()

# ========== 2. 计算每个分区中心坐标 ==========
labels = np.unique(atlas_data)
labels = labels[labels != 0]  # 去掉0背景
coords = []
for label in labels:
    inds = np.argwhere(atlas_data == label)
    center_voxel = inds.mean(axis=0)
    center_mm = nib.affines.apply_affine(aal_img.affine, center_voxel)
    coords.append(center_mm)
coords = np.array(coords)
print("coords shape:", coords.shape)  # 应该是(116, 3)

# ========== 3. 读取你的FC特征并还原116×116 ==========
fc_csv = 'Results/NYU/AAL/FC_features_NYU_AAL.csv'
fc_arr = pd.read_csv(fc_csv, header=None).values   # shape: [N, 6670]

def vec_to_symmat(vec, n_node=116):
    mat = np.zeros((n_node, n_node))
    iu = np.triu_indices(n_node, 1)
    mat[iu] = vec
    mat = mat + mat.T
    return mat

# 这里取所有样本均值（你可以换成任意一行对应单个样本）
mean_fc_vec = np.nanmean(fc_arr, axis=0)
fc_mat = vec_to_symmat(mean_fc_vec, 116)
np.fill_diagonal(fc_mat, 0)

# ========== 4. 画b部分: 三正交切面AAL分区 ==========
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
plotting.plot_roi(aal_img, display_mode='y', cut_coords=[-40], cmap='tab20b', axes=axes[0], draw_cross=True)
plotting.plot_roi(aal_img, display_mode='x', cut_coords=[2],  cmap='tab20b', axes=axes[1], draw_cross=True)
plotting.plot_roi(aal_img, display_mode='z', cut_coords=[-1], cmap='tab20b', axes=axes[2], draw_cross=True)
for a in axes: a.set_axis_off()
plt.suptitle('b   AAL atlas', x=0.13, y=0.98, fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# ========== 5. 画c部分: AAL 116节点+FC连线图 ==========
plotting.plot_connectome(fc_mat, coords,
                         node_size=22,
                         edge_threshold="98.5%",    # 最强的20%线
                         edge_cmap='bwr',
                         node_color='auto',
                         title='c   AAL 116')
plt.show()