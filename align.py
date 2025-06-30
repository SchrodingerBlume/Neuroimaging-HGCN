import os
import numpy as np
import pandas as pd
import re

# 参数
fc_dir = './FC_matrices/NYU_CC200'            # FC .csv 文件夹
pheno_csv = 'Phenotypic_V1_0b_preprocessed1.csv'
out_X_csv = 'Results/NYU/HO/FC_features_NYU_CC200.csv'
out_y_csv = 'Results/NYU/HO/labels_NYU_CC200.csv'

fc_files = [f for f in os.listdir(fc_dir) if f.endswith('_AAL_FC.csv')]
print(f"共找到{len(fc_files)}个FC矩阵csv文件。")

pheno = pd.read_csv(pheno_csv)
pheno = pheno[pheno['SITE_ID'].str.contains('NYU')]
pheno['SUBJ_ID'] = pheno['FILE_ID']

X_list, y_list, subj_list = [], [], []

for fcfile in fc_files:
    # 使用正则提取 NYU_0050952
    match = re.search(r'NYU_\d{7}', fcfile)
    if match:
        subj_id = match.group(0)
    else:
        print(f"无法提取受试者ID: {fcfile}，跳过")
        continue
    label_row = pheno[pheno['SUBJ_ID'] == subj_id]
    if label_row.empty:
        print(f"未找到标签: {fcfile}，跳过")
        continue
    label = int(label_row['DX_GROUP'].values[0])  # 1=ASD, 2=对照

    fc = pd.read_csv(os.path.join(fc_dir, fcfile), header=None).values
    fc_vec = fc[np.triu_indices(fc.shape[0], k=1)]
    X_list.append(fc_vec)
    y_list.append(label)
    subj_list.append(subj_id)

print(f"最终可用样本数：{len(X_list)}")

X = np.vstack(X_list)
y = np.array(y_list)
subj_list = np.array(subj_list)

pd.DataFrame(X).to_csv(out_X_csv, index=False, header=False)
pd.DataFrame({'subj_id': subj_list, 'label': y}).to_csv(out_y_csv, index=False)

print(f"已保存特征表: {out_X_csv}，标签表: {out_y_csv}")
print(f"X形状: {X.shape}, y形状: {y.shape}")
