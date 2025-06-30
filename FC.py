import os
import numpy as np
import pandas as pd

# ---------- 参数 ----------
roi_dir    = ('./roi_timeseries/NYU_AAL')       # ROI 时序文件夹
fc_out_dir = './FC_matrices/NYU_AAL'          # FC 输出文件夹
os.makedirs(fc_out_dir, exist_ok=True)

# 要剔除的 15 名受试者（只写数字）
exclude_raw = [
    50964, 50977, 50974, 51095, 50968,
    50981, 50983, 50990, 50997, 50982,
    50998, 51025, 50952, 50959, 50960
]
exclude_ids = {f'NYU_{i:06d}' for i in exclude_raw}

# ---------- 遍历文件 ----------
csv_files = [f for f in os.listdir(roi_dir) if f.endswith('.csv')]
print(f'共找到 {len(csv_files)} 个 ROI 时序文件')
for csvf in csv_files:
    subj_id = '_'.join(csvf.split('_')[:2])        # NYU_0050952
    if subj_id in exclude_ids:
        print(f'跳过剔除受试者 {subj_id}')
        continue

    ts = pd.read_csv(os.path.join(roi_dir, csvf), header=None).values  # [T, ROI]
    if ts.shape[0] < 10 or ts.shape[1] < 10:
        print(f'跳过异常文件 {csvf}, 形状 {ts.shape}')
        continue

    # ---------- 方案 2：剔除 σ = 0 ROI ----------
    stds = np.std(ts, axis=0)
    zero_var_mask = stds == 0
    if zero_var_mask.any():
        n_bad = zero_var_mask.sum()
        print(f'  {subj_id}: 检测到 {n_bad} 个 σ=0 ROI，置为 NaN')
        ts[:, zero_var_mask] = np.nan               # 这些列后续相关系数会得到 NaN

    # ---------- 计算 FC ----------
    fc = np.corrcoef(ts.T)                         # [ROI, ROI]，含 NaN

    # ---------- 把 NaN / Inf 统一填 0 ----------
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- 保存 ----------
    out_prefix = os.path.join(fc_out_dir, f'{subj_id}_AAL_FC')
    np.save(out_prefix + '.npy', fc)
    pd.DataFrame(fc).to_csv(out_prefix + '.csv', index=False, header=False)
    print(f'  已保存 {out_prefix}.npy / .csv')

print('全部 FC 矩阵计算完毕！')