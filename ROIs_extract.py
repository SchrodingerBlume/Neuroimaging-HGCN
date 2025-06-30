import os
import glob
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker

# --------- 参数 ---------
func_dir = "./abide_NYU_func_preproc"
atlas_dict = {
    "AAL": "./atlas/aal.nii.gz",
    "HO": "./atlas/ho.nii.gz",
    "CC200": "./atlas/cc200.nii.gz",
}
out_base = "./roi_timeseries"
for name in atlas_dict:
    os.makedirs(os.path.join(out_base, f"NYU_{name}"), exist_ok=True)

# --------- 功能像文件和受试者ID ---------
func_files = sorted(glob.glob(os.path.join(func_dir, "*_func_preproc.nii.gz")))
subj_ids = ['_'.join(os.path.basename(f).split('_')[:2]) for f in func_files]
print(f"共找到功能像受试者：{len(subj_ids)} 个")

# --------- 统计每atlas目录已存在的受试者 ---------
atlas_subj = {}
for name in atlas_dict:
    ts_dir = os.path.join(out_base, f"NYU_{name}")
    csvs = glob.glob(os.path.join(ts_dir, f"*_{name}_timeseries.csv"))
    ids = set(['_'.join(os.path.basename(f).split('_')[:2]) for f in csvs])
    atlas_subj[name] = ids

# --------- 取功能像受试者与三atlas已有csv的交集 ---------
existing = set(subj_ids)
for ids in atlas_subj.values():
    if ids:
        existing &= ids
print(f"三atlas交集受试者数: {len(existing)}")

# --------- 针对每atlas批量提取，只保留交集 ---------
for name, atlas_file in atlas_dict.items():
    ts_dir = os.path.join(out_base, f"NYU_{name}")
    masker = NiftiLabelsMasker(labels_img=atlas_file, standardize=True)
    for func_file in func_files:
        basename = os.path.basename(func_file)
        subj_id = '_'.join(basename.split('_')[:2])
        if subj_id not in existing:
            continue
        out_csv = os.path.join(ts_dir, f"{subj_id}_{name}_timeseries.csv")
        if os.path.exists(out_csv):
            print(f"[{name}] 跳过已有: {out_csv}")
            continue
        try:
            ts = masker.fit_transform(func_file)  # shape: [n_timepoints, n_ROI]
            pd.DataFrame(ts).to_csv(out_csv, index=False, header=False)
            print(f"[{name}] 已保存: {out_csv}, 形状: {ts.shape}")
        except Exception as e:
            print(f"[{name}] 处理失败: {func_file}, 错误: {e}")

print("三atlas交集ROI时序提取并分三目录保存，全部完成！")