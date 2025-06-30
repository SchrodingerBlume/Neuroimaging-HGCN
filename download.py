import os
import requests
import pandas as pd

def safe_download(url, out_path):
    # 断点续传：如果已存在则跳过
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"File exists, skip: {out_path}")
        return
    try:
        print(f"Downloading: {url}")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(resp.content)
        else:
            print(f"Failed: HTTP {resp.status_code} - {url}")
    except Exception as e:
        print(f"Failed: {url}\n{e}")

# 1. 下载元数据表
phenotype_url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
csv_file = "Phenotypic_V1_0b_preprocessed1.csv"
if not os.path.exists(csv_file):
    safe_download(phenotype_url, csv_file)

# 2. 设置参数
sites = ['USM','UM_1','UM_2'] # , 'NYU',, 'USM'   'UCLA_1', 'UCLA_2'  'UM_1','UM_2'
pipeline = 'cpac'
strategy = 'filt_global'
derivatives = [
     ('rois_aal', '1D'),
     ('rois_ho', '1D'),
     ('rois_cc200', '1D'),
     ('func_preproc', 'nii.gz')
]

# 3. 读取表格，筛选需要的数据
df = pd.read_csv(csv_file)
df = df[df['SITE_ID'].isin(sites)]
df = df[df['FILE_ID'] != 'no_filename']

# 4. 批量下载
for site in sites:
    for derivative, ext in derivatives:
        out_dir = f'./abide_{site}_{derivative}'
        os.makedirs(out_dir, exist_ok=True)
        sub_df = df[df['SITE_ID'] == site]
        for _, row in sub_df.iterrows():
            file_id = row['FILE_ID']
            url = f"https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/{pipeline}/{strategy}/{derivative}/{file_id}_{derivative}.{ext}"
            out_path = os.path.join(out_dir, f"{file_id}_{derivative}.{ext}")
            safe_download(url, out_path)

print("All downloads complete!")