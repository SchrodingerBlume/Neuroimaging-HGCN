import requests
import pandas as pd

def url_exists(url):
    try:
        resp = requests.head(url, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        # 输出异常信息调试用
        print(f"Exception for url {url}: {e}")
        return False

# --- 参数区 ---
sites = ['NYU']
pipeline = 'cpac'
strategy = 'filt_global'
derivatives = [
    ('rois_aal', '1D'),
    ('rois_ho', '1D'),
    ('rois_cc200', '1D'),
    ('func_preproc', 'nii.gz')
]

# --- 读取phenotype表 ---
csv_file = "Phenotypic_V1_0b_preprocessed1.csv"
df = pd.read_csv(csv_file)
df = df[df['SITE_ID'].isin(sites) & (df['FILE_ID'] != 'no_filename')]

for derivative, ext in derivatives:
    found = 0
    found_ids = []
    total = len(df)
    print(f"\n正在检查 {derivative}，总共 {total} 个file_id，请稍候……")
    for idx, file_id in enumerate(df['FILE_ID']):
        url = f"https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/{pipeline}/{strategy}/{derivative}/{file_id}_{derivative}.{ext}"
        if url_exists(url):
            found += 1
            found_ids.append(file_id)
        # 实时输出进度
        if (idx+1) % 50 == 0 or (idx+1)==total:
            print(f"已检查 {idx+1}/{total}，当前可用 {found}")
    print(f"\n{derivative} 可用总数: {found}")
    print(f"前10个可用 file_id: {found_ids[:10]}")
    print("="*40)

print("所有类型检查完毕。")
