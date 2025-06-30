import pandas as pd

df = pd.read_csv('Phenotypic_V1_0b_preprocessed1.csv')
nyu_asd = df[(df['SITE_ID']=='NYU') & (df['DX_GROUP']==1)]

# 1. 人工fail/maybe/skull-striping fail/tip of frontal lobe cropped
def is_manual_bad(row):
    bad_keywords = ['fail', 'maybe', 'skull-striping fail', 'tip of frontal lobe cropped']
    cols = [
        'qc_rater_1', 'qc_notes_rater_1', 'qc_anat_rater_2', 'qc_anat_notes_rater_2',
        'qc_func_rater_2', 'qc_func_notes_rater_2'
    ]
    for c in cols:
        v = str(row.get(c, '')).lower()
        if any(bad in v for bad in bad_keywords):
            return True
    return False

manual_bad = nyu_asd[nyu_asd.apply(is_manual_bad, axis=1)]
manual_bad_ids = manual_bad['SUB_ID'].tolist()

# 2. BMI=-9999，人工没被标记bad
auto_bad = nyu_asd[(nyu_asd['BMI'] == -9999) & (~nyu_asd['SUB_ID'].isin(manual_bad_ids))]
auto_bad_ids = auto_bad['SUB_ID'].tolist()

# 3. 剩下还不够15个，再补其他自动化异常
# 这里暂以BMI=-9999为例，实际你可以加上其它指标的异常（比如anat/func_fber极端值等）
need_n = 17 - len(manual_bad_ids)
final_bad_ids = manual_bad_ids + auto_bad_ids[:max(0, need_n)]

print('最优先人工fail/maybe/skull-striping fail的SUB_ID：', manual_bad_ids)
print('补足15个的SUB_ID（自动异常）:', auto_bad_ids[:max(0, need_n)])
print('合计15个最可能被剔除的SUB_ID：', final_bad_ids)