import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# 读取FC矩阵
fc_file = './FC_matrices/NYU_CC200/NYU_0050955_AAL_FC.csv'
fc = pd.read_csv(fc_file, header=None).values

plt.figure(figsize=(4, 3.6))
ax = sns.heatmap(
    fc, cmap='RdBu_r', center=0, vmin=-0.6, vmax=0.8,
    cbar=True, cbar_kws={'shrink':0.93}
)

plt.xlabel('CC200 atlas')
plt.tight_layout()

# 只显示主刻度
N = fc.shape[0]
major_ticks = np.arange(0, N+1, 20)
ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)

# 如果想只显示0、20、40、60、80、100，可以截断到100
labels = [str(x) for x in major_ticks]
if N > 100:
    labels = [str(x) for x in major_ticks if x <= 100]
    ax.set_xticks([x for x in major_ticks if x <= 100])
    ax.set_yticks([x for x in major_ticks if x <= 100])
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels)
else:
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels)

# 主热图四周加黑色框（用patch确保四边完整）
rect = patches.Rectangle(
    (0, 0), N, N, linewidth=1, edgecolor='black', facecolor='none',
    transform=ax.transData, clip_on=False, zorder=10
)
ax.add_patch(rect)

# 色条加黑框
cbar = ax.collections[0].colorbar
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)

plt.show()
