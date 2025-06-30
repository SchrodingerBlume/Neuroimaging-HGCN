import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

# ==== 数据准备：假定已经FC特征拼好并对齐 ====
# 路径要按实际修改，特征 shape: [N, D]，标签 shape: [N]
X_aal = pd.read_csv('Results/NYU/AAL/FC_features_NYU_AAL.csv', header=None).values
X_ho = pd.read_csv('Results/NYU/HO/FC_features_NYU_HO.csv', header=None).values
X_cc200 = pd.read_csv('Results/NYU/CC200/FC_features_NYU_CC200.csv', header=None).values
y = pd.read_csv('Results/NYU/AAL/labels_NYU_AAL.csv')['label'].values - 1  # 保证和AAL对齐

X_aal = np.nan_to_num(X_aal, nan=0.0)
X_ho = np.nan_to_num(X_ho, nan=0.0)
X_cc200 = np.nan_to_num(X_cc200, nan=0.0)
N = X_aal.shape[0]
print("样本总数:", N)

def build_identity_hypergraph(num_nodes, num_edges):
    return torch.eye(num_nodes, num_edges)

H_aal = build_identity_hypergraph(N, X_aal.shape[1])
H_ho = build_identity_hypergraph(N, X_ho.shape[1])
H_cc200 = build_identity_hypergraph(N, X_cc200.shape[1])

# ==== HGCN定义 ====
class HypergraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def forward(self, X, H):
        Dv = torch.diag(H.sum(1))
        De = torch.diag(H.sum(0))
        Dv_inv_sqrt = torch.linalg.inv(torch.sqrt(Dv + 1e-5 * torch.eye(Dv.size(0), device=X.device)))
        De_inv = torch.linalg.inv(De + 1e-5 * torch.eye(De.size(0), device=X.device))
        H_t = H.t()
        XW = X @ self.weight
        out = Dv_inv_sqrt @ H @ De_inv @ H_t @ Dv_inv_sqrt @ XW
        if self.bias is not None:
            out = out + self.bias
        return out

class BranchHGCN(nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=32):
        super().__init__()
        self.hg1 = HypergraphConvolution(in_dim, hid_dim)
        self.hg2 = HypergraphConvolution(hid_dim, out_dim)
    def forward(self, X, H):
        x = F.relu(self.hg1(X, H))
        x = F.relu(self.hg2(x, H))
        return x

class MultiBranch_HGCN_Classifier(nn.Module):
    def __init__(self, dim_aal, dim_ho, dim_cc200, out_dim=32, num_classes=2):
        super().__init__()
        self.branch_aal = BranchHGCN(dim_aal, 64, out_dim)
        self.branch_ho = BranchHGCN(dim_ho, 64, out_dim)
        self.branch_cc200 = BranchHGCN(dim_cc200, 64, out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x1, h1, x2, h2, x3, h3):
        f1 = self.branch_aal(x1, h1)
        f2 = self.branch_ho(x2, h2)
        f3 = self.branch_cc200(x3, h3)
        fused = torch.cat([f1, f2, f3], dim=1)
        out = self.classifier(fused)
        return out

def eval_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp + 1e-8)
    return acc, rec, specificity, pre, f1, auc

# ==== 5折交叉验证 ====
from sklearn.model_selection import StratifiedKFold
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_aal, y), 1):
    print(f"\n=== Fold {fold} ===")
    X1_train = torch.tensor(X_aal[train_idx], dtype=torch.float32)
    X1_test  = torch.tensor(X_aal[test_idx], dtype=torch.float32)
    H1_train = H_aal[train_idx]
    H1_test  = H_aal[test_idx]
    X2_train = torch.tensor(X_ho[train_idx], dtype=torch.float32)
    X2_test  = torch.tensor(X_ho[test_idx], dtype=torch.float32)
    H2_train = H_ho[train_idx]
    H2_test  = H_ho[test_idx]
    X3_train = torch.tensor(X_cc200[train_idx], dtype=torch.float32)
    X3_test  = torch.tensor(X_cc200[test_idx], dtype=torch.float32)
    H3_train = H_cc200[train_idx]
    H3_test  = H_cc200[test_idx]
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    y_test  = torch.tensor(y[test_idx], dtype=torch.long)
    train_ds = TensorDataset(X1_train, H1_train, X2_train, H2_train, X3_train, H3_train, y_train)
    test_ds  = TensorDataset(X1_test, H1_test, X2_test, H2_test, X3_test, H3_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    model = MultiBranch_HGCN_Classifier(
        X_aal.shape[1], X_ho.shape[1], X_cc200.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for x1, h1, x2, h2, x3, h3, label in train_loader:
            x1, h1, x2, h2, x3, h3, label = \
                x1.to(device), h1.to(device), x2.to(device), h2.to(device), x3.to(device), h3.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(x1, h1, x2, h2, x3, h3)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * label.size(0)
        if epoch % 5 == 0 or epoch == 1 or epoch == 20:
            print(f"Epoch {epoch} 训练损失: {total_loss/len(train_ds):.5f}")

    # --- 每折评估 ---
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x1, h1, x2, h2, x3, h3, label in test_loader:
            x1, h1, x2, h2, x3, h3, label = \
                x1.to(device), h1.to(device), x2.to(device), h2.to(device), x3.to(device), h3.to(device), label.to(device)
            output = model(x1, h1, x2, h2, x3, h3)
            prob = F.softmax(output, dim=1)[:, 1].cpu().numpy()
            pred = output.argmax(1).cpu().numpy()
            y_pred.extend(pred)
            y_prob.extend(prob)
            y_true.extend(label.cpu().numpy())
    acc, rec, spe, pre, f1, auc = eval_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    all_metrics.append([acc, rec, spe, pre, f1, auc])
    print(f"[Fold {fold} 验证] acc={acc:.4f} | recall={rec:.4f} | specificity={spe:.4f} | precision={pre:.4f} | f1={f1:.4f} | auc={auc:.4f}")

# ==== 输出5折平均与标准差 ====
all_metrics = np.array(all_metrics)
names = ['acc', 'recall', 'spec', 'precision', 'f1', 'auc']
print("\n=== 5折均值±方差 ===")
for i, n in enumerate(names):
    print(f"{n}: {all_metrics[:,i].mean():.4f} ± {all_metrics[:,i].std():.4f}")