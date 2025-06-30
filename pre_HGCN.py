import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix
)
from torch.utils.data import TensorDataset, DataLoader

# === 1. 数据读取与超图构建（只用AAL） ===
X_aal = pd.read_csv('Results/NYU/AAL/FC_features_NYU_AAL.csv', header=None).values
y = pd.read_csv('Results/NYU/AAL/labels_NYU_AAL.csv')['label'].values - 1  # 0/1分类
N = X_aal.shape[0]
X_aal = np.nan_to_num(X_aal, nan=0.0)

def build_identity_hypergraph(num_nodes, num_edges):
    return torch.eye(num_nodes, num_edges)


# def build_identity_hypergraph(num_nodes, num_edges):
 #   H = torch.ones((num_nodes, num_edges))
  #  return H
H_aal = build_identity_hypergraph(N, X_aal.shape[1])
idx = np.arange(N)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
X_train = torch.tensor(X_aal[train_idx], dtype=torch.float32)
X_test  = torch.tensor(X_aal[test_idx], dtype=torch.float32)
H_train = H_aal[train_idx]
H_test  = H_aal[test_idx]
y_train = torch.tensor(y[train_idx], dtype=torch.long)
y_test  = torch.tensor(y[test_idx], dtype=torch.long)

# === 2. 单分支HGCN ===
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

class AAL_HGCN_Classifier(nn.Module):
    def __init__(self, dim_aal, out_dim=32, num_classes=2):
        super().__init__()
        self.branch_aal = BranchHGCN(dim_aal, 64, out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x1, h1):
        f1 = self.branch_aal(x1, h1)
        out = self.classifier(f1)
        return out

# === 3. 训练与多指标验证 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AAL_HGCN_Classifier(X_aal.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
batch_size = 16
train_ds = TensorDataset(X_train, H_train, y_train)
test_ds  = TensorDataset(X_test, H_test, y_test)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)

def eval_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp + 1e-8)
    return acc, rec, specificity, pre, f1, auc

for epoch in range(1, 31):
    model.train()
    total_loss = 0
    for x1, h1, label in train_loader:
        x1, h1, label = x1.to(device), h1.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(x1, h1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.size(0)
    print(f"Epoch {epoch} 训练损失: {total_loss/len(train_ds):.4f}")
    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for x1, h1, label in test_loader:
                x1, h1, label = x1.to(device), h1.to(device), label.to(device)
                output = model(x1, h1)
                prob = F.softmax(output, dim=1)[:, 1].cpu().numpy()
                pred = output.argmax(1).cpu().numpy()
                y_pred.extend(pred)
                y_prob.extend(prob)
                y_true.extend(label.cpu().numpy())
        acc, rec, spe, pre, f1, auc = eval_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
        print(
            f"[验证] acc={acc:.3f} | recall/sens={rec:.3f} | specificity={spe:.3f} | precision={pre:.3f} | f1={f1:.3f} | auc={auc:.3f}"
        )