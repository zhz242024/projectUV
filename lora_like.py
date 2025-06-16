#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lora_scan.py ―― 逐个 eGeMAPS 特征 + LoRA 微调
------------------------------------------------
* 只依赖 torch / numpy / pandas / sklearn
* 冻结原 Helix-Transformer；embed 层换成 rank-4 LoRA
* 输入维度固定 12 (= 11 core + 1 eGe)
* 每个特征在 train/dev 上训练 N epoch，保存
      - lora_runs/ege_###.pth     （微调后权重）
      - lora_runs/ege_###.json    （F1 / P / R / AUROC）
"""

import argparse, json, math
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (f1_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

# ---------- 自己的代码 ----------
from train_helix import load_data_for_id
from models.helix_transformer import HelixTransformer

CORE = ['trustworthiness', 'arousal', 'valence',
        'trust_diff', 'arousal_diff', 'valence_diff',
        'trust_diff2', 'arousal_diff2', 'valence_diff2',
        'VA_adjusted', 'log_trust_VA']
EGEMAPS = [str(i) for i in range(88)]                       # 0-87


# ──────────────────────────────────────────────────────────────
#  LoRA Linear
# ──────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():          # 冻结原权重
            p.requires_grad = False

        in_f, out_f   = base.in_features, base.out_features
        dev           = base.weight.device        # <<< 关键 1：跟 embed 同设备
        self.A        = nn.Parameter(torch.randn(r,  in_f, device=dev) * 0.01)  # <<< 关键 2
        self.B        = nn.Parameter(torch.zeros(out_f, r,      device=dev))    # <<< 关键 3
        self.scaling  = alpha / r
        self.drop     = nn.Dropout(dropout)

    def forward(self, x):
        y    = self.base(x)                       # (B,T,d)
        lora = (self.B @ (self.A @ x.transpose(-1, -2))).transpose(-1, -2)
        return y + self.scaling * self.drop(lora)



# ──────────────────────────────────────────────────────────────
#  派生特征 / 标签
# ──────────────────────────────────────────────────────────────
def add_feats(df):
    for c in ("trustworthiness", "arousal", "valence"):
        df[f"{c}_diff"] = df[c].diff().fillna(0)
        df[f"{c}_diff2"] = df[f"{c}_diff"].diff().fillna(0)

    df["trust_diff"]      = df["trustworthiness_diff"]
    df["trust_diff2"]     = df["trustworthiness_diff2"]

    a, b = 1, .5
    df["VA_adjusted"]      = np.sqrt(a*df.valence**2 + b*df.arousal**2)
    df["VA_adjusted_diff"] = df.VA_adjusted.diff().fillna(0)
    df["trust_VA_ratio"]   = (df.trustworthiness_diff.abs()
                              / (df.VA_adjusted_diff.abs() + 1e-10))
    df["log_trust_VA"]     = np.log1p(df.trust_VA_ratio)

    return df


def make_windows(arr, L=4):
    return [arr[i:i + L] for i in range(0, len(arr) - L + 1, L)]


def make_labels(df, L=4):
    t = df.timestamp.astype(float).values
    lab = []
    for s in range(0, len(df) - L + 1, L):
        v = np.polyfit(t[s:s + L], df.valence.values[s:s + L], 1)[0]
        a = np.polyfit(t[s:s + L], df.arousal.values[s:s + L], 1)[0]
        tr = np.polyfit(t[s:s + L], df.trustworthiness.values[s:s + L], 1)[0]
        lab.append(1 if (v < 0 and a > 0 and tr < 0) else 0)
    return np.array(lab, dtype=np.int8)


# ──────────────────────────────────────────────────────────────
def build(base_ckpt, d_model, n_heads, n_layers, device, idx):
    # mdl = HelixTransformer(
    #     feat_dim=12, d_model=d_model, n_heads=n_heads,
    #     n_layers=n_layers, dropout=0.1, max_len=1024).to(device)
    # ckpt = torch.load(base_ckpt, map_location=device)
    # ckpt.pop("embed.weight", None)
    # model.load_state_dict(ckpt, strict=False)

    # # mdl.load_state_dict(torch.load(base_ckpt, map_location=device), strict=False)
    # mdl.embed = LoRALinear(mdl.embed, r=4, alpha=8, dropout=0.05)


#for bas
    # mdl = HelixTransformer(
    #     feat_dim=12, d_model=d_model, n_heads=n_heads,
    #     n_layers=n_layers, dropout=0.1, max_len=1024).to(device)


    # feat_dim = 11 if args.ege_idx == -1 else 12

    feat_dim = 11 if idx == -1 else 12
    mdl = HelixTransformer(
        feat_dim=feat_dim,
         d_model=d_model, n_heads=n_heads,
         n_layers=n_layers, dropout=0.1, max_len=1024).to(device)



    # —— 载入 99 维 checkpoint，但 ❶ 把 embed.weight 去掉 ❷ 冻结主干 ——
    ckpt = torch.load(base_ckpt, map_location=device)
    ckpt.pop("embed.weight", None)          # ← 只删这一块
    mdl.load_state_dict(ckpt, strict=False)

    # ❸ 主干全部冻结（除了 embed）
    for n, p in mdl.named_parameters():
        if not n.startswith("embed"):
            p.requires_grad = False

    # ❹ 用 LoRA 包装 embed
    mdl.embed = LoRALinear(mdl.embed, r=4, alpha=8, dropout=0.05)

    return mdl


def run_one(idx, args, train_ids, dev_ids):
    device = args.device
    idx = args.ege_idx 
    mdl = build(args.base_ckpt, args.d_model,
                args.n_heads, args.n_layers, device, idx)
    

    # ============ 数据 =============
    train, dev = [], []
    for fid in train_ids + dev_ids:
        df = add_feats(load_data_for_id(fid))
        # feats = df[CORE + [EGEMAPS[idx]]].to_numpy(np.float32)

        if idx == -1:                        # 纯 11 维 baseline
            feats = df[CORE].to_numpy(np.float32)
        else:                                # 11 + 1 eGe
            feats = df[CORE + [EGEMAPS[idx]]].to_numpy(np.float32)

        wins = make_windows(feats)
        labs = make_labels(df)
        ds = list(zip(wins, labs))
        (train if fid in train_ids else dev).extend(ds)

    # 重新 fit 12-维 scaler
    scaler = StandardScaler().fit(np.vstack([w for w, _ in train]))
    train = [(scaler.transform(w), l) for w, l in train]
    dev = [(scaler.transform(w), l) for w, l in dev]

    # ============ 训练 =============
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, mdl.parameters()), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    mdl.train()
    for ep in range(args.epochs):
        np.random.shuffle(train)
        total = 0
        for w, l in train:
            x = torch.tensor(w, device=device).unsqueeze(0)
            y = torch.tensor([[l]], device=device, dtype=torch.float32)
            opt.zero_grad()
            # loss = loss_fn(mdl(x)[0], y)
            logits = mdl(x)[0].mean(dim=1)           # (B,1)  ← 对 T 取均值池化
            loss   = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[{idx:02d}] epoch {ep + 1}  loss={total / len(train):.4f}")

    # ============ 评测 =============
    mdl.eval(); y_true, y_prob = [], []
    with torch.no_grad():
        for w, l in dev:
            p = torch.sigmoid(
                mdl(torch.tensor(w, device=device).unsqueeze(0))[0]).mean().item()
            y_prob.append(p); y_true.append(l)

    y_true = np.array(y_true); y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    metr = dict(F1=round(f1, 4), P=round(p, 4), R=round(r, 4), AUROC=round(auc, 4))
    print(f"[{idx:02d}] dev {metr}")

    # out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    out = Path(os.getenv("OUT_DIR", args.out_dir)); out.mkdir(exist_ok=True)
    torch.save(mdl.state_dict(), out / f"ege_{idx:03d}.pth")
    json.dump(metr, open(out / f"ege_{idx:03d}.json", "w"), indent=2)
    return metr


# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ege_idx", type=int, required=True)
    ap.add_argument("--base_ckpt", required=True,
                    help="原 Helix-Transformer 99 维 checkpoint")
    ap.add_argument("--out_dir", default="lora_runs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    part = pd.read_csv("partition.csv")
    train_ids = part[part.Proposal == "train"].Id.tolist()
    dev_ids = part[part.Proposal == "devel"].Id.tolist()

    run_one(args.ege_idx, args, train_ids, dev_ids)


if __name__ == "__main__":
    main()