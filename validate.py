# import os
# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
# from joblib import load



# # 设置模型路径和验证数据路径
# model_dir = "./models"
# data_path = "./data"
# partition_file = "./partition.csv"

# # 加载 partition.csv，获取验证集 ID
# partition_df = pd.read_csv(partition_file)
# dev_ids = partition_df[partition_df['Proposal'] == 'devel']['Id'].tolist()

# # 加载数据函数
# def load_data_for_id(file_id):
#     try:
#         label_paths = {
#             "valence": os.path.join(data_path, "label_segments", "valence"),
#             "arousal": os.path.join(data_path, "label_segments", "arousal"),
#             "trustworthiness": os.path.join(data_path, "label_segments", "trustworthiness")
#         }
#         egemaps_path = os.path.join(data_path, "egemaps", "egemaps")

#         valence_df = pd.read_csv(os.path.join(label_paths["valence"], f"{file_id}.csv"))
#         arousal_df = pd.read_csv(os.path.join(label_paths["arousal"], f"{file_id}.csv"))
#         trust_df = pd.read_csv(os.path.join(label_paths["trustworthiness"], f"{file_id}.csv"))

#         valence_df = valence_df[['timestamp', 'value']].rename(columns={'value': 'valence'})
#         arousal_df = arousal_df[['timestamp', 'value']].rename(columns={'value': 'arousal'})
#         trust_df = trust_df[['timestamp', 'value']].rename(columns={'value': 'trustworthiness'})

#         labels_df = valence_df.merge(arousal_df, on='timestamp', how='outer')
#         labels_df = labels_df.merge(trust_df, on='timestamp', how='outer').sort_values(by='timestamp')
#         labels_df = labels_df.interpolate(method='linear')

#         egemaps_df = pd.read_csv(os.path.join(egemaps_path, f"{file_id}.csv"))
#         egemaps_columns = ['timestamp'] + [str(i) for i in range(88)]
#         egemaps_df = egemaps_df[egemaps_columns]

#         combined_df = labels_df.merge(egemaps_df, on='timestamp', how='inner')
#         combined_df['File_ID'] = file_id
#         return combined_df

#     except Exception as e:
#         print(f"Error loading data for ID {file_id}: {e}")
#         return None

# # 创建 uncanny_label
# def create_uncanny_labels_with_window(data, window_size_ms=1000, sampling_rate_ms=250, 
#                                       trust_quantile=0.75, valence_quantile=0.75, arousal_quantile=0.75):
#     window_size = max(1, int(window_size_ms / sampling_rate_ms))
#     data['trust_diff1'] = data['trustworthiness'].diff().fillna(0)
#     data['valence_diff1'] = data['valence'].diff().fillna(0)
#     data['arousal_diff1'] = data['arousal'].diff().fillna(0)

#     trust_thresh = data['trust_diff1'].quantile(trust_quantile)
#     valence_thresh = data['valence_diff1'].quantile(valence_quantile)
#     arousal_thresh = data['arousal_diff1'].quantile(arousal_quantile)

#     trust_std = data['trust_diff1'].rolling(window=window_size).std().fillna(0)
#     valence_std = data['valence_diff1'].rolling(window=window_size).std().fillna(0)
#     arousal_std = data['arousal_diff1'].rolling(window=window_size).std().fillna(0)

#     data['uncanny_label'] = ((trust_std > trust_thresh) &
#                              (valence_std > valence_thresh) &
#                              (arousal_std > arousal_thresh)).astype(int)
#     return data

# # 加载验证数据
# def load_dev_data():
#     dev_data_list = []
#     for file_id in dev_ids:
#         data = load_data_for_id(file_id)
#         if data is not None:
#             # 在验证数据中创建 uncanny_label
#             data = create_uncanny_labels_with_window(data)
#             dev_data_list.append(data)
#     return pd.concat(dev_data_list) if dev_data_list else None

# # 模型评估函数
# def evaluate_model(model, X, y):
#     preds = model.predict(X)
#     acc = accuracy_score(y, preds)
#     recall = recall_score(y, preds)
#     f1 = f1_score(y, preds)
#     return acc, recall, f1

# # 主程序
# def main():
#     # 加载验证数据
#     dev_data = load_dev_data()
#     drop_cols = ['timestamp', 'valence', 'arousal', 'trustworthiness', 'uncanny_label',
#                  'trust_diff1', 'valence_diff1', 'arousal_diff1']
#     X_dev = dev_data.drop(columns=[col for col in drop_cols if col in dev_data.columns])
#     X_dev = X_dev[[str(i) for i in range(88)]]
#     y_dev = dev_data['uncanny_label']

#     # 需要比较的模型文件
#     model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]

#     results = []
#     for model_file in model_files:
#         model_path = os.path.join(model_dir, model_file)
#         model = load(model_path)
#         acc, recall, f1 = evaluate_model(model, X_dev, y_dev)
#         results.append({
#             'Model': model_file,
#             'Accuracy': acc,
#             'Recall': recall,
#             'F1 Score': f1
#         })
#         print(f"\n=== {model_file} ===")
#         print(f"Accuracy: {acc:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

#     # 比较所有模型性能
#     results_df = pd.DataFrame(results)
#     results_df.to_csv("model_validation_results.csv",mode = 'a',header =False, index=False)
#     print("\n模型验证结果已保存到 model_validation_results.csv")

# if __name__ == "__main__":
#     main()



# validate_helix.py
# -----------------------------------------------------------
# 评估 Helix-Transformer 在 dev 集上的窗口 / 帧级性能
# -----------------------------------------------------------
# """Validate Helix‑Transformer on the development split
# Usage (example):
#     python validate_helix.py \
#         --model_path ./transformer_outputs/helix_transformer_model.pth \
#         --d_model 192 --n_heads 4 \
#         --threshold 0.5
# The script re‑uses load_data_for_id and create_feature_embeddings from train_helix.py,
# computes window‑level probabilities, then prints F1 / Precision / Recall / AUROC.
# """
# import os
# import argparse
# import json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import (
#     f1_score,
#     precision_recall_fscore_support,
#     roc_auc_score,
# )

# # ---------- paths & IDs ----------
# BASE_DIR     = Path(__file__).parent.resolve()
# DATA_PATH    = BASE_DIR / "data"
# PARTITION_CSV = BASE_DIR / "partition.csv"

# partition_df = pd.read_csv(PARTITION_CSV)
# DEV_IDS      = partition_df[partition_df["Proposal"] == "devel"]["Id"].tolist()

# # ---------- import helpers from training script ----------
# from train_helix import load_data_for_id, create_feature_embeddings  # noqa: E402
# from models.helix_transformer import HelixTransformer               # noqa: E402


# # ---------- inference helper ----------
# @torch.no_grad()
# def infer_window_probs(model, window_list, device):
#     """Return a list of scalar probabilities (sigmoid‑mean over window)."""
#     probs = []
#     for w in window_list:
#         inp = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,F)
#         logits, _ = model(inp)            # (1,T,1)
#         p = torch.sigmoid(logits).mean().item()
#         probs.append(p)
#     return probs


# # ---------- main ----------

# def main():
#     parser = argparse.ArgumentParser("Validate Helix‑Transformer")
#     parser.add_argument("--model_path", required=True)
#     parser.add_argument("--d_model", type=int, default=192)
#     parser.add_argument("--n_heads", type=int, default=4)
#     parser.add_argument("--n_layers", type=int, default=2)
#     parser.add_argument("--threshold", type=float, default=0.5,
#                         help="probability threshold for positive class")
#     args = parser.parse_args()

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # build model skeleton
#     model = HelixTransformer(
#         feat_dim=11,
#         d_model=args.d_model,
#         n_heads=args.n_heads,
#         n_layers=args.n_layers,
#         dropout=0.1,
#         max_len=1024,
#     ).to(device)
#     state = torch.load(args.model_path, map_location=device)
#     model.load_state_dict(state)
#     model.eval()

#     y_true, y_prob = [], []

#     for file_id in DEV_IDS:
#         data = load_data_for_id(file_id)
#         if data is None:
#             continue
#         # we only need windows + window_labels
#         _, _, windows, window_labels = create_feature_embeddings(data, window_size_ms=1000)
#         # infer probabilities
#         probs = infer_window_probs(model, windows, device)
#         y_prob.extend(probs)
#         y_true.extend([int(lbl[0]) for lbl in window_labels])

#     y_prob_arr = np.array(y_prob)
#     y_true_arr = np.array(y_true)
#     y_pred = (y_prob_arr >= args.threshold).astype(int)

#     # metrics
#     f1 = f1_score(y_true_arr, y_pred)
#     prec, rec, _, _ = precision_recall_fscore_support(
#         y_true_arr, y_pred, average="binary", zero_division=0
#     )
#     auc = roc_auc_score(y_true_arr, y_prob_arr)

#     report = {
#         "model_path": str(args.model_path),
#         "threshold": args.threshold,
#         "F1": round(f1, 4),
#         "Precision": round(prec, 4),
#         "Recall": round(rec, 4),
#         "AUROC": round(auc, 4),
#     }
#     print(json.dumps(report, indent=2))

#     # optionally save to a csv log
#     out_csv = BASE_DIR / "helix_dev_metrics.csv"
#     row = {"model": Path(args.model_path).name, **report}
#     df = pd.DataFrame([row])
#     if out_csv.exists():
#         df.to_csv(out_csv, mode="a", header=False, index=False)
#     else:
#         df.to_csv(out_csv, index=False)
#     print(f"Metrics appended to {out_csv}")


# if __name__ == "__main__":
#     main()













"""validate_helix.py  –  Compare Helix‑Transformer *vs* Random‑Forest baseline (window‑level)
==========================================================================
Run example
-----------
```bash
python validate_helix.py \
  --model_path ./transformer_outputs/helix_transformer_model.pth \
  --d_model 192 --n_heads 4 --threshold 0.5 \
  --rf_model_path ./models/global_random_forest_model.joblib
```
The script prints **F1 / Precision / Recall / AUROC** for both models, saves the
numbers to *helix_dev_metrics.csv* **and** generates a quick bar‑plot (saved as
*compare_helix_rf.png*).
"""
# ------------------------------------------------------------
# Imports & CLI
# ------------------------------------------------------------
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from joblib import load as jl_load
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR / "data"
PARTITION_CSV = BASE_DIR / "partition.csv"

partition_df = pd.read_csv(PARTITION_CSV)
DEV_IDS = partition_df[partition_df["Proposal"] == "devel"]["Id"].tolist()

def load_scaler(path):
    if path and os.path.exists(path):
        return jl_load(path)
    raise FileNotFoundError(f"Cannot find scaler at {path}")



# helpers reused from training
# from train_helix import load_data_for_id, create_feature_embeddings  # noqa: E402
from models.helix_transformer import HelixTransformer               # noqa: E402
from whole import TimeSeriesTransformer      
try:
    from whole import TimeSeriesTransformer
except ImportError:
    TimeSeriesTransformer = None

from train_helix import load_data_for_id                             # noqa: E402

@torch.no_grad()
def infer_window_prob_helix(model, window_np, device):
    inp = torch.tensor(window_np, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model(inp)  # (1, T, 1)
    return torch.sigmoid(logits).mean().item()


def infer_rf_label(rf_model, window_df):
    frame_vecs = window_df[[str(i) for i in range(88)]].to_numpy()
    frame_preds = rf_model.predict(frame_vecs)
    # 50% 以上帧为 1 则窗口为 1
    return int(frame_preds.mean() > 0.5)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Validate Helix & RF baseline (window‑level)")
    p.add_argument("--model_path", required=True)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=2)
    # p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=None,
                   help="single threshold; if None will scan [0.3-0.6]")
    p.add_argument("--rf_model_path", default=None, help="path to RF .joblib baseline")
    p.add_argument("--baseline_model_path", default=None,
               help="path to transformer_model.pth baseline")
    p.add_argument("--scaler_path", default=None,
                   help="(optional) explicit scaler path; otherwise "
                        "auto-detect transformer_outputs/scaler_<dim>d.pkl")
    p.add_argument("--baseline_scaler_path",
                default="./transformer_outputs/baseline_scaler.pkl")
    p.add_argument("--ege_idx_list", type=str, default="",
                   help="Comma-sep indices kept during training, e.g. 4,6,8")
    args = p.parse_args()
    keep_ege = ([int(t) for t in args.ege_idx_list.split(',') if t.strip()]
                if args.ege_idx_list else list(range(88)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # scaler = load_scaler(args.scaler_path)
    # base_scaler = load_scaler(args.baseline_scaler_path)
    # helix_scaler = load_scaler(args.scaler_path)                   
    ckpt      = torch.load(args.model_path, map_location="cpu")
    feat_dim  = ckpt["embed.weight"].shape[1]        # 11 / 26 / 33 / 99 …
    default_sc = BASE_DIR / f"transformer_outputs/scaler_{feat_dim}d.pkl"
    sc_path    = args.scaler_path or default_sc
    helix_scaler = load_scaler(sc_path)
    if helix_scaler.n_features_in_ != feat_dim:
        raise ValueError(
          f"  模型 {feat_dim}d, 但 {sc_path.name} 里是"
          f" {helix_scaler.n_features_in_}d —— scaler 写错路径或被覆盖！")

    base_scaler  = load_scaler(args.baseline_scaler_path) if args.baseline_model_path else None
    # ---- Helix skeleton + weights ----
    # helix = HelixTransformer(11, args.d_model, args.n_heads, args.n_layers, 0.1, 1024).to(device)
    # helix.load_state_dict(torch.load(args.model_path, map_location=device))



#temp jump

    # helix = HelixTransformer(99, args.d_model, args.n_heads, args.n_layers, 0.05, 1024).to(device)
    # # 允许跳过旧 checkpoint 里多出来的 match.gamma 参数
    # ckpt = torch.load(args.model_path, map_location=device)
    # helix.load_state_dict(ckpt, strict=False)  

    # # ➋ 读取真正的输入维度 (embed.in_features)
    # real_dim = helix.embed.in_features
    # assert real_dim in (11, 99), \
    #     f"Unexpected input dim {real_dim}, only 11 / 99 supported"
    # use_egemaps = real_dim == 99


    helix     = HelixTransformer(feat_dim, args.d_model,
                                   args.n_heads, args.n_layers,
                                   0.05, 1024).to(device)

    helix.load_state_dict(ckpt, strict=False)
    use_egemaps = feat_dim > 4 
 
    helix.eval()
    print("ignored:", [k for k in ckpt if k not in helix.state_dict()])

    if args.threshold is None:
        th_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    else:
        th_grid = [args.threshold]

    # ---- RF baseline (optional) ----
    rf_model = jl_load(args.rf_model_path) if args.rf_model_path else None



    # ---- baseline Transformer (optional) ----
    # if args.baseline_model_path:
    if args.baseline_model_path and not use_egemaps:
        base_tf = TimeSeriesTransformer(
            11, args.d_model, args.n_heads, args.n_layers, 0.1, 1024
        ).to(device)
        base_tf.load_state_dict(torch.load(args.baseline_model_path,
                                           map_location=device))
        base_tf.eval()
    else:
        base_tf = None
    # ---- iterate dev set ----
    y_true, h_prob = [], []
    rf_pred, b_prob = [], []


    for fid in DEV_IDS:
        df = load_data_for_id(fid)
        if df is None:
            continue

#消融对齐用
        # for col in ("trustworthiness", "arousal", "valence"):
        #     df[f"{col}_diff"]  = df[col].diff().fillna(0)
        #     df[f"{col}_diff2"] = df[f"{col}_diff"].diff().fillna(0)
       # ---------------------------------------------------------
        # ➊ 派生特征，与 train_helix.py 完全一致 ------------------
        # for col in ("trustworthiness", "arousal", "valence"):
        #     if col == "trustworthiness":          # <- 保持训练脚本里的列名
        #         df["trust_diff"]  = df[col].diff().fillna(0)
        #         df["trust_diff2"] = df["trust_diff"].diff().fillna(0)
        #     else:
        #         df[f"{col}_diff"]  = df[col].diff().fillna(0)
        #         df[f"{col}_diff2"] = df[f"{col}_diff"].diff().fillna(0)
        alpha, beta = 1, 0.5
        df["VA_adjusted"]      = np.sqrt(alpha*df["valence"]**2 + beta*df["arousal"]**2)
        # df["VA_adjusted_diff"] = df["VA_adjusted"].diff().fillna(0)
        # df["trust_VA_ratio"]   = (df["trust_diff"].abs()
        #                           / (df["VA_adjusted_diff"].abs() + 1e-10))
        # df["log_trust_VA"]     = np.log1p(df["trust_VA_ratio"])
        # ---------------------------------------------------------


        # ➋ 根据 real_dim 选择 11 or 99 维 feats --------------------
        # if use_egemaps:                       # 99 维
        #     core_cols = [
        #         "trustworthiness","arousal","valence",
        #         "trust_diff","arousal_diff","valence_diff",
        #         "trust_diff2","arousal_diff2","valence_diff2",
        #         "VA_adjusted","log_trust_VA"
        #     ]
        core_cols = ["trustworthiness", "arousal", "valence", "VA_adjusted"]

        if use_egemaps:
            # egemap_cols = [str(i) for i in range(88)]
            egemap_cols = [str(i) for i in keep_ege] 
            feats = df[core_cols + egemap_cols].to_numpy()
        else:                                 # 11 维
            # feats = df[[
            #     "trustworthiness","arousal","valence",
            #     "trust_diff","arousal_diff","valence_diff",
            #     "trust_diff2","arousal_diff2","valence_diff2",
            #     "VA_adjusted","log_trust_VA"
            # ]].to_numpy()
            feats = df[core_cols].to_numpy()

   #消融前的版本
        # _, _, windows, window_labels = create_feature_embeddings(df, window_size_ms=1000)

        # 划 1 s 窗口
        win_len = int(1000 / 250)                    # 4 帧
        windows = [feats[i:i+win_len]
                   for i in range(0, len(feats) - win_len + 1, win_len)]
        # label 沿用原先斜率规则
        window_labels = [(1 if (np.polyfit(
                            df['timestamp'].values[s:s+win_len].astype(float),
                            df['valence'].values[s:s+win_len] ,1)[0]  < 0 and
                           np.polyfit(
                            df['timestamp'].values[s:s+win_len].astype(float),
                            df['arousal'].values[s:s+win_len] ,1)[0]  > 0 and
                           np.polyfit(
                            df['timestamp'].values[s:s+win_len].astype(float),
                            df['trustworthiness'].values[s:s+win_len] ,1)[0]  < 0)
                        else 0)  for s in range(0, len(feats) - win_len + 1, win_len)]

        
        windows_h = [helix_scaler.transform(w) for w in windows]    # Helix 版本
        # windows_b = [base_scaler.transform(w)  for w in windows] if base_scaler else None
        if base_scaler is not None and not use_egemaps:
            windows_b = [base_scaler.transform(w) for w in windows]
        else:
            windows_b = None

        starts = range(0, len(df) - len(windows[0]) + 1, len(windows[0]))

        for idx,(lbl,beg) in enumerate(zip(window_labels, starts)):
            # ------ Helix ------
            w_np_h = windows_h[idx]
            h_prob.append(infer_window_prob_helix(helix, w_np_h, device))

            # baseline prob
            if base_tf is not None and not use_egemaps:
                # b_prob.append(
                #     torch.sigmoid(base_tf(
                #         torch.tensor(w_np, dtype=torch.float32,
                #                      device=device).unsqueeze(0)
                #     )).mean().item()
                # )

                w_np_b = windows_b[idx]
                b_prob.append(
                    torch.sigmoid(
                       base_tf(torch.tensor(w_np_b,
                                             dtype=torch.float32,
                                             device=device
                                    ).unsqueeze(0))
                    ).mean().item()
                )
            # RF label (if provided)

            if rf_model is not None:
                win_df = df.iloc[beg : beg + len(windows[0])]

                # win_df = df.iloc[beg : beg + len(w_np)]
                rf_pred.append(infer_rf_label(rf_model, win_df))
            # y_true.append(int(lbl[0]))
            y_true.append(int(lbl if np.isscalar(lbl) else lbl[0]))
    y_true = np.array(y_true)

    # # ---- Helix metrics ----
    # h_pred = (np.array(h_prob) >= args.threshold).astype(int)
    # h_f1 = f1_score(y_true, h_pred)
    # h_p, h_r, _, _ = precision_recall_fscore_support(y_true, h_pred, average="binary")
    # h_auc = roc_auc_score(y_true, np.array(h_prob))

    best_row = None
    for TH in th_grid:
        h_pred = (np.array(h_prob) >= TH).astype(int)
        h_f1   = f1_score(y_true, h_pred)
        h_p, h_r, _, _ = precision_recall_fscore_support(
            y_true, h_pred, average="binary", zero_division=0)
        h_auc  = roc_auc_score(y_true, np.array(h_prob))

        row = {"thr": TH, "F1": h_f1, "P": h_p, "R": h_r, "AUC": h_auc}
        if best_row is None or row["F1"] > best_row["F1"]:
            best_row = row

    # 用最佳阈值汇报
    h_f1, h_p, h_r, h_auc = best_row["F1"], best_row["P"], best_row["R"], best_row["AUC"]
    args.threshold = best_row["thr"]           # 之后打印 / 保存用

    # ---- RF metrics ----
    if rf_model is not None:
        rf_pred = np.array(rf_pred)
        rf_f1 = f1_score(y_true, rf_pred)
        rf_p, rf_r, _, _ = precision_recall_fscore_support(y_true, rf_pred, average="binary")
        # RF 只有 hard label，没有概率；AUROC 用 nan
        rf_auc = float("nan")

    # ---- baseline metrics ----
    if base_tf is not None:
        b_pred = (np.array(b_prob) >= args.threshold).astype(int)
        b_f1 = f1_score(y_true, b_pred)
        b_p, b_r, _, _ = precision_recall_fscore_support(y_true, b_pred,
                                                        average="binary")
        b_auc = roc_auc_score(y_true, np.array(b_prob))

    # ---- print + save ----
    # helix_report = {"model":"helix","F1":round(h_f1,4),"P":round(h_p,4),"R":round(h_r,4),"AUROC":round(h_auc,4)}
    helix_report = {"model":"helix",
                    "thr":round(args.threshold,2),
                    "F1":round(h_f1,4),
                    "P":round(h_p,4),
                    "R":round(h_r,4),
                    "AUROC":round(h_auc,4)}

    print("Helix:", json.dumps(helix_report))
    if rf_model is not None:
        rf_report = {"model":"rf","F1":round(rf_f1,4),"P":round(rf_p,4),"R":round(rf_r,4),"AUROC":rf_auc}
        print("RF:", json.dumps(rf_report))

    if base_tf is not None:
        base_report = {"model":"baseline_tf","F1":round(b_f1,4),
                       "P":round(b_p,4),"R":round(b_r,4),"AUROC":round(b_auc,4)}
        print("Baseline-TF:", json.dumps(base_report))

    # append to csv
    out_csv = BASE_DIR / "helix_dev_metrics.csv"
    rows = [{"run":"helix", **helix_report}]
    if rf_model is not None:
        rows.append({"run":"rf", **rf_report})

    if base_tf is not None:
        rows.append({"run":"baseline_tf", **base_report})

    pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)


    # ---- Quick bar‑plot ----
    if rf_model is not None or base_tf is not None:
        labels = ["Precision","Recall","F1"]
        helix_vals = [h_p, h_r, h_f1]
        # rf_vals    = [rf_p, rf_r, rf_f1]
        bars = [helix_vals]
        lab_names = ["Helix"]
        if base_tf is not None:
            bars.append([b_p, b_r, b_f1]); lab_names.append("Baseline-TF")
        if rf_model is not None:
            bars.append([rf_p, rf_r, rf_f1]); lab_names.append("RF")
        x = np.arange(len(labels))
        width = 0.25
        plt.figure(figsize=(6,4))
        # plt.bar(x-width/2, helix_vals, width, label="Helix")
        # plt.bar(x+width/2, rf_vals,    width, label="RF Baseline")
        
        for i,vals in enumerate(bars):
            plt.bar(x + (i-len(bars)/2)*width, vals, width, label=lab_names[i])

        plt.xticks(x, labels)
        plt.ylim(0,1)
        plt.ylabel("score")
        plt.title("Helix vs Baselines winndow=1s)")
        plt.legend()
        plt.tight_layout()
        plot_path = BASE_DIR / "compare_helix_rf.png"
        plt.savefig(plot_path)
        print(f"Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    main()
