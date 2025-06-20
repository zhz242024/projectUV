# # save as gather_lora_results.py and run in project root
# # check_feature_sort.py  ―― 把 88 个日志的 “dev …” 行收进 DataFrame
# import ast, re, glob, pandas as pd

# pat = re.compile(r'\[(\d+)] dev ({.*})')   # 例如  [51] dev {'F1': 0.1831, ...}

# rows = []
# for f in glob.glob('logs/lora_1085*.out'):
#     for line in open(f):
#         m = pat.search(line)
#         if m:
#             idx  = int(m.group(1))
#             metr = ast.literal_eval(       # 比 json.loads() 更宽容
#                      m.group(2)
#                      .replace('np.float64','')   # → 0.859
#                      .replace('array(','')      # 万一出现 numpy array(...)
#                      .rstrip(')') )
#             metr['idx'] = idx
#             rows.append(metr)

# df = pd.DataFrame(rows).sort_values('F1', ascending=False)
# print(df.head(15))          # Top-15
# df.to_csv('ege_scan_metrics.csv', index=False)




#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gather_lora_results.py  —  仅保留 F1 高于 baseline 的特征
"""

import re, glob, ast, os, pandas as pd

LOG_GLOB   = "logs/lora_*.out"               # 日志文件匹配
BASE_FILE  = "logs/lora_4846_4294967294.out" # baseline 日志完整路径
OUT_CSV    = "ege_scan_over_baseline.csv"    # 输出

pat = re.compile(r'\[(\-?\d+)] dev ({.*})')
rows = []

def grab_metrics(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            m = pat.search(ln)
            if m:
                idx  = int(m.group(1))
                raw  = (m.group(2)
                        .replace("np.float64","")
                        .replace("array(","")
                        .rstrip(")"))
                metr = ast.literal_eval(raw)
                metr["idx"] = idx
                metr["file"] = os.path.basename(path)
                rows.append(metr)

# baseline 文件先读（确保一定抓到）
grab_metrics(BASE_FILE)

# 其余日志
for fp in glob.glob(LOG_GLOB):
    if fp == BASE_FILE:
        continue
    grab_metrics(fp)

df = pd.DataFrame(rows)

# ―― 找 baseline（idx == -1；若没有则用 BASE_FILE 第1条）
base_row = df.loc[df.idx == -1]
if base_row.empty:
    base_row = df[df.file == os.path.basename(BASE_FILE)].iloc[[0]]
baseline_f1 = base_row["F1"].values[0]
print(f"Baseline F1 = {baseline_f1:.4f}")

# ―― 只保留高于 baseline 的
better = df[df.F1 > baseline_f1].sort_values("F1", ascending=False)
print(f"\n共有 {len(better)} 个特征 F1 超过 baseline：")
print(better[["idx","F1","P","R","AUROC"]].to_string(index=False))

better.to_csv(OUT_CSV, index=False)
print(f"\n✓  已写入 {OUT_CSV}")
