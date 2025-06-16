import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
all_processed_data = [] 
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
# from torch.optim.lr_scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR as CosineLRScheduler
from joblib import dump
import warnings
import time
import argparse

warnings.filterwarnings("ignore")

# 数据路径设置（可通过命令行参数调整）
base_path = "./c3_muse_trust"
data_path = "./data"
partition_file = "./partition.csv"

label_paths = {
    "valence": os.path.join(data_path, "label_segments", "valence"),
    "arousal": os.path.join(data_path, "label_segments", "arousal"),
    "trustworthiness": os.path.join(data_path, "label_segments", "trustworthiness")
}

egemaps_path = os.path.join(data_path, "egemaps", "egemaps")
partition_df = pd.read_csv(partition_file)
TRAIN_IDS = partition_df[partition_df['Proposal'] == 'train']['Id'].tolist()
DEV_IDS   = partition_df[partition_df['Proposal'] == 'devel']['Id'].tolist()
test_ids = partition_df[partition_df['Proposal'] == 'test']['Id'].tolist()

# 用于保存输出
output_dir = "./baseline_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 数据加载函数
def load_data_for_id(file_id):
    if file_id in test_ids:
        return None
    try:
        valence_df = pd.read_csv(os.path.join(label_paths["valence"], f"{file_id}.csv"))
        arousal_df = pd.read_csv(os.path.join(label_paths["arousal"], f"{file_id}.csv"))
        trust_df = pd.read_csv(os.path.join(label_paths["trustworthiness"], f"{file_id}.csv"))

        valence_df = valence_df[['timestamp', 'value']].rename(columns={'value': 'valence'})
        arousal_df = arousal_df[['timestamp', 'value']].rename(columns={'value': 'arousal'})
        trust_df = trust_df[['timestamp', 'value']].rename(columns={'value': 'trustworthiness'})

        labels_df = valence_df.merge(arousal_df, on='timestamp', how='outer')
        labels_df = labels_df.merge(trust_df, on='timestamp', how='outer').sort_values(by='timestamp')
        labels_df = labels_df.interpolate(method='linear')

        egemaps_df = pd.read_csv(os.path.join(egemaps_path, f"{file_id}.csv"))
        egemaps_columns = ['timestamp'] + [str(i) for i in range(88)]
        egemaps_df = egemaps_df[egemaps_columns]

        combined_df = labels_df.merge(egemaps_df, on='timestamp', how='inner')
        combined_df['File_ID'] = file_id
        return combined_df
    except Exception as e:
        print(f"Error loading data for ID {file_id}: {e}")
        return None

# -----------------------------
# 特征嵌入函数
def create_feature_embeddings(data, window_size_ms=1000, sampling_rate_ms=250):
    window_size = max(1, int(window_size_ms / sampling_rate_ms))
    
    # # 计算一阶和二阶导数（如果作为其他特征仍需保留）
    # data['trust_diff'] = data['trustworthiness'].diff().fillna(0)
    # data['arousal_diff'] = data['arousal'].diff().fillna(0)
    # data['valence_diff'] = data['valence'].diff().fillna(0)
    # data['trust_diff2'] = data['trust_diff'].diff().fillna(0)
    # data['arousal_diff2'] = data['arousal_diff'].diff().fillna(0)
    # data['valence_diff2'] = data['valence_diff'].diff().fillna(0)
    
    # 如果你还需要其他特征，可以继续计算（例如 VA_adjusted 等）
    alpha, beta = 1, 0.5
    data['VA_adjusted'] = np.sqrt(alpha * data['valence']**2 + beta * data['arousal']**2)
    # data['VA_adjusted_diff'] = data['VA_adjusted'].diff().fillna(0)
    # data['trust_VA_ratio'] = np.abs(data['trust_diff']) / (np.abs(data['VA_adjusted_diff']) + 1e-10)
    # data['log_trust_VA'] = np.log1p(data['trust_VA_ratio'])
    
    # 构造特征向量（token），此处你可以保留原有的特征
    # feature_columns = [
    #     'trustworthiness', 'arousal', 'valence',
    #     'trust_diff', 'arousal_diff', 'valence_diff',
    #     'trust_diff2', 'arousal_diff2', 'valence_diff2',
    #     'VA_adjusted', 'log_trust_VA'
    # ]

    feature_columns = ['trustworthiness', 'arousal', 'valence', 'VA_adjusted']


    # # tokens = data[feature_columns].values
    # tokens = scaler.transform(tokens)
    tokens = data[feature_columns].values
    # tokens= std_scaler.transform(tokens)
    # adding rule
    # scaler = StandardScaler()
    # all_processed_data = []  
    # # tokens = scaler.fit_transform(tokens)
    # all_tokens = np.vstack([tokens for d in all_processed_data for tokens in d["windows"]])
    # scaler.fit(all_tokens)

    # 按时间窗划分 tokens
    n_samples = len(data)
    windows = [tokens[i:i+window_size] for i in range(0, n_samples - window_size + 1, window_size)]
    
    # 利用线性回归计算每个时间窗内的趋势，并生成窗口标签
    window_labels = []
    for start in range(0, n_samples - window_size + 1, window_size):
        window_data = data.iloc[start:start+window_size]
        # 转换 timestamp 为 float 类型（确保时间可以进行回归计算）
        x = window_data['timestamp'].astype(float).values
        # 计算各指标的斜率（使用 np.polyfit 拟合一次直线，取斜率）
        slope_valence = np.polyfit(x, window_data['valence'].values, 1)[0]
        slope_arousal = np.polyfit(x, window_data['arousal'].values, 1)[0]
        slope_trust = np.polyfit(x, window_data['trustworthiness'].values, 1)[0]
        # 根据斜率判断整体趋势：valence 下降、arousal 上升、trustworthiness 下降
        label = 1 if (slope_valence < 0 and slope_arousal > 0 and slope_trust < 0) else 0
        # 将该窗口的标签复制到窗口内所有时间步（保持与 transformer 输出一致）
        window_labels.append(np.full(window_size, label))
        
    # 如果需要单个采样点的标签，可以根据需要设计，此处返回 None
    sample_labels = None
    
    return tokens, sample_labels, windows, window_labels

# -----------------------------
# Focal Loss 实现（可选）
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# -----------------------------
# Transformer 模型定义
class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1, max_length=1024):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoding = self.create_positional_encoding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, 1)
    
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x  # 返回 logits，由损失函数处理 sigmoid

# -----------------------------
# 训练函数
def train_transformer(args):
    # all_processed_data = []
    start_time = time.time()
    
    # 数据加载和处理
    for file_name in os.listdir(label_paths["trustworthiness"]):
        file_id = int(file_name.split(".")[0])
        if file_id not in TRAIN_IDS:
            continue 
        print(f"Processing file ID: {file_id}")

        file_start = time.time()
        data = load_data_for_id(file_id)
        if data is not None:
            tokens, labels, windows, window_labels = create_feature_embeddings(data, window_size_ms=args.window_size)
            all_processed_data.append({
                'file_id': file_id,
                'windows': windows,
                'window_labels': window_labels
            })
        print(f"Finished file ID {file_id} in {time.time() - file_start:.2f} seconds")
    print(f"Data processing completed in {time.time() - start_time:.2f} seconds")
    if all_processed_data:
        all_tokens = np.vstack([w for d in all_processed_data for w in d["windows"]])
        std_scaler.fit(all_tokens)                      # 先整体 fit
        # 再把所有窗口做一次 transform（就地改写）  
        for d in all_processed_data:
            d["windows"] = [std_scaler.transform(w) for w in d["windows"]]

# ---------- 追加保存 ----------
        dump(std_scaler,
             os.path.join(output_dir, "baseline_scaler.pkl"))

    if not all_processed_data:
        print("No valid data available for training. Exiting.")
        return
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        feature_dim=4, 
        d_model=args.d_model, 
        n_heads=args.n_heads, 
        n_layers=args.n_layers, 
        dropout=args.dropout, 
        max_length=args.max_length
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 损失函数选择
    if args.loss_type == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
    
    if args.lr_scheduler_type:
        total_steps = args.num_train_epochs * sum(len(data['windows']) for data in all_processed_data) // args.gradient_accumulation_steps
        # scheduler = CosineLRScheduler(
        #     optimizer, t_initial=total_steps, lr_min=1e-6, warmup_t=int(args.warmup_ratio * total_steps)
        # )
        scheduler = CosineLRScheduler(
            optimizer, T_max=total_steps, eta_min=1e-6)
    amp_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # 训练循环
    model.train()
    for epoch in range(args.num_train_epochs):
        epoch_loss = 0
        step = 0
        optimizer.zero_grad()
        for data in all_processed_data:
            for window, window_label in zip(data['windows'], data['window_labels']):
                if len(window) == 0:
                    continue
                inputs = torch.tensor(window, dtype=torch.float32).to(device)
                labels = torch.tensor(window_label, dtype=torch.float32).unsqueeze(-1).to(device)
                
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs.unsqueeze(0))
                        loss = criterion(outputs, labels.unsqueeze(0)) / args.gradient_accumulation_steps
                    amp_scaler.scale(loss).backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        amp_scaler.step(optimizer)
                        amp_scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(inputs.unsqueeze(0))
                    loss = criterion(outputs, labels.unsqueeze(0)) / args.gradient_accumulation_steps
                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                step += 1
                
                if args.lr_scheduler_type:
                    scheduler.step()
        
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}, Loss: {epoch_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_dir, "transformer_model.pth"))
    print(f"Transformer 模型已保存到 {os.path.join(output_dir, 'transformer_model.pth')}")

# -----------------------------
# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Time Series Transformer for uncanny valley detection")
    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.2, help="Warmup ratio for LR scheduler")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_restarts", choices=[None, "cosine_with_restarts"], help="LR scheduler type")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    # 损失函数参数
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal"], help="Loss function type")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha")
    
    # 模型参数
    parser.add_argument("--d_model", type=int, default=384, help="Transformer hidden size")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    
    # 数据参数
    parser.add_argument("--window_size", type=int, default=1000, help="Window size in ms")
    
    return parser.parse_args()

# -----------------------------
# 主函数
def main():
    args = parse_args()
    train_transformer(args)
    print("终于跑完了！")

if __name__ == "__main__":
    main()

