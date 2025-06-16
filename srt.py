import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.metrics import make_scorer, accuracy_score, recall_score
from joblib import dump
from sklearn.model_selection import cross_val_score



warnings.filterwarnings("ignore")

base_path = "./c3_muse_trust"  # c3_muse_trust 文件夹的相对路径
data_path = "./data"           # data 文件夹的相对路径
partition_file = "./partition.csv"  # partition.csv 文件的相对路径label_paths = {
label_paths = {
    "valence": os.path.join(data_path, "label_segments", "valence"),
    "arousal": os.path.join(data_path, "label_segments", "arousal"),
    "trustworthiness": os.path.join(data_path, "label_segments", "trustworthiness")
}
egemaps_path = os.path.join(data_path, "egemaps","egemaps")  # egemaps 的相对路径
partition_df = pd.read_csv(partition_file)
test_ids = partition_df[partition_df['Proposal'] == 'test']['Id'].tolist()

# 1. load data
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
        combined_df['File_ID'] = file_id  # 确保包含 File_ID 列
        print(f"ID {file_id} - Final Data Shape: {combined_df.shape}")
        return combined_df

    except Exception as e:
        print(f"Error loading data for ID {file_id}: {e}")
        return None
    
def calculate_dynamic_thresholds(data, trust_quantile=0.75, valence_quantile=0.75, arousal_quantile=0.75):
    """
    quaantile for threshold
    """
    trust_thresh = data['trust_diff1'].quantile(trust_quantile)
    valence_thresh = data['valence_diff1'].quantile(valence_quantile)
    arousal_thresh = data['arousal_diff1'].quantile(arousal_quantile)
    return trust_thresh, valence_thresh, arousal_thresh


# 2. dynamic window 
def create_uncanny_labels_with_window(data, window_size_ms=1000, sampling_rate_ms=250,
                                      trust_quantile=0.75, valence_quantile=0.75, arousal_quantile=0.75):
    window_size = max(1, int(window_size_ms / sampling_rate_ms))  

    data['trust_diff1'] = data['trustworthiness'].diff().fillna(0)
    data['valence_diff1'] = data['valence'].diff().fillna(0)
    data['arousal_diff1'] = data['arousal'].diff().fillna(0)

    # 修改：传入分位数
    trust_thresh, valence_thresh, arousal_thresh = calculate_dynamic_thresholds(
        data, 
        trust_quantile=trust_quantile, 
        valence_quantile=valence_quantile, 
        arousal_quantile=arousal_quantile
    )

    trust_std = data['trust_diff1'].rolling(window=window_size).std().fillna(0)
    valence_std = data['valence_diff1'].rolling(window=window_size).std().fillna(0)
    arousal_std = data['arousal_diff1'].rolling(window=window_size).std().fillna(0)

    data['uncanny_label'] = ((trust_std > trust_thresh) & 
                             (valence_std > valence_thresh) & 
                             (arousal_std > arousal_thresh)).astype(int)

    print(f"Uncanny Label Distribution:\n{data['uncanny_label'].value_counts()}")
    return data




# 3. imp
# ... existing code ...

def calculate_feature_importance(data):
    drop_cols = ['timestamp', 'valence', 'arousal', 'trustworthiness', 'uncanny_label', 
                 'trust_diff1', 'valence_diff1', 'arousal_diff1']
    X = data.drop(columns=[col for col in drop_cols if col in data.columns])
    X = X[[str(i) for i in range(88)]]
    y = data['uncanny_label']

    if y.sum() < 5:
        print("正样本数量不足")
        return None

    weights = {0: 1, 1: max(1, int(len(y) / (2 * y.sum())))}  
    model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight=weights)
    model.fit(X, y)

    # 保存模型
    model_file = f"random_forest_model_{data['File_ID'].iloc[0]}.joblib"
    dump(model, model_file)
    print(f"模型已保存为 {model_file}")

    # 添加File_ID列到特征重要性DataFrame
    importance_df = pd.DataFrame({
        'File_ID': data['File_ID'].iloc[0],
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df

def evaluate_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

    scoring = {
        'f1': 'f1',
        'accuracy': make_scorer(accuracy_score),
        'recall': make_scorer(recall_score)
    }

    scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    recall_scores = cross_val_score(model, X, y, cv=5, scoring='recall')

    print(f"Cross-Validation F1 Scores: {scores}")
    print(f"Cross-Validation Accuracy Scores: {accuracy_scores}")
    print(f"Cross-Validation Recall Scores: {recall_scores}")

    print(f"Average F1 Score: {np.mean(scores):.4f}")
    print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"Average Recall: {np.mean(recall_scores):.4f}")


def find_optimal_parameters(data):
    """
    网格搜索最优参数
    """
    quantiles = [0.65, 0.70, 0.75, 0.80, 0.85]
    window_sizes = [500, 750, 1000, 1250, 1500]  # 毫秒
    n_estimators_range = [50, 100, 150, 200, 250]
    
    best_score = -float('inf')
    best_params = {}
    
    for quantile in quantiles:
        for window_size in window_sizes:
            # 使用当前窗口大小和分位数创建标签
            temp_data = create_uncanny_labels_with_window(
                data.copy(), 
                window_size_ms=window_size,
                sampling_rate_ms=250,
                trust_quantile=quantile,
                valence_quantile=quantile,
                arousal_quantile=quantile
            )
            
            if temp_data['uncanny_label'].sum() < 5:
                continue
                
            # 准备特征和标签
            drop_cols = ['timestamp', 'valence', 'arousal', 'trustworthiness', 'uncanny_label',
                        'trust_diff1', 'valence_diff1', 'arousal_diff1']
            X = temp_data.drop(columns=[col for col in drop_cols if col in temp_data.columns])
            X = X[[str(i) for i in range(88)]]
            y = temp_data['uncanny_label']
            
            # 寻找最优n_estimators
            for n_estimators in n_estimators_range:
                weights = {0: 1, 1: max(1, int(len(y) / (2 * y.sum())))}
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    class_weight=weights,
                    oob_score=True  # 启用OOB评分
                )
                rf.fit(X, y)
                
                current_score = rf.oob_score_
                
                if current_score > best_score:
                    best_score = current_score
                    best_params = {
                        'quantile': quantile,
                        'window_size': window_size,
                        'n_estimators': n_estimators,
                        'oob_score': current_score
                    }
                    best_model = rf
    
    print(f"最优参数: {best_params}")
    return best_params, best_model



# 主循环
def main():
    all_feature_importances = []
    all_data = {}  # 新增：存储所有有效的数据
    best_window_size = 1000

    for file_name in os.listdir(label_paths["trustworthiness"]):
        file_id = int(file_name.split(".")[0])
        print(f"\n处理ID: {file_id}")
        data = load_data_for_id(file_id)

        if data is not None:
            data = create_uncanny_labels_with_window(data, window_size_ms=1000, sampling_rate_ms=250)

            if data['uncanny_label'].sum() >= 5:
                feature_importance_df = calculate_feature_importance(data)
                if feature_importance_df is not None:
                    all_feature_importances.append(feature_importance_df)
                    all_data[file_id] = data  # 新增：保存有效的数据
                    print(f"ID {file_id} 的重要特征:\n{feature_importance_df.head()}")
            else:
                print(f"ID {file_id} 的正样本数量不足，跳过")
        else:
            print(f"ID {file_id} 没有有效数据")

    # 保存结果
    if all_feature_importances:
        final_importance_df = pd.concat(all_feature_importances)
        
        # 透视表格式，更容易查看每个模型的特征重要性
        pivot_importance_df = final_importance_df.pivot(
            index='Feature',
            columns='File_ID',
            values='Importance'
        )
        
        # 添加平均重要性列
        pivot_importance_df['Mean_Importance'] = pivot_importance_df.mean(axis=1)
        pivot_importance_df = pivot_importance_df.sort_values('Mean_Importance', ascending=False)
        
        # 保存两种格式
        final_importance_df.to_csv("feature_importances_raw.csv", index=False)
        pivot_importance_df.to_csv("feature_importances_by_model.csv")
        
        print("\n=== 开始优化参数 ===")
        # 对每个有效的数据集进行参数优化
        all_best_params = []
        for file_id, data in all_data.items():
            print(f"\n优化 ID {file_id} 的参数")
            best_params, best_model = find_optimal_parameters(data)
            best_params['File_ID'] = file_id
            all_best_params.append(best_params)
        
        # 保存所有优化结果
        params_df = pd.DataFrame(all_best_params)
        params_df.to_csv("optimal_parameters.csv", index=False)
        
        print("\n结果已保存到:")
        print("1. feature_importances_raw.csv (原始格式)")
        print("2. feature_importances_by_model.csv (透视表格式)")
        print("3. optimal_parameters.csv (优化参数)")
    else:
        print("没有有效结果可保存")

    print("\n=== 程序执行完成 ===")

if __name__ == "__main__":
    main()