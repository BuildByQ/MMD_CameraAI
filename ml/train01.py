import json
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime

# プロジェクトのルートディレクトリを設定
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'
ML_ROOT = PROJECT_ROOT.parent / 'ml'

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"train_log_{timestamp}.txt")
    
    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, obj):
            for f in self.files: f.write(obj); f.flush()
        def flush(self):
            for f in self.files: f.flush()

    logfile = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = Tee(sys.stderr, logfile)
    print(f"=== Logging to {log_path} ===")

def load_config(config_path="config.json"):
    path = ML_ROOT / config_path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_dynamic_columns(config: dict, df_columns: list):
    """JSON設定から実際のCSVヘッダー名を組み立てる"""
    
    # 特徴量: ボーン名 -> _w_x, _w_y, _w_z
    motion_bones = config.get('motion_features', [])
    motion_feats = [f"{b}{s}" for b in motion_bones for s in ['_w_x', '_w_y', '_w_z']]
    audio_feats = config.get('audio_features', [])
    feature_cols = [f for f in (motion_feats + audio_feats) if f in df_columns]

    # ラベル: base + anchor_definitionsからの自動生成
    base_labels = config.get('labels_features', [])
    anchor_defs = config.get('anchor_definitions', {})
    dynamic_labels = []
    for entry in anchor_defs.values():
        for b in entry.get("bones", []):
            dynamic_labels.extend([f"bin_target_{b}_front", f"bin_target_{b}_back", 
                                   f"bin_target_{b}_focused_strict", f"bin_target_{b}_focused", 
                                   f"bin_target_{b}_focused_loose"])
    
    # 順序を維持して重複排除
    all_label_list = base_labels + dynamic_labels
    seen = set()
    label_cols = [l for l in all_label_list if l in df_columns and not (l in seen or seen.add(l))]

    return feature_cols, label_cols

def load_and_prepare_dataloaders(config: dict):
    data_dir = Path(config['data']['normalized_dir'])
    dfs = []
    for name in ['camera', 'motion', 'audio']:
        p = data_dir / f'normalized_{name}.csv'
        if p.exists():
            df_tmp = pd.read_csv(p)
            
            # frameを整数（または共通のfloat）に統一
            df_tmp['song_id'] = df_tmp['song_id'].astype(str)
            df_tmp['frame'] = df_tmp['frame'].astype(float).astype(int)
            
            dfs.append(df_tmp)
    
    df = dfs[0]
    for d in dfs[1:]:
        df = pd.merge(df, d, on=['song_id', 'frame'], how='inner')

    # 学習対象曲のフィルタリング
    test_songs = config["data"].get("test_songs", [])
    train_songs = config["data"].get("train_songs", [])
    if not train_songs:
        train_songs = [sid for sid in df["song_id"].unique() if sid not in test_songs]
    df = df[df["song_id"].isin(train_songs)].sort_values(["song_id", "frame"]).fillna(0)

    # カラムの解決
    feature_cols, label_cols = get_dynamic_columns(config, df.columns.tolist())
    
    # ウィンドウ生成
    seq_len = config["model"]["seq_len"]
    center = seq_len // 2
    song_ids = df["song_id"].unique().tolist()
    song_to_idx = {sid: i for i, sid in enumerate(song_ids)}
    
    X_list, y_list, s_idx_list = [], [], []

    for sid in song_ids:
        df_s = df[df["song_id"] == sid]
        X_s = df_s[feature_cols].values
        y_s = df_s[label_cols].values
        s_idx = song_to_idx[sid]
        
        # 曲統計量 (mean, std)
        stats = df_s[feature_cols].agg(['mean', 'std']).values.flatten()
        stats_b = np.repeat(stats.reshape(1, -1), seq_len, axis=0)

        for i in range(len(df_s) - seq_len + 1):
            window = np.concatenate([X_s[i:i+seq_len], stats_b], axis=1)
            X_list.append(window)
            y_list.append(y_s[i+center])
            s_idx_list.append(s_idx)

    # Split
    X_all, y_all, s_all = np.array(X_list), np.array(y_list), np.array(s_idx_list)
    tr_idx, val_idx = train_test_split(np.arange(len(X_all)), test_size=0.1, shuffle=True, random_state=42)

    def to_loader(idx, shuffle=True):
        ds = TensorDataset(torch.tensor(X_all[idx], dtype=torch.float32),
                           torch.tensor(y_all[idx], dtype=torch.float32),
                           torch.tensor(s_all[idx], dtype=torch.long))
        return DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=shuffle)
    
    input_dim = X_all.shape[2] 
    return to_loader(tr_idx), to_loader(val_idx, False), len(song_ids), input_dim, len(label_cols), label_cols

class CameraLabelGRUModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_songs, embed_dim=4, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.song_embedding = nn.Embedding(num_songs, embed_dim)
        self.embed_dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(input_size=input_dim + embed_dim, hidden_size=hidden_size, 
                          num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x, s_idx):
        s_emb = F.normalize(self.song_embedding(s_idx), dim=1)
        s_emb = self.embed_dropout(s_emb).unsqueeze(1).repeat(1, x.size(1), 1)
        out, h = self.gru(torch.cat([x, s_emb], dim=2))
        return self.fc(h[-1])
    
def main():
    config = load_config()
    setup_logging(config["output_dir"])
    device = config["training"].get("device", "cpu")

    tr_loader, val_loader, num_songs, in_dim, out_dim, label_names = load_and_prepare_dataloaders(config)
    
    model = CameraLabelGRUModel(input_dim=in_dim, output_dim=out_dim, num_songs=num_songs + 1, 
                                embed_dim=config["model"].get("embed_dim", 4),
                                hidden_size=config["model"]["hidden_size"],
                                num_layers=config["model"]["num_layers"],
                                dropout=config["model"]["dropout"]).to(device)

    # 不均衡データ対策の重み計算
    pos_weight = torch.ones([out_dim]).to(device) # 必要に応じて計算ロジック追加可
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])

    # 学習
    for epoch in range(config["training"]["epochs"]):
        model.train()
        t_loss = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch+1}")
        for Xb, yb, sb in pbar:
            Xb, yb, sb = Xb.to(device), yb.to(device), sb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb, sb), yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # 検証
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for Xv, yv, sv in val_loader:
                v_loss += criterion(model(Xv.to(device), sv.to(device)), yv.to(device)).item()
        
        print(f"Epoch {epoch+1} Summary - Train Loss: {t_loss/len(tr_loader):.6f}, Val Loss: {v_loss/len(val_loader):.6f}")

    torch.save(model.state_dict(), os.path.join(config["output_dir"], "model_final.pth"))
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    main()