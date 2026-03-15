import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.interpolate import interp1d

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data' / 'normalization_params'
PRED01TO02_ROOT = PROJECT_ROOT.parent / 'predict_01to02'
PRED02_ROOT = PROJECT_ROOT.parent / 'predict_02'
MODEL_SAVE_PATH = PRED02_ROOT / 'model02_weights.pth'
CONFIG_PATH = PROJECT_ROOT / 'config.json'

# --- 1. モデル定義 ---
class CameraLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # ターゲットの数に合わせて自動で変わるようにする
        self.fc = nn.Linear(hidden_size * 2, output_size) 

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. 補助損失関数（拡張用プレースホルダー） ---
def combined_loss(pred, target, input_x, col_map, x_cols):
    mse_loss = nn.MSELoss()(pred, target)
    # ここに将来、「物理制約（速度制限など）」を追加
    return mse_loss

# --- 3. データ処理・学習ロジック ---
def load_config():
    if not CONFIG_PATH.exists():
        return {"training": {"batch_size": 16, "epochs": 8, "learning_rate": 0.0005, "weight_decay": 0.00001, "seq_len": 11}}
    with open(CONFIG_PATH, "r", encoding='utf-8') as f:
        return json.load(f)

def load_and_merge_data():
    print("Step 1: 学習用データのロード（正解ラベル付きカメラデータを使用）...")
    df_mot = pd.read_csv(DATA_ROOT / "normalized_motion.csv")
    df_cam = pd.read_csv(DATA_ROOT / "normalized_camera.csv")
    
    # 姿勢データと、「正解ラベルを含んだカメラデータ」をマージ
    merged = pd.merge(df_cam, df_mot, on='frame', how='inner')
    return merged

def prepare_features(df):
    print("Step 2: 特徴量抽出（ワールド座標 _w_x 系の厳選）...")
    # 入力: 演出ラベル(bin_) + 姿勢座標(_w_x/y/z)
    x_cols = [c for c in df.columns if (
        c.startswith('bin_') or # カメラデータ内の正解ラベル
        (any(c.endswith(s) for s in ['_w_x', '_w_y', '_w_z'])) # 姿勢
    )]
    
    # 教師: カメラ8パラメータ（distanceを追加）
    y_cols = [
        'pos_x', 'pos_y', 'pos_z', 
        'distance', 'fov',
        'rot_x_speed', 'rot_x_sin', 'rot_x_cos',
        'rot_y_speed', 'rot_y_sin', 'rot_y_cos',
        'rot_z_speed', 'rot_z_sin', 'rot_z_cos'
    ]

    X = df[x_cols].values.astype(np.float32)
    y = df[y_cols].values.astype(np.float32)
    col_map = {name: i for i, name in enumerate(x_cols)}
    print(f"DEBUG_X_COLS = {x_cols}")
    return X, y, col_map, x_cols, y_cols

def create_dataloader(X, y, df, seq_len=11, batch_size=16):
    print(f"Step 3: 案A（{seq_len}fスライド窓＋短いカット救済）でデータ作成...")
    all_x, all_y = [], []
    cut_indices = [0] + df.index[df['event_cut'] == 1].tolist() + [len(df)]

    for i in range(len(cut_indices)-1):
        s, e = cut_indices[i], cut_indices[i+1]
        sec_x, sec_y = X[s:e], y[s:e]
        curr_len = len(sec_x)
        if curr_len < 2: continue

        if curr_len >= seq_len:
            # スライド窓
            for j in range(curr_len - seq_len + 1):
                all_x.append(sec_x[j : j + seq_len])
                all_y.append(sec_y[j + seq_len - 1])
        else:
            # 救済：リサンプリングで seq_len に引き延ばす
            x_idx = np.linspace(0, curr_len - 1, seq_len)
            f_x = interp1d(np.arange(curr_len), sec_x, axis=0, kind='linear', fill_value="extrapolate")
            all_x.append(f_x(x_idx).astype(np.float32))
            all_y.append(sec_y[-1])

    return DataLoader(TensorDataset(torch.tensor(np.array(all_x)), torch.tensor(np.array(all_y))), 
                      batch_size=batch_size, shuffle=True)

# --- 4. メイン実行 ---
def main():
    config = load_config()
    tc = config["training2"]
    seq_len = tc.get("seq_len", 11)
    
    df = load_and_merge_data()
    X, y, col_map, x_cols, y_cols = prepare_features(df)
    train_loader = create_dataloader(X, y, df, seq_len=seq_len, batch_size=tc["batch_size"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CameraLSTM(input_size=len(x_cols), output_size=len(y_cols))
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    
    print(f"学習開始 (Epochs: {tc['epochs']})...")
    for epoch in range(tc["epochs"]):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = combined_loss(out, by, bx, col_map, x_cols)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{tc['epochs']}], Loss: {running_loss/len(train_loader):.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'x_cols': x_cols,
        'y_cols': y_cols,
        'col_map': col_map,
        'input_size': len(x_cols),
        'seq_len': seq_len
    }, MODEL_SAVE_PATH)
    print(f"モデル保存完了: {MODEL_SAVE_PATH}")

    # モデル保存と同じディレクトリに列名リストを書き出す
    config_path = MODEL_SAVE_PATH.parent / "model02_config.json"
    config_data = {
        "x_cols": x_cols,
        "y_cols": y_cols,
        "seq_len": 11
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)
    print(f"Config saved to {config_path}")

if __name__ == "__main__":
    main()