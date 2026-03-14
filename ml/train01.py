import json
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime

# プロジェクトのルートディレクトリを設定
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'  # 1つ上の階層のdataフォルダを指す
ML_ROOT = PROJECT_ROOT.parent / 'ml'  # 1つ上の階層のmlフォルダを指す
PYTHON = sys.executable  # 現在のPythonインタプリタを使用

# ディレクトリパスの設定
PATHS = {
    'normalized_dir': DATA_ROOT / 'normalization_params',
    'analysis_result_dir': PROJECT_ROOT / 'analysis_result'  # 出力ディレクトリはスクリプトと同じ階層に
}

def setup_logging(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"train_log_{timestamp}.txt")

    # 標準出力をファイルにコピーする
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    # 標準出力 + ファイル の両方に出力
    logfile = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = Tee(sys.stderr, logfile)

    print(f"=== Logging to {log_path} ===")

# 1. 設定読み込み
def load_config(config_path="config.json"):
    """
    設定ファイル（config.json）を読み込んで辞書として返す関数。
    train01.py の全処理がこの設定を参照する。
    """
    config_path = ML_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"設定を読み込みました: {config_path}")
    return config

# 2. 正規化済み CSV を読み込み
def load_normalized_csvs(config: Dict[str, Any]) -> pd.DataFrame:
    """
    正規化されたCSVファイルを読み込み、1つのDataFrameに結合する
    
    Args:
        config: 設定ファイルの内容
        
    Returns:
        pd.DataFrame: 結合されたデータフレーム
    """
    
    # 正規化済みデータのディレクトリを取得
    data_dir = Path(config.get('data', {}).get('normalized_dir', []))
    
    # 読み込むCSVファイルのパス
    csv_files = [
        data_dir / 'normalized_camera.csv',    # カメラデータ
        data_dir / 'normalized_motion.csv',    # モーションデータ
        data_dir / 'normalized_audio.csv',     # 音声データ
    ]
    
    # 各CSVファイルを読み込む
    dfs = {}
    for file_path in csv_files:
        if file_path.exists():
            print(f"読み込み中: {file_path}")
            try:
                # ファイル名から拡張子を除いた部分をキーとして使用
                key = file_path.stem.replace('normalized_', '')
                dfs[key] = pd.read_csv(file_path)
                print(f"  - 読み込み完了: {len(dfs[key])} 行")
            except Exception as e:
                print(f"  - エラー: {str(e)}")
                continue
        else:
            print(f"警告: ファイルが存在しません: {file_path}")
    
    if not dfs:
        raise FileNotFoundError("読み込むCSVファイルが見つかりません")
    
    # すべてのデータフレームを結合
    # キー（song_id, frame）で結合
    print("\nデータを結合しています...")
    result_df = None
    
    for key, df in dfs.items():
        if result_df is None:
            result_df = df
        else:
            # キーでマージ（内部結合）
            merge_on = ['song_id', 'frame']
            common_columns = set(result_df.columns) & set(df.columns)
            merge_on = list(set(merge_on) & common_columns)
            
            if not merge_on:
                raise ValueError(f"結合キーが見つかりません: {key}")
                
            print(f"  - {key} をマージ (キー: {merge_on})")
            result_df = pd.merge(result_df, df, on=merge_on, how='inner')
    
    if result_df is None:
        raise ValueError("有効なデータがありません")
    
    print(f"\n結合完了: 合計 {len(result_df)} 行")
    
    # 重複行の確認
    if result_df.duplicated().any():
        print("警告: 重複行が見つかりました。重複を削除します。")
        result_df = result_df.drop_duplicates()
        print(f"重複削除後: {len(result_df)} 行")
    
    # 欠損値の確認
    missing = result_df.isnull().sum()
    if missing.any():
        print("\n欠損値の数:")
        print(missing[missing > 0])
        # 欠損値を0で補完（必要に応じて変更）
        result_df = result_df.fillna(0)
        print("欠損値を0で補完しました")
    
    return result_df

def create_features_and_labels(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """データフレームから特徴量とラベルを抽出する"""

    # 設定ファイルから特徴量を取得
    camera_features = config.get('camera_features', [])
    motion_features = config.get('motion_features', [])
    audio_features = config.get('audio_features', [])
    label_columns = config.get('labels_features', [])

    # カテゴリ別の辞書
    categories = {
        "カメラ": camera_features,
        "モーション": motion_features,
        "音響": audio_features
    }

    # --- カテゴリ別の重複チェック ---
    print("\n=== 特徴量カテゴリ別の重複チェック ===")
    for name, feats in categories.items():
        dup = [f for f in feats if feats.count(f) > 1]
        if dup:
            unique_dup = list(dict.fromkeys(dup))
            print(f"⚠ {name} に重複あり: {unique_dup}")
        else:
            print(f"✓ {name} に重複なし")

    # --- 全体の重複チェック（順序維持） ---
    raw_features = camera_features + motion_features + audio_features

    all_features = []
    seen = set()
    for f in raw_features:
        if f not in seen:
            seen.add(f)
            all_features.append(f)

    if len(raw_features) != len(all_features):
        duplicates = [x for x in raw_features if raw_features.count(x) > 1]
        unique_duplicates = list(dict.fromkeys(duplicates))
        print(f"\n⚠ 全体で重複していた特徴量: {unique_duplicates}")
        print(f"重複を削除しました（{len(raw_features)} → {len(all_features)}）")

    # --- 存在チェック ---
    available_features = [f for f in all_features if f in df.columns]
    missing_features = set(all_features) - set(available_features)

    if missing_features:
        print(f"\n⚠ データフレームに存在しない特徴量: {missing_features}")

    print(f"使用可能な特徴量: {len(available_features)}/{len(all_features)}")

    # --- ラベル ---

    # 存在チェック
    missing_labels = [col for col in label_columns if col not in df.columns]
    if missing_labels:
        raise ValueError(f"ラベルとして指定された列が df に存在しません: {missing_labels}")

    # ラベル抽出
    y = df[label_columns].values

    if not label_columns:
        raise ValueError("ラベルとして使用できるカラムが見つかりません")

    # --- 抽出 ---
    X = df[available_features].values
    y = df[label_columns].values

    print(f"\n特徴量の次元: {X.shape}")
    print(f"ラベルの次元: {y.shape}")

    return X, y

def split_windows_random(windows_X, windows_y, val_ratio=0.2, seed=42):
    """
    windows_X: list or array of shape [num_windows, window_size, feature_dim]
    windows_y: list or array of shape [num_windows, label_dim]
    """

    # インデックスをランダムに分割
    idx = np.arange(len(windows_X))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_ratio, random_state=seed, shuffle=True
    )

    X_train = windows_X[train_idx]
    y_train = windows_y[train_idx]

    X_val = windows_X[val_idx]
    y_val = windows_y[val_idx]

    return X_train, y_train, X_val, y_val

def create_dataloaders_random(df: pd.DataFrame, config: dict):

    seq_len = config["model"]["seq_len"]
    batch_size = config["training"]["batch_size"]
    shuffle = config["training"]["shuffle"]
    val_ratio = 1 - config["training"]["train_split"]

    feature_cols = (
        config["motion_features"]
        + config["audio_features"]
    )
    label_cols = config["labels_features"]

    center = seq_len // 2

    # ============================================================
    # 0. song_id → 連番ID（style embedding 用）
    # ============================================================
    song_ids = df["song_id"].unique().tolist()
    song_to_idx = {sid: i for i, sid in enumerate(song_ids)}

    # ============================================================
    # 1. 全曲まとめてスライディングウィンドウ生成
    # ============================================================
    X_list = []
    y_list = []
    song_idx_list = []   # ★ 追加

    for sid in song_ids:
        df_song = df[df["song_id"] == sid].sort_values("frame")
        X_song = df_song[feature_cols].values
        y_song = df_song[label_cols].values

        # ★ 曲全体の統計量を計算
        song_stats = df_song[feature_cols].agg(['mean', 'std']).values.flatten()
        stats_broadcast = np.repeat(song_stats.reshape(1, -1), seq_len, axis=0)

        song_idx = song_to_idx[sid]  # ★ embedding 用 ID

        N = len(X_song)
        if N < seq_len:
            continue

        for i in range(N - seq_len + 1):
            window = X_song[i:i+seq_len]
            window_with_stats = np.concatenate([window, stats_broadcast], axis=1)

            X_list.append(window_with_stats)
            y_list.append(y_song[i+center])
            song_idx_list.append(song_idx)  # ★ 追加

    X_all = np.array(X_list)
    y_all = np.array(y_list)
    song_idx_all = np.array(song_idx_list)

    print("\n=== 全ウィンドウ数 ===")
    print("total windows:", len(X_all))

    # ============================================================
    # 2. ウィンドウ単位ランダム split
    # ============================================================
    idx = np.arange(len(X_all))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_ratio, shuffle=True, random_state=42
    )

    X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_all[train_idx], dtype=torch.float32)
    song_idx_train = torch.tensor(song_idx_all[train_idx], dtype=torch.long)

    X_val = torch.tensor(X_all[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_all[val_idx], dtype=torch.float32)
    song_idx_val = torch.tensor(song_idx_all[val_idx], dtype=torch.long)

    # ============================================================
    # 3. DataLoader 作成（song_idx を追加）
    # ============================================================
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, song_idx_train),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val, song_idx_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    print("\n=== Random Window Split ===")
    print(f"train samples: {len(X_train)}")
    print(f"val samples:   {len(X_val)}")

    return train_loader, val_loader, len(song_ids)   # ★ num_songs を返す

def create_dataloaders_one_song(df: pd.DataFrame, config: dict):

    seq_len = config["model"]["seq_len"]
    batch_size = config["training"]["batch_size"]
    shuffle = config["training"]["shuffle"]

    feature_cols = (
        config["motion_features"]
        + config["audio_features"]
    )
    label_cols = config["labels_features"]

    center = seq_len // 2

    # ============================================================
    # 1. 曲 ID を取得し、シャッフルして val=1曲 にする
    # ============================================================
    song_ids = df["song_id"].unique().tolist()
    random.shuffle(song_ids)

    val_song_ids = [song_ids[0]]
    train_song_ids = song_ids[1:]

    print("\n=== 曲単位で train/val を分割（val=1曲） ===")
    print(f"train 曲数: {len(train_song_ids)}")
    print(f"val 曲数:   {len(val_song_ids)}")
    print(f"val 曲:     {val_song_ids}")

    # ★ song_id → 連番ID に変換する辞書（style embedding 用）
    song_to_idx = {sid: i for i, sid in enumerate(song_ids)}

    # ============================================================
    # 2. train 用のスライディングウィンドウ生成
    # ============================================================
    X_train_list = []
    y_train_list = []
    song_idx_train_list = []   # ★ 追加

    for sid in train_song_ids:
        df_song = df[df["song_id"] == sid].sort_values("frame")
        X_song = df_song[feature_cols].values
        y_song = df_song[label_cols].values

        # ★ 曲全体の統計量を計算
        song_stats = df_song[feature_cols].agg(['mean', 'std']).values.flatten()
        stats_broadcast = np.repeat(song_stats.reshape(1, -1), seq_len, axis=0)

        song_idx = song_to_idx[sid]  # ★ この曲の embedding ID

        N = len(X_song)
        if N < seq_len:
            continue

        for i in range(N - seq_len + 1):
            window = X_song[i:i+seq_len]
            window_with_stats = np.concatenate([window, stats_broadcast], axis=1)

            X_train_list.append(window_with_stats)
            y_train_list.append(y_song[i+center])
            song_idx_train_list.append(song_idx)  # ★ 追加

    # ============================================================
    # 3. val 用のスライディングウィンドウ生成（1曲だけ）
    # ============================================================
    X_val_list = []
    y_val_list = []
    song_idx_val_list = []   # ★ 追加

    for sid in val_song_ids:
        df_song = df[df["song_id"] == sid].sort_values("frame")
        X_song = df_song[feature_cols].values
        y_song = df_song[label_cols].values

        # ★ 曲全体の統計量を計算
        song_stats = df_song[feature_cols].agg(['mean', 'std']).values.flatten()
        stats_broadcast = np.repeat(song_stats.reshape(1, -1), seq_len, axis=0)

        song_idx = song_to_idx[sid]  # ★ val 曲の embedding ID

        N = len(X_song)
        if N < seq_len:
            continue

        for i in range(N - seq_len + 1):
            window = X_song[i:i+seq_len]
            window_with_stats = np.concatenate([window, stats_broadcast], axis=1)

            X_val_list.append(window_with_stats)
            y_val_list.append(y_song[i+center])
            song_idx_val_list.append(song_idx)  # ★ 追加

    # ============================================================
    # 4. Tensor 化
    # ============================================================
    X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train_list), dtype=torch.float32)
    song_idx_train = torch.tensor(np.array(song_idx_train_list), dtype=torch.long)  # ★ long

    X_val = torch.tensor(np.array(X_val_list), dtype=torch.float32)
    y_val = torch.tensor(np.array(y_val_list), dtype=torch.float32)
    song_idx_val = torch.tensor(np.array(song_idx_val_list), dtype=torch.long)  # ★ long

    # ============================================================
    # 5. DataLoader 作成（song_idx を追加）
    # ============================================================
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, song_idx_train),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val, song_idx_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    print("\n=== DataLoader 情報 ===")
    print(f"train samples: {len(X_train)}")
    print(f"val samples:   {len(X_val)}")

    return train_loader, val_loader, len(train_song_ids) + len(val_song_ids)


class CameraAIModel(nn.Module):
    """
    カメラ制御のためのニューラルネットワークモデル
    - 入力: カメラ、モーション、音響特徴量
    - 出力: カメラパラメータ（位置、回転など）
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]  # デフォルトの隠れ層のユニット数
        layers = []
        prev_dim = input_dim
        
        # 隠れ層を動的に構築
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim
        
        # 出力層
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class CameraLabelGRUModel(nn.Module):
    def __init__(
        self,
        input_dim,          # 統計量込みの特徴量次元
        output_dim,         # ラベル数
        num_songs,          # 曲数（embedding 用）
        embed_dim=4,        # ★ 小さくして過学習を防ぐ
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        bidirectional=False
    ):
        super().__init__()

        # ★ song_id embedding（方式A）
        self.song_embedding = nn.Embedding(num_songs, embed_dim)
        nn.init.xavier_uniform_(self.song_embedding.weight)  # ★ 安定化のための初期化

        self.embed_dim = embed_dim

        # ★ embedding dropout（ID丸暗記を防ぐ）
        self.embed_dropout = nn.Dropout(0.3)

        # ★ GRU の入力次元は「特徴量 + embedding」
        self.gru = nn.GRU(
            input_size=input_dim + embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_size * (2 if bidirectional else 1)

        # 最終 hidden → ラベル
        self.fc = nn.Linear(gru_out_dim, output_dim)

    def forward(self, x, song_idx):
        """
        x: (batch, seq_len, input_dim)
        song_idx: (batch,)
        """

        # ★ song embedding を取得
        song_emb = self.song_embedding(song_idx)  # (batch, embed_dim)

        # ★ embedding を L2 正規化（暴走防止）
        song_emb = F.normalize(song_emb, dim=1)

        # ★ dropout（ID丸暗記を防ぐ）
        song_emb = self.embed_dropout(song_emb)

        # ★ seq_len にブロードキャスト
        song_emb = song_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        # → (batch, seq_len, embed_dim)

        # ★ 特徴量と embedding を concat
        x = torch.cat([x, song_emb], dim=2)
        # → (batch, seq_len, input_dim + embed_dim)

        # GRU
        out, h = self.gru(x)

        # 最後の hidden を使用
        last_hidden = h[-1]  # (batch, hidden)

        logits = self.fc(last_hidden)
        return logits

def build_model(config: Dict[str, Any], input_dim: int, num_songs: int) -> nn.Module:
    model_cfg = config.get("model", {})

    hidden_size = model_cfg.get("hidden_size", 256)
    num_layers = model_cfg.get("num_layers", 2)
    dropout = model_cfg.get("dropout", 0.5)
    bidirectional = model_cfg.get("bidirectional", False)
    embed_dim = model_cfg.get("embed_dim", 4)   # ★ 追加

    label_columns = config.get("labels_features", [])
    if not label_columns:
        raise ValueError("config['labels_features'] が空です。")

    output_dim = len(label_columns)

    model = CameraLabelGRUModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_songs=num_songs,   # ★ 追加
        embed_dim=embed_dim,   # ★ 追加
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    print("\n=== 第1モデル（GRU + Style Embedding）構造 ===")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n学習可能パラメータ数: {total_params:,}")

    return model


def setup_training(model, config, pos_weight, device="cpu"):
    """
    第1モデル（演出ラベル分類モデル）の損失関数と最適化手法を設定する。

    - 損失関数: BCEWithLogitsLoss（sigmoid を内部で計算するため数値安定性が高い）
    - Optimizer: AdamW（正しい weight decay を実装した Adam の改良版）
    - learning_rate / weight_decay は config["training"] から取得

    Returns:
        criterion: 損失関数
        optimizer: 最適化手法
    """

    train_cfg = config.get("training", {})

    learning_rate = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-5)

    # --- 損失関数 ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    print("\n=== Training Setup ===")
    print(f"Loss Function: BCEWithLogitsLoss")
    print(f"Optimizer: AdamW")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")

    return criterion, optimizer

def train(
    model,
    train_loader,
    criterion,
    optimizer,
    config,
    device="cpu",
    resume=False,
    start_epoch=0
):
    """
    save_interval ごとに checkpoint を保存し、
    途中停止（Ctrl+C）や途中再開に対応した学習ループ。
    """

    model.to(device)
    model.train()

    epochs = config["training"]["epochs"]
    save_interval = config["training"].get("save_interval", 10)
    output_dir = config.get("output_dir", "models")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")

    # --- ★ main() 側で resume 判定済みなので、train() では start_epoch を使うだけ ---
    print(f"Training will start from epoch {start_epoch}")

    # --- 学習ループ ---
    try:
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False
            )

            for X_batch, y_batch, song_idx_batch in progress_bar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                song_idx_batch = song_idx_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch, song_idx_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"batch_loss": loss.item()})

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            # --- save_interval ごとに checkpoint 保存 ---
            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch+1}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        print("You can resume training by setting resume=True")

    # --- 最終モデル保存 ---
    final_path = os.path.join(output_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    save_model(model, config, filename=f"model_epoch_{epoch}.pth")
    print(f"Final model saved to: {final_path}")


def validate(model, val_loader, criterion, config, device="cpu"):
    """
    第1モデル（演出ラベル分類）の検証ループ。
    - tqdm による進捗バー付き
    - epoch ごとの平均 loss を返す
    """

    model.to(device)
    model.eval()

    val_loss = 0.0

    # 勾配計算を無効化
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)

        for X_batch, y_batch, song_idx_batch in progress_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            song_idx_batch = song_idx_batch.to(device)

            logits = model(X_batch, song_idx_batch)
            loss = criterion(logits, y_batch)

            val_loss += loss.item()

            progress_bar.set_postfix({"batch_loss": loss.item()})

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.6f}")

    return avg_val_loss

def save_model(model, config, filename="model.pth"):
    """
    モデルの state_dict を保存する関数。
    - output_dir は config["output_dir"] から取得
    - ディレクトリが無ければ自動作成
    """

    output_dir = config.get("output_dir", "models")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, filename)

    # state_dict のみ保存（推奨方式）
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to: {save_path}")


def main():
    # 1. 設定読み込み
    config = load_config("config.json")

    # 2. ログ設定（config 読み込み後に行うべき）
    setup_logging(config["output_dir"])

    # 3. データ読み込み
    df = load_normalized_csvs(config)

    # 3.5 train/test の曲IDを取得
    train_songs = config["data"].get("train_songs", [])
    test_songs = config["data"].get("test_songs", [])

    # train_songs が空なら → test_songs 以外すべてを train に使う
    if len(train_songs) == 0:
        all_songs = df["song_id"].unique().tolist()
        train_songs = [sid for sid in all_songs if sid not in test_songs]

    print(f"Train songs: {train_songs}")
    print(f"Test songs (excluded from training): {test_songs}")

    # df を train 用にフィルタ
    df = df[df["song_id"].isin(train_songs)]

    # 4. DataLoader 作成
    train_loader, val_loader, base_num_songs = create_dataloaders_one_song(df, config)

    # 5. モデル定義
    num_songs = base_num_songs + 1   # ★ unknown embedding を追加

    base_dim = len(config["motion_features"]) + len(config["audio_features"])
    stats_dim = 2 * base_dim   # mean + std
    input_dim = base_dim + stats_dim
    model = build_model(config, input_dim, num_songs)

    # 6. 損失関数・最適化
    label_cols = config["labels_features"]
    label_means = df[label_cols].mean()

    pos_weight = torch.tensor(np.sqrt(1.0 / label_means.values + 1e-6), dtype=torch.float32)

    criterion, optimizer = setup_training(
        model, config, pos_weight, config["training"].get("device", "cpu")
    )

    # --- ★ resume 判定（config + checkpoint の両方を見る） ---
    output_dir = config["output_dir"]
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")

    # config 側の resume フラグ
    resume_flag = config["training"].get("resume", False)

    # checkpoint が存在し、かつ resume_flag が True のときだけ再開
    resume = resume_flag and os.path.exists(checkpoint_path)

    if resume:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config["training"].get("device", "cpu"))
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("Starting fresh training.")
        start_epoch = 0

    # 7. 学習
    train(
        model,
        train_loader,
        criterion,
        optimizer,
        config,
        device=config["training"].get("device", "cpu"),
        resume=resume,
        start_epoch=start_epoch
    )

    # 8. 検証
    validate(
        model,
        val_loader,
        criterion,
        config,
        device=config["training"].get("device", "cpu")
    )

    # 9. 最終モデル保存
    save_model(model, config, filename="model_final.pth")


if __name__ == "__main__":
    main()