import os
import torch
import numpy as np
import json
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ML_ROOT = PROJECT_ROOT.parent / "ml"
LABEL_ROOT = PROJECT_ROOT.parent / 'predict_01' # 1つ上の階層のpredict_01フォルダを指す

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

def load_trained_model(config, df, device="cpu", model_path=None):

    if model_path is None:
        model_path = "model_final.pth"

    full_path = os.path.join(config["output_dir"], model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")

    # ★ train_songs を config から取得
    train_songs = config["data"].get("train_songs", [])
    test_songs = config["data"].get("test_songs", [])

    # ★ train_songs が空なら → df ではなく全データから取得する
    if len(train_songs) == 0:
        
        df_all = load_normalized_csvs(config)   # ← ここが重要
        all_songs = df_all["song_id"].unique().tolist()
        train_songs = [sid for sid in all_songs if sid not in test_songs]

    # ★ predict.py の song_ids は train候補曲と一致させる
    song_ids = train_songs

    # unknown embedding を追加
    num_songs = len(song_ids) + 1
    unknown_idx = num_songs - 1

    # input_dim 計算
    base_dim = len(config["motion_features"]) + len(config["audio_features"])
    stats_dim = 2 * base_dim
    input_dim = base_dim + stats_dim

    # モデル構築
    model = build_model(config, input_dim, num_songs)

    # 重みロード
    state_dict = torch.load(full_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return model, song_ids, unknown_idx

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

def preprocess_full_csv(df, config, song_ids, unknown_idx, device="cpu"):
    """
    正規化済みCSVを前提とした、推論用の前処理（未知曲対応版）
    - motion + audio 特徴量
    - 曲ごとの統計量（mean + std）
    - seq_len のスライディングウィンドウ
    - song_idx（未知曲は unknown_idx）
    - original_frames（各ウィンドウの中心フレーム番号）を返す
    """

    seq_len = config["model"]["seq_len"]
    center = seq_len // 2

    # 特徴量列（camera は使わない）
    feature_cols = (
        config["motion_features"]
        + config["audio_features"]
    )

    base_dim = len(feature_cols)
    stats_dim = 2 * base_dim  # mean + std
    input_dim = base_dim + stats_dim

    # song_id → embedding ID（未知曲用の unknown_idx を追加）
    song_to_idx = {sid: i for i, sid in enumerate(song_ids)}
    song_to_idx["unknown"] = unknown_idx

    X_list = []
    song_idx_list = []
    frame_list = []

    print("\n=== 推論用スライディングウィンドウ生成（未知曲対応版） ===")

    # df に存在する曲IDをループ（predict 対象の曲だけでも OK）
    for sid in df["song_id"].unique():
        df_song = df[df["song_id"] == sid].sort_values("frame")
        X_song = df_song[feature_cols].values
        frames = df_song["frame"].values

        N = len(X_song)
        if N < seq_len:
            print(f"  - song_id={sid}: フレーム不足のためスキップ")
            continue

        # 曲全体の統計量（mean + std）
        song_stats = df_song[feature_cols].agg(['mean', 'std']).values.flatten()
        stats_broadcast = np.repeat(song_stats.reshape(1, -1), seq_len, axis=0)

        # ★ 未知曲なら unknown_idx を使う
        if sid in song_to_idx:
            sid_idx = song_to_idx[sid]
        else:
            sid_idx = unknown_idx

        print(f"  - song_id={sid}: {N} フレーム → {N - seq_len + 1} シーケンス (sid_idx={sid_idx})")

        # スライディングウィンドウ
        for i in range(N - seq_len + 1):
            window = X_song[i:i+seq_len]
            window_with_stats = np.concatenate([window, stats_broadcast], axis=1)

            X_list.append(window_with_stats)
            song_idx_list.append(sid_idx)

            # 中心フレーム番号
            frame_list.append(int(frames[i + center]))

    if len(X_list) == 0:
        raise ValueError("推論用のウィンドウが1つも生成されませんでした。")

    # numpy → tensor
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    song_idx_tensor = torch.tensor(np.array(song_idx_list), dtype=torch.long).to(device)

    print(f"\n生成された総シーケンス数: {len(X_tensor)}")
    print(f"X_tensor shape: {X_tensor.shape}")

    return X_tensor, song_idx_tensor, frame_list


def preprocess_single_frame(df, song_id, frame_index, config, song_ids, unknown_idx, device="cpu"):
    """
    特定曲の特定フレームを中心に seq_len の入力を作る（未知曲対応版）
    - motion + audio 特徴量
    - 曲ごとの統計量（mean + std）
    - パディング対応
    - song_idx（未知曲は unknown_idx）
    """

    seq_len = config["model"]["seq_len"]
    half = seq_len // 2

    # 特徴量列（camera は使わない）
    feature_cols = (
        config["motion_features"]
        + config["audio_features"]
    )

    base_dim = len(feature_cols)
    stats_dim = 2 * base_dim  # mean + std
    input_dim = base_dim + stats_dim

    # song_id → embedding ID（未知曲対応）
    song_to_idx = {sid: i for i, sid in enumerate(song_ids)}

    # ★ 未知曲なら unknown_idx を使う
    if song_id in song_to_idx:
        sid_idx = song_to_idx[song_id]
    else:
        print(f"[Info] song_id={song_id} は学習データに存在しません → unknown embedding を使用します")
        sid_idx = unknown_idx

    # --- 1. 対象曲だけ抽出（未知曲なら空になる可能性あり） ---
    df_song = df[df["song_id"] == song_id].sort_values("frame")
    X_song = df_song[feature_cols].values
    N = len(X_song)

    # ★ 未知曲で df にデータが無い場合の処理
    if N == 0:
        print(f"[Info] song_id={song_id} のデータが df に存在しません → 未知曲としてゼロ特徴量で処理します")

        # ゼロ特徴量で seq_len を作成
        zero_frame = np.zeros(base_dim, dtype=np.float32)
        seq = np.array([zero_frame for _ in range(seq_len)], dtype=np.float32)

        # 統計量もゼロ
        stats_broadcast = np.zeros((seq_len, stats_dim), dtype=np.float32)

        seq_with_stats = np.concatenate([seq, stats_broadcast], axis=1)

        X_tensor = torch.tensor(seq_with_stats, dtype=torch.float32).unsqueeze(0).to(device)
        song_idx_tensor = torch.tensor([sid_idx], dtype=torch.long).to(device)

        return X_tensor, song_idx_tensor

    # --- 2. 曲全体の統計量（mean + std） ---
    song_stats = df_song[feature_cols].agg(['mean', 'std']).values.flatten()
    stats_broadcast = np.repeat(song_stats.reshape(1, -1), seq_len, axis=0)

    # --- 3. frame_index を中心に seq_len のウィンドウを作成 ---
    start = frame_index - half
    end = frame_index + half

    seq = []
    for i in range(start, end + 1):
        if i < 0:
            seq.append(X_song[0])       # 先頭フレームでパディング
        elif i >= N:
            seq.append(X_song[-1])      # 末尾フレームでパディング
        else:
            seq.append(X_song[i])

    seq = np.array(seq, dtype=np.float32)

    # --- 4. 統計量を concat ---
    seq_with_stats = np.concatenate([seq, stats_broadcast], axis=1)

    # --- 5. tensor 化 ---
    X_tensor = torch.tensor(seq_with_stats, dtype=torch.float32).unsqueeze(0).to(device)
    song_idx_tensor = torch.tensor([sid_idx], dtype=torch.long).to(device)

    return X_tensor, song_idx_tensor


def predict(model, X, song_idx, config, threshold=0.5):
    """
    推論関数（train.py と完全整合版）
    - model(X, song_idx) を正しく呼び出す
    - sigmoid → 確率
    - threshold → バイナリ
    - ラベル名・確率・バイナリを返す
    """

    model.eval()

    with torch.no_grad():
        # ★ train.py と同じ forward 呼び出し
        logits = model(X, song_idx)  # shape: (batch, num_labels)
        probs = torch.sigmoid(logits)

    probs_np = probs.cpu().numpy()
    binary_np = (probs_np >= threshold).astype(int)

    label_names = config["labels_features"]

    results = []

    for i in range(len(probs_np)):
        prob_dict = {label_names[j]: float(probs_np[i][j]) for j in range(len(label_names))}
        binary_list = binary_np[i].tolist()

        # 1 のラベルだけ抽出
        active_labels = [
            label_names[j]
            for j in range(len(label_names))
            if binary_np[i][j] == 1
        ]

        results.append({
            "labels": active_labels,     # ① ラベル名
            "probabilities": prob_dict,  # ② 確率
            "binary": binary_list        # ③ バイナリ配列
        })

    return results

def save_results(results, output_path, config, original_frames=None):
    """
    推論結果を CSV に保存する（train.py と整合した predict 用）
    results: predict() の返り値（リスト）
    original_frames: 各予測に対応する元の frame 番号（任意）
    """

    label_names = config["labels_features"]
    rows = []

    for i, r in enumerate(results):
        row = {}

        # ① 元のフレーム番号（あれば）
        if original_frames is not None:
            row["frame"] = int(original_frames[i])
        else:
            row["frame"] = i  # fallback

        # ② ラベル名（1 のラベルだけ）
        row["labels"] = ",".join(r["labels"])

        # ③ 確率（各ラベル）
        for label in label_names:
            row[f"prob_{label}"] = r["probabilities"][label]

        # ④ バイナリ（0/1）
        for j, label in enumerate(label_names):
            row[f"bin_{label}"] = r["binary"][j]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved prediction results to {output_path}")

def main():
    # 1. 設定読み込み
    config = load_config("config.json")
    device = config["training"].get("device", "cpu")

    # 2. 正規化済み CSV を読み込み
    df = load_normalized_csvs(config)

    # ★ test_songs のみを推論対象にする
    test_songs = config["data"].get("test_songs", [])
    if len(test_songs) > 0:
        df = df[df["song_id"].isin(test_songs)]
        print(f"Predicting only test_songs: {test_songs}")
    else:
        print("Warning: test_songs is empty. Predicting all songs.")

    # 3. モデル読み込み（unknown_idx を受け取る）
    model, song_ids, unknown_idx = load_trained_model(config, df, device=device)

    # 4. 推論モードの選択
    # "full"   → 曲全体を推論
    # "single" → 特定フレームだけ推論
    mode = "full"   # 必要に応じて変更

    if mode == "full":
        print("Running full-sequence prediction...")

        # ★ unknown_idx を渡す（未知曲対応）
        X, song_idx, original_frames = preprocess_full_csv(
            df, config, song_ids, unknown_idx, device=device
        )

        # 推論
        results = predict(model, X, song_idx, config)

        # 結果の例として最初の3件だけ表示
        for i in range(min(3, len(results))):
            print(f"Frame {i}: {results[i]}")

        # CSV 保存
        output_prediction_path = LABEL_ROOT / "prediction_full.csv"
        save_results(results, output_prediction_path, config, original_frames)

    elif mode == "single":
        # 推論したい曲とフレーム番号
        song_id = 1
        frame_index = 300

        print(f"Running single-frame prediction for song_id={song_id}, frame={frame_index}...")

        # ★ unknown_idx を渡す（未知曲対応）
        X, song_idx = preprocess_single_frame(
            df, song_id, frame_index, config, song_ids, unknown_idx, device=device
        )

        # 推論（1件だけ）
        result = predict(model, X, song_idx, config)[0]

        print("Prediction result:")
        print(result)

    else:
        print("Invalid mode. Choose 'full' or 'single'.")

if __name__ == "__main__":
    main()