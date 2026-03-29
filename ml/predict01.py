import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from pathlib import Path
from model01 import (
    load_config, get_dynamic_columns, build_model, CameraLabelGRUModel,
    ML_ROOT, LABEL_ROOT
)
# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent

def load_trained_model(config, df, device="cpu", model_path=None):
    if model_path is None:
        model_path = "model_final.pth"

    full_path = os.path.join(config["output_dir"], model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")

    # 1. カラム解決
    feature_cols, label_cols = get_dynamic_columns(config, df.columns.tolist())
    input_dim = len(feature_cols) * 3 
    output_dim = len(label_cols)

    # 2. 学習時の曲リストを再現
    # df（フィルタリング前の全データ）から、学習に使ったはずの曲を特定する
    test_songs = config["data"].get("test_songs", [])
    train_songs_config = config["data"].get("train_songs", [])

    if train_songs_config:
        # configに明記されているならそれを使う
        train_songs = train_songs_config
    else:
        # 明記されていない場合、全曲からテスト曲を除いたものが学習に使われたとみなす
        all_possible_songs = sorted(df["song_id"].unique().tolist())
        train_songs = [sid for sid in all_possible_songs if sid not in test_songs]

    # 学習時: num_songs = len(train_songs) + 1 (unknown用)
    num_songs = len(train_songs) + 1
    unknown_idx = num_songs - 1

    print(f"DEBUG: train_songs detected: {train_songs} (count: {len(train_songs)})")
    print(f"DEBUG: total num_songs for embedding: {num_songs}")

    # 3. モデル構築
    model = build_model(config, input_dim, output_dim, num_songs)

    # 4. 重みロード
    state_dict = torch.load(full_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return model, train_songs, unknown_idx, feature_cols, label_cols

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
    feature_cols, _ = get_dynamic_columns(config, df.columns.tolist())

    base_dim = len(feature_cols)
    stats_dim = 2 * base_dim  # mean + std
    # input_dim = base_dim + stats_dim

    # song_id → embedding ID（未知曲用の unknown_idx を追加）
    song_to_idx = {sid: i for i, sid in enumerate(song_ids)}
    song_to_idx["unknown"] = unknown_idx

    X_list = []
    song_idx_list = []
    frame_list = []
    song_id_str_list = []

    print("\n=== 推論用スライディングウィンドウ生成（未知曲対応版） ===")

    # df に存在する曲IDをループ（predict 対象の曲だけでも OK）
    for sid in df["song_id"].unique():
        df_song = df[df["song_id"] == sid].drop_duplicates(subset=['frame']).sort_values("frame")
        
        X_song = df_song[feature_cols].values
        frames = df_song["frame"].values
        N = len(X_song)
        
        print(f"   - song_id={sid}: 修正後フレーム数 {N}")
        if N < seq_len:
            print(f"  - song_id={sid}: フレーム不足のためスキップ")
            continue

        # 曲全体の統計量（mean + std）
        song_stats = df_song[feature_cols].agg(['mean', 'std']).values.flatten()
        stats_broadcast = np.repeat(song_stats.reshape(1, -1), seq_len, axis=0)

        # 未知曲なら unknown_idx を使う
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
            frame_list.append(int(frames[i + center]))
            song_id_str_list.append(sid)

    if len(X_list) == 0:
        raise ValueError("推論用のウィンドウが1つも生成されませんでした。")

    # numpy → tensor
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    song_idx_tensor = torch.tensor(np.array(song_idx_list), dtype=torch.long).to(device)

    print(f"\n生成された総シーケンス数: {len(X_tensor)}")
    print(f"X_tensor shape: {X_tensor.shape}")

    return X_tensor, song_idx_tensor, frame_list, song_id_str_list


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

    # 未知曲なら unknown_idx を使う
    if song_id in song_to_idx:
        sid_idx = song_to_idx[song_id]
    else:
        print(f"[Info] song_id={song_id} は学習データに存在しません → unknown embedding を使用します")
        sid_idx = unknown_idx

    # --- 1. 対象曲だけ抽出（未知曲なら空になる可能性あり） ---
    df_song = df[df["song_id"] == song_id].sort_values("frame")
    X_song = df_song[feature_cols].values
    N = len(X_song)

    # 未知曲で df にデータが無い場合の処理
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


def predict(model, X, song_idx, label_names, threshold=0.5, temperature=1.5):
    model.train()
    with torch.no_grad():
        logits = model(X, song_idx)
        probs = torch.sigmoid(logits / temperature)

    probs_np = probs.cpu().numpy()
    binary_np = (probs_np >= threshold).astype(int)

    results = []
    for i in range(len(probs_np)):
        prob_dict = {label_names[j]: float(probs_np[i][j]) for j in range(len(label_names))}
        active_labels = [label_names[j] for j in range(len(label_names)) if binary_np[i][j] == 1]
        results.append({
            "labels": active_labels,
            "probabilities": prob_dict,
            "binary": binary_np[i].tolist()
        })
    return results

def save_results(results, output_path, label_names, original_frames=None, song_id_list=None):
    """
    推論結果を CSV に保存する（train.py と整合した predict 用）
    results: predict() の返り値（リスト）
    original_frames: 各予測に対応する元の frame 番号（任意）
    """

    rows = []

    for i, r in enumerate(results):
        row = {}

        if song_id_list is not None:
            row["song_id"] = song_id_list[i]
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
    cols = ["song_id", "frame", "labels"] + [c for c in df.columns if c not in ["song_id", "frame", "labels"]]
    df[cols].to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved prediction results to {output_path}")

def main():
    # 1. 設定とデバイスの準備
    config = load_config("config.json")
    device = config["training"].get("device", "cpu")
    
    # config.json に "prediction_count": 10 のように記述がある想定
    num_runs = config["predict"].get("prediction_count", 10) 

    # 2. 全データの読み込み（学習曲の特定とEmbedding次元の計算用）
    df_full = load_normalized_csvs(config)
    
    # 3. モデルの準備（一度だけ実行）
    model, train_songs, unknown_idx, feat_cols, label_cols = load_trained_model(
        config, df_full, device=device
    )

    # 4. 推論対象曲のリストアップ
    test_songs = config["data"].get("test_songs", [])
    if not test_songs:
        print("推論対象 (test_songs) が空です。")
        return

    # --- 監督、ここからがループ処理です ---
    for sid in test_songs:
        print(f"\nターゲット曲: {sid}")
        
        # その曲のデータだけを抽出
        df_target = df_full[df_full["song_id"] == sid].copy()
        if df_target.empty:
            print(f"{sid} のデータが見つかりません。スキップします。")
            continue

        # 5. 推論用前処理（特徴量作成は曲ごとに1回でOK）
        X, song_idx, original_frames, song_id_list = preprocess_full_csv(
            df_target, config, train_songs, unknown_idx, device=device
        )

        # 6. 指定回数だけ推論を実行して保存
        for run_idx in range(1, num_runs + 1):
            # 再現性を確保するためのシード固定（回数ごとに異なるが、毎回同じ結果になる）
            # もし predict() 内で np.random を使っている場合、これで固定されます
            np.random.seed(run_idx)
            torch.manual_seed(run_idx)

            run_id_str = f"{run_idx:02d}"
            print(f"  └─ 推論中... [{run_id_str}/{num_runs}]")

            # 既存の predict 関数をそのまま呼び出し
            # ※ temperature 等の引数は、監督が書き換えた predict の定義に合わせて調整してください
            results = predict(model, X, song_idx, label_cols)

            # 7. 結果の保存（曲名と連番をファイル名に付与）
            os.makedirs(LABEL_ROOT, exist_ok=True)
            output_filename = f"predict_{sid}_{run_id_str}.csv"
            output_path = LABEL_ROOT / output_filename
            
            # 既存の save_results 関数をそのまま呼び出し
            save_results(results, output_path, label_cols, original_frames, song_id_list)

    print(f"\n=== 全 {len(test_songs)} 曲 × {num_runs} 回の推論が完了しました ===")

if __name__ == "__main__":
    main()