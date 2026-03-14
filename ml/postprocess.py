import pandas as pd
import json
import sys
import numpy as np
from pathlib import Path

# プロジェクトのルートディレクトリを設定
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'  # 1つ上の階層のdataフォルダを指す
ML_ROOT = PROJECT_ROOT.parent / 'ml'  # 1つ上の階層のmlフォルダを指す
PYTHON = sys.executable  # 現在のPythonインタプリタを使用
LABEL_ROOT = PROJECT_ROOT.parent / 'predict_01' # 1つ上の階層のpredict_01フォルダを指す

# ディレクトリパスの設定
PATHS = {
    'normalized_dir': DATA_ROOT / 'normalization_params',
    'analysis_result_dir': PROJECT_ROOT / 'analysis_result'  # 出力ディレクトリはスクリプトと同じ階層に
}

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

def load_prediction_csv(path: str) -> pd.DataFrame:
    """
    第1モデルの出力CSVを読み込み、
    後続処理が扱いやすいDataFrameに整形して返す。
    """

    # CSV読み込み
    config_path = LABEL_ROOT / 'prediction_full.csv'
    df = pd.read_csv(config_path)

    # frame を int に統一（念のため）
    if "frame" in df.columns:
        df["frame"] = df["frame"].astype(int)

    # ラベル列を float に統一（確率として扱うため）
    # frame や補間係数などは除外
    label_cols = [
        col for col in df.columns
        if col not in ["frame"]
        and not col.endswith("_x1")
        and not col.endswith("_x2")
        and not col.endswith("_y1")
        and not col.endswith("_y2")
        and col not in ["pos_x", "pos_y", "pos_z",
                        "rot_x", "rot_y", "rot_z",
                        "distance", "fov",
                        "rot_x_speed", "rot_x_sin", "rot_x_cos",
                        "rot_y_speed", "rot_y_sin", "rot_y_cos",
                        "rot_z_speed", "rot_z_sin", "rot_z_cos"]
    ]

    # ラベル列を float に変換
    df[label_cols] = df[label_cols].astype(float)

    print(f"Loaded CSV: {path}")
    print(f"Rows: {len(df)}, Label columns: {len(label_cols)}")

    return df

def smooth_change_point(df: pd.DataFrame, postprocess_config: dict) -> pd.DataFrame:
    """
    change_point を「10秒あたりの目標イベント数」に合わせて自動調整し、
    区間境界として扱いやすいように整形する。
    区間分割は行わず、あくまで change_point 列の整形のみを行う。
    """

    # 設定値の取得
    target_per_10s = postprocess_config.get("target_change_points_per_10s", None)
    expand = postprocess_config.get("change_point_expand", 0)

    if target_per_10s is None:
        return df

    FPS = 30
    frames_per_10s = FPS * 10
    total_frames = len(df)

    # 曲全体で必要な change_point 数
    target_total = int((total_frames / frames_per_10s) * target_per_10s)

    # raw の change_point（確率）
    raw = df["change_point"].values

    # -----------------------------
    # ① 閾値探索（0.99 → 0.01）
    # -----------------------------
    best_threshold = 0.5
    best_diff = float("inf")
    best_binary = None

    for th in np.linspace(0.99, 0.01, 99):
        binary = (raw >= th).astype(int)
        count = binary.sum()

        diff = abs(count - target_total)
        if diff < best_diff:
            best_diff = diff
            best_threshold = th
            best_binary = binary

    # 最適閾値で確定
    change = best_binary.copy()

    # -----------------------------
    # ② 孤立点（1フレームだけ）を削除
    # -----------------------------
    cleaned = change.copy()
    for i in range(1, len(change) - 1):
        if change[i] == 1 and change[i - 1] == 0 and change[i + 1] == 0:
            cleaned[i] = 0
    change = cleaned

    # -----------------------------
    # ③ 連続区間を解析し、2フレームに丸める
    #    3フレーム以上の場合は最大確率位置を中心に2フレーム採用
    # -----------------------------
    final_change = np.zeros_like(change)
    i = 0
    while i < len(change):
        if change[i] == 1:
            start = i
            while i < len(change) and change[i] == 1:
                i += 1
            end = i  # end は 1 の次の位置

            length = end - start

            if length == 1:
                # 孤立点は削除済みなので無視
                pass

            elif length == 2:
                # そのまま採用
                final_change[start] = 1
                final_change[start + 1] = 1

            else:
                # 3フレーム以上 → 最大確率位置を中心に2フレーム採用
                segment = raw[start:end]
                max_idx = np.argmax(segment)
                center = start + max_idx

                final_change[center] = 1
                if center + 1 < len(change):
                    final_change[center + 1] = 1

        else:
            i += 1

    change = final_change

    # -----------------------------
    # ④ expand（前後補強）
    # -----------------------------
    if expand > 0:
        expanded = change.copy()
        for i in range(len(change)):
            if change[i] == 1:
                start = max(0, i - expand)
                end = min(len(change), i + expand + 1)
                expanded[start:end] = 1
        change = expanded

    # df に反映
    df = df.copy()
    df["change_point"] = change

    print(f"[change_point] target={target_total}, threshold={best_threshold:.3f}, actual={change.sum()}")

    return df

def split_intervals(change_point_series):
    """
    change_point が 1 の位置で区間を切り、
    [(start, end), (start, end), ...] のリストを返す。
    """
    intervals = []
    start = 0

    for i in range(1, len(change_point_series)):
        if change_point_series.iloc[i] == 1:
            intervals.append((start, i - 1))
            start = i

    intervals.append((start, len(change_point_series) - 1))
    return intervals


def sharpen_prob(p: np.ndarray, T: float = 0.5) -> np.ndarray:
    """
    確率方向のシャープ化。
    0 に近い値はより 0 に、1 に近い値はより 1 に近づける。
    """
    eps = 1e-8
    # logit
    x = np.log((p + eps) / (1 - p + eps))
    # 温度スケーリング
    x = x / T
    # sigmoid
    return 1 / (1 + np.exp(-x))


def smooth_labels(df: pd.DataFrame, postprocess_config: dict) -> pd.DataFrame:
    """
    第1モデルのラベル出力を、第2モデルが扱いやすい形に整形する。
    ・区間抽出（change_point）
    ・区間内 smoothing（rolling mean）
    ・区間内で平均が低いラベルを強制 0（滲み除去）
    ・確率方向のシャープ化（logit-based）
    ・自然なショット遷移は残す（均一化しない）
    """

    window = postprocess_config.get("label_smooth_window", 5)
    threshold = postprocess_config.get("label_zero_threshold", 0.1)
    T = postprocess_config.get("label_sharpen_temperature", 0.5)

    # change_point から区間抽出
    intervals = split_intervals(df["change_point"])

    # change_point 以外のラベル列を対象にする
    label_cols = [c for c in df.columns if c != "change_point"]

    df_out = df.copy()

    for (start, end) in intervals:
        interval_df = df_out.iloc[start:end+1]

        # -----------------------------
        # ① 区間内 smoothing（rolling mean）
        # -----------------------------
        smoothed = interval_df[label_cols].rolling(
            window=window, min_periods=1, center=True
        ).mean()

        # -----------------------------
        # ② 区間内で平均が低いラベルを強制 0（滲み除去）
        # -----------------------------
        means = smoothed.mean(axis=0)
        low_labels = means[means < threshold].index.tolist()
        smoothed[low_labels] = 0.0

        # -----------------------------
        # ③ 確率方向のシャープ化（logit-based）
        # -----------------------------
        sharpened = smoothed.copy()
        for col in label_cols:
            sharpened[col] = sharpen_prob(sharpened[col].values, T=T)

        # -----------------------------
        # ④ df に書き戻し
        # -----------------------------
        df_out.loc[start:end, label_cols] = sharpened.values

    return df_out


def main(config_path: str):
    # 0. JSON 読み込み
    config = load_config(config_path)

    # ① CSV読み込み
    df = load_prediction_csv(config["data"]["prediction_csv"])

    # ② smoothing
    df = smooth_change_point(df, config["postprocess"]) 
    df = smooth_labels(df, config["postprocess"])

    # ③ multi-label 安定化
    # df = stabilize_multilabel(df)

    # ④ 状態系/動き系の分離
    # df = split_state_motion(df)

    # ⑤ change_point smoothing
    # df = smooth_change_point(df)

    # ⑥ 特徴量正規化
    # df = normalize_features(df)

    # ⑦ 意図的な調整（tilt/pan/change など）
    # df = apply_intentional_modifications(df, config["postprocess"])

    # ⑧ Transformer 入力形式に整形
    # sequences = build_sequences(df, config["labels_features"])

    # ⑨ 曲単位に分割
    # song_sequences = split_by_song(sequences)

    # ⑩ padding/mask
    # padded, masks = create_padding_and_masks(song_sequences, config["model"]["seq_len"])

    # ⑪ 保存
    # save_dataset(padded, masks, config["output_dir"])