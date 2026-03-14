import numpy as np
import pandas as pd
import librosa

# --- 基本変換関数 ---

def log_transform_distance(x):
    """距離: 負値は0クリップ → log1p変換"""
    x = np.clip(x, a_min=0, a_max=None)
    return np.log1p(x)

def inverse_log_transform_distance(y):
    return np.expm1(np.maximum(y, 0))

def signed_log_transform(x):
    """位置座標: -∞〜∞対応の符号付きlog変換"""
    return np.sign(x) * np.log1p(np.abs(x))

def inverse_signed_log_transform(y):
    return np.sign(y) * (np.expm1(np.abs(y)))

def yaw_to_features(yaw_deg, rotations_count=None, rotations_range=(-2, 2)):
    """Yaw角度を sin/cos + 回転回数 one-hot に変換"""
    yaw_rad = np.deg2rad(yaw_deg)
    feats = [np.sin(yaw_rad), np.cos(yaw_rad)]
    if rotations_count is not None:
        low, high = rotations_range
        bins = list(range(low, high + 1))
        onehot = np.zeros(len(bins))
        idx = int(np.clip(rotations_count, low, high)) - low
        onehot[idx] = 1.0
        feats += list(onehot)
    return np.array(feats, dtype=np.float32)

def normalize_fov(fov, mode="minmax", rng=(1, 125), mean=None, std=None):
    """FOVを正規化"""
    if mode == "minmax":
        lo, hi = rng
        return (fov - lo) / (hi - lo)
    elif mode == "zscore":
        assert mean is not None and std is not None and std > 0
        return (fov - mean) / std
    else:
        return fov

# --- メイン処理関数 ---

def load_camera_csv(csv_path, config):
    """
    カメラCSVを読み込み、前処理済み特徴量を返す
    戻り値: features (N, D), frames (N,)
    """
    df = pd.read_csv(csv_path)

    # 基本列
    frames = df["frame"].values
    pos = df[["pos_x", "pos_y", "pos_z"]].values
    rot = df[["rot_x", "rot_y", "rot_z"]].values
    distance = df["distance"].values
    fov = df["fov"].values

    # --- 前処理 ---
    # 位置座標: signed log
    pos_proc = signed_log_transform(pos)

    # 距離: log変換
    dist_proc = log_transform_distance(distance)

    # 回転角度
    pitch = rot[:, 0]  # rot_x
    yaw = rot[:, 1]    # rot_y
    roll = rot[:, 2]   # rot_z

    yaw_feats = np.stack([yaw_to_features(y) for y in yaw], axis=0)

    # FOV
    fov_proc = np.array([normalize_fov(v, mode=config["preprocess"]["fov"]["normalize"],
                                       rng=tuple(config["preprocess"]["fov"]["range"])) for v in fov])

    # --- 特徴量結合 ---
    features = np.concatenate([
        pos_proc,
        dist_proc.reshape(-1, 1),
        pitch.reshape(-1, 1),
        roll.reshape(-1, 1),
        yaw_feats,
        fov_proc.reshape(-1, 1)
    ], axis=1)

    return features, frames

def load_motion_csv(csv_path, config):
    """
    モデルモーションCSVを読み込み、主要24ボーンのみ抽出して前処理済み特徴量を返す
    戻り値: features (N, D), frames (N,), bone_names (N,)
    """

    # 主要24ボーンセット（固定）
    target_bones = [
        "センター","上半身","下半身","上半身2","首","頭","腰","グルーブ","すべての親",
        "右肩","右腕","右ひじ","右手首",
        "左肩","左腕","左ひじ","左手首",
        "右足","右ひざ","右足首",
        "左足","左ひざ","左足首",
        "両目"
    ]

    df = pd.read_csv(csv_path)

    # 対象ボーンのみ抽出
    df = df[df["bone_name"].isin(target_bones)]

    frames = df["frame"].values
    bone_names = df["bone_name"].values

    # 位置座標: signed log
    pos = df[["pos_x", "pos_y", "pos_z"]].values
    pos_proc = signed_log_transform(pos)

    # 回転: quaternionそのまま
    rot = df[["rot_x", "rot_y", "rot_z", "rot_w"]].values

    # 特徴量結合
    features = np.concatenate([pos_proc, rot], axis=1)

    return features, frames, bone_names

def load_audio_mfcc(wav_path, fps=30, n_mfcc=13, hop_length=512):
    """
    wavファイルからMFCCを抽出し、MMDのframe列に対応させる
    戻り値: features (N, n_mfcc), frames (N,)
    """

    # wav読み込み
    y, sr = librosa.load(wav_path, sr=None)  # sr=Noneで元のサンプリングレートを保持

    # MFCC抽出
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc = mfcc.T  # (time, n_mfcc) に転置

    # 各フレームの時間（秒）
    times = librosa.frames_to_time(np.arange(mfcc.shape[0]), sr=sr, hop_length=hop_length)

    # MMDのframe番号に変換
    frames = np.floor(times * fps).astype(int)

    return mfcc, frames


# --- シーケンス化ユーティリティ ---

def build_sequences(features, frames, seq_len=11):
    """
    前後フレームを含めたシーケンスを構築
    戻り値: X_seq (M, seq_len, D), center_frames (M,)
    """
    half = seq_len // 2
    X_seq, center_frames = [], []
    for i in range(len(features)):
        start, end = i - half, i + half
        if start < 0 or end >= len(features):
            continue
        window = features[start:end+1]
        X_seq.append(window)
        center_frames.append(frames[i])
    return np.array(X_seq), np.array(center_frames)