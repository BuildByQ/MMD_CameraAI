import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'

PATHS = {
    'normalization_dir': DATA_ROOT / 'normalization_params',
    'describe_dir': DATA_ROOT / 'describe_result_per_song_camera'
}

PATHS['describe_dir'].mkdir(exist_ok=True)

# --- 読み込み ---
df_cam = pd.read_csv(PATHS['normalization_dir'] / "normalized_camera.csv")

# --- 曲IDのカラム名（必要に応じて変更） ---
SONG_ID_COL = "song_id"   # もし "music_id" ならここを変更

# --- describe を出力する関数 ---
def describe_camera_per_song(df):
    out_dir = PATHS['describe_dir']
    out_dir.mkdir(exist_ok=True)

    # 全体の describe
    df.describe().round(4).to_csv(out_dir / "camera_all_describe.csv")

    # 曲ごとの describe
    for song_id, group in df.groupby(SONG_ID_COL):
        desc = group.describe().round(4)
        desc.to_csv(out_dir / f"camera_song_{song_id}_describe.csv")
        print(f"[OK] camera: song_id={song_id} → {desc.shape}")

# --- 実行 ---
describe_camera_per_song(df_cam)