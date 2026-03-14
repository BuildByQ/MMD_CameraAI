import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'

PATHS = {
    'normalization_dir': DATA_ROOT / 'normalization_params',
    'describe_dir': DATA_ROOT / 'describe_result_per_song_camera'
}

PATHS['describe_dir'].mkdir(exist_ok=True)

df_cam = pd.read_csv(PATHS['normalization_dir'] / "normalized_camera.csv")

SONG_ID_COL = "song_id"   # 必要なら music_id に変更

# ラベル列だけ抽出（あなたの config に合わせて調整）
label_cols = [
    "long_shot","full_shot","medium_shot","high_angle","low_angle","dutch_angle",
    "tilt","pan","roll","dolly_in","dolly_out","tracking_left","tracking_right",
    "crane_up","crane_down","zoom_in","zoom_out"
]

# 曲ごとの平均値だけをまとめる
per_song_means = df_cam.groupby(SONG_ID_COL)[label_cols].mean().round(4)

# CSV 出力
per_song_means.to_csv(PATHS['describe_dir'] / "camera_per_song_means.csv")

print(per_song_means)