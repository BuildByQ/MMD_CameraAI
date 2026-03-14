import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'  # 1つ上の階層のdataフォルダを指す

# ディレクトリパスの設定
PATHS = {
    'normalization_dir': DATA_ROOT / 'normalization_params',  # 出力ディレクトリはスクリプトと同じ階層に
    'describe_dir': DATA_ROOT / 'describe_result'  # 出力ディレクトリはスクリプトと同じ階層に
}

PATHS['describe_dir'].mkdir(exist_ok=True)

df = pd.read_csv(PATHS['normalization_dir'] / "normalized_audio.csv")
print(df.describe())
df.describe().round(4).to_csv(PATHS['describe_dir'] / "audio_describe.csv")

df_cam = pd.read_csv(PATHS['normalization_dir'] / "normalized_camera.csv")
print(df_cam.describe())
df_cam.describe().round(4).to_csv(PATHS['describe_dir'] / "camera_describe.csv")

df_mot = pd.read_csv(PATHS['normalization_dir'] / "normalized_motion.csv")
print(df_mot.describe())
df_mot.describe().round(4).to_csv(PATHS['describe_dir'] / "motion_describe.csv")