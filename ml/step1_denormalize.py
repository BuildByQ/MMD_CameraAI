import pandas as pd
import numpy as np
from pathlib import Path

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent
DIR_INPUT = PROJECT_ROOT.parent / 'predict_02'
DIR_STEP1 = PROJECT_ROOT.parent / 'output' / 'Step1'

def denormalize_position_log(norm_val, ref_distance=24.0):
    """位置情報の対数圧縮を解除する"""
    k = (np.e - 1.0) / ref_distance
    sign = np.sign(norm_val)
    abs_norm = np.abs(norm_val)
    abs_norm = np.clip(abs_norm, 0, 0.9999) 
    log_val = 1.0 / (1.0 - abs_norm) - 1.0
    abs_val = (np.exp(log_val) - 1.0) / k
    return sign * abs_val

def run_denormalize(song_id):
    """
    指定されたsong_idのファイルを読み込み、逆正規化・補完を行って保存する
    """
    input_file = DIR_INPUT / f"prediction_{song_id}.csv"
    
    if not input_file.exists():
        print(f"入力ファイルが見つかりません: {input_file}")
        return

    print(f"--- [Step 1] 逆正規化開始: {song_id} ---")
    
    df = pd.read_csv(input_file)
    df_denorm = pd.DataFrame()
    
    # 1. 基本データの復元
    if 'frame' in df.columns:
        df_denorm['frame'] = df['frame']
    
    # 位置 & 距離
    for axis in ['x', 'y', 'z']:
        col = f'pos_{axis}'
        if col in df.columns:
            df_denorm[col] = df[col].apply(lambda x: denormalize_position_log(x))
    
    if 'distance' in df.columns:
        df_denorm['distance'] = df['distance'].apply(lambda x: denormalize_position_log(x))

    # 回転角度 (sin/cos -> degree)
    for axis in ['x', 'y', 'z']:
        sin_col = f'rot_{axis}_sin'
        cos_col = f'rot_{axis}_cos'
        if sin_col in df.columns and cos_col in df.columns:
            radians = np.arctan2(df[sin_col], df[cos_col])
            df_denorm[f'rot_{axis}'] = np.degrees(radians)

    # FOV
    if 'fov' in df.columns:
        df_denorm['fov'] = (df['fov'] * 124.0) + 1.0

    # 2. 0-4フレームの完全補完 (5フレーム目の値をコピー)
    min_f = df_denorm['frame'].min()
    if min_f > 0:
        first_actual_row = df_denorm.iloc[0].copy()
        missing_rows = []
        for f in range(0, int(min_f)):
            new_row = first_actual_row.copy()
            new_row['frame'] = float(f) # 後の処理に合わせfloat保持
            missing_rows.append(new_row)
        
        df_denorm = pd.concat([pd.DataFrame(missing_rows), df_denorm], ignore_index=True).sort_values('frame')

    # 3. 保存
    DIR_STEP1.mkdir(parents=True, exist_ok=True)
    output_path = DIR_STEP1 / f"step1_denorm_{song_id}.csv"
    
    # floatのまま(1.0等)で出力
    df_denorm.to_csv(output_path, index=False)
    
    print(f"Step 1 完了: {output_path.name}")

if __name__ == "__main__":
    # 単体デバッグ実行
    predict_files = list(DIR_INPUT.glob("prediction_*.csv"))
    for pf in predict_files:
        sid = pf.stem.replace("prediction_", "")
        run_denormalize(sid)