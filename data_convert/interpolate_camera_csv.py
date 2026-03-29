#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'
INPUT_DIR = DATA_ROOT / 'camera_csv'
OUTPUT_DIR = DATA_ROOT / 'camera_interpolated'

# 自作ユーティリティのインポート用パス通し
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from interpolate import interpolate_bezier_frame
except ImportError:
    print("エラー: utils/interpolate.py が見つかりません。パス設定を確認してください。")
    sys.exit(1)

def interpolate_camera_frames(input_csv: Path, output_csv: Path):
    df = pd.read_csv(input_csv)
    df = df.sort_values('frame')
    
    # カラム名をCSVの実態に合わせる (R_x1などはカメラ回転共通)
    param_groups = {
        'pos_x': ['X_x1', 'X_y1', 'X_x2', 'X_y2'],
        'pos_y': ['Y_x1', 'Y_y1', 'Y_x2', 'Y_y2'],
        'pos_z': ['Z_x1', 'Z_y1', 'Z_x2', 'Z_y2'],
        'rot_x': ['R_x1', 'R_y1', 'R_x2', 'R_y2'],
        'rot_y': ['R_x1', 'R_y1', 'R_x2', 'R_y2'], # カメラは回転1系統
        'rot_z': ['R_x1', 'R_y1', 'R_x2', 'R_y2'],
        'distance': ['L_x1', 'L_y1', 'L_x2', 'L_y2'],
        'fov': ['V_x1', 'V_y1', 'V_x2', 'V_y2']
    }
    
    result_frames = []
    
    for i in range(len(df) - 1):
        frame_start = df.iloc[i].to_dict()
        frame_end = df.iloc[i + 1].to_dict()
        
        start_f = int(frame_start['frame'])
        end_f = int(frame_end['frame'])
        
        # 始点を追加
        result_frames.append(frame_start)
        
        for f_num in range(start_f + 1, end_f):
            t = (f_num - start_f) / (end_f - start_f)
            
            # 正しい順序で渡す
            new_frame = interpolate_bezier_frame(frame_start, frame_end, t, param_groups)
            
            new_frame['frame'] = f_num
            result_frames.append(new_frame)
    
    if not df.empty:
        result_frames.append(df.iloc[-1].to_dict())
    
    result_df = pd.DataFrame(result_frames).drop_duplicates(subset=['frame'])
    result_df = result_df.sort_values('frame').reset_index(drop=True)
    
    # 型と精度の最終調整
    float_cols = result_df.select_dtypes(include=[np.float64, np.float32]).columns
    result_df[float_cols] = result_df[float_cols].round(6)
    
    result_df.to_csv(output_csv, index=False, float_format='%.6f')
    return len(df), len(result_df)

def main():
    """未処理のカメラCSVを補間"""
    print(f"[Step 3] Camera CSV Interpolation 開始")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(INPUT_DIR.glob('*.csv'))
    if not csv_files:
        print(f"処理対象のCSVが見つかりません: {INPUT_DIR}")
        return

    processed = 0
    skipped = 0

    for csv_file in csv_files:
        output_path = OUTPUT_DIR / csv_file.name
        
        # --- 既存チェック ---
        if output_path.exists():
            skipped += 1
            continue
            
        try:
            old_count, new_count = interpolate_camera_frames(csv_file, output_path)
            print(f"補間完了: {csv_file.name} ({old_count}f -> {new_count}f)")
            processed += 1
        except Exception as e:
            print(f"エラー ({csv_file.name}): {e}")

    print(f"\n処理結果: 新規補間 {processed} 件 / スキップ {skipped} 件")

if __name__ == "__main__":
    main()