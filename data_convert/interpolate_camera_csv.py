import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import glob

# プロジェクトのルートディレクトリをパスに追加
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
from utils.interpolate import interpolate_bezier_frame

def process_single_file(input_path, output_dir):
    """単一のCSVファイルを処理する関数"""
    # 出力ファイルパスを作成
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    
    # 補間処理を実行
    interpolate_camera_frames(input_path, output_path)

def interpolate_camera_frames(input_csv, output_csv):
    # CSVを読み込む
    df = pd.read_csv(input_csv)
    
    # フレーム番号でソート
    df = df.sort_values('frame')
    
    # 補間パラメータのマッピング
    param_groups = {
        'pos_x': ['X_x1', 'X_y1', 'X_x2', 'X_y2'],
        'pos_y': ['Y_x1', 'Y_y1', 'Y_x2', 'Y_y2'],
        'pos_z': ['Z_x1', 'Z_y1', 'Z_x2', 'Z_y2'],
        'rot_x': ['R_x1', 'R_y1', 'R_x2', 'R_y2'],
        'rot_y': ['R_x1', 'R_y1', 'R_x2', 'R_y2'],
        'rot_z': ['R_x1', 'R_y1', 'R_x2', 'R_y2'],
        'distance': ['L_x1', 'L_y1', 'L_x2', 'L_y2'],
        'fov': ['V_x1', 'V_y1', 'V_x2', 'V_y2']
    }
    
    # 補間用のデータフレームを作成（辞書のリストとして管理）
    result_frames = []
    
    # 各キーフレーム間を処理
    for i in range(len(df) - 1):
        frame_start = df.iloc[i].to_dict()
        frame_end = df.iloc[i + 1].to_dict()
        
        # 現在のキーフレーム間のフレームを取得
        start_frame = int(frame_start['frame'])
        end_frame = int(frame_end['frame'])
        
        # 開始フレームを追加（重複を避けるため、まだ追加されていない場合のみ）
        if not any(f['frame'] == start_frame for f in result_frames):
            result_frames.append(frame_start)
        
        # キーフレーム間のフレーム数を計算
        frame_count = end_frame - start_frame - 1
        if frame_count <= 0:
            continue
            
        # 各フレームに対して補間
        for frame in range(start_frame + 1, end_frame):
            # 補間係数を計算 (0.0 ～ 1.0)
            t = (frame - start_frame) / (end_frame - start_frame)
            
            # 補間を実行
            new_frame = interpolate_bezier_frame(frame_start, frame_end, t, param_groups)
            new_frame['frame'] = frame
            
            # 補間されたフレームを追加
            result_frames.append(new_frame)
    
    # 最後のフレームを追加（まだ追加されていない場合）
    if len(df) > 0:
        last_frame = df.iloc[-1].to_dict()
        if not any(f['frame'] == last_frame['frame'] for f in result_frames):
            result_frames.append(last_frame)
    
    # データフレームに変換してフレーム番号でソート
    result_df = pd.DataFrame(result_frames).sort_values('frame')
    
    # カラムの順序を元のCSVと合わせる
    result_df = result_df[df.columns]
    
    # フロートの精度を揃える
    for col in result_df.columns:
        if result_df[col].dtype == float:
            result_df[col] = result_df[col].round(6)
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # CSVとして保存
    result_df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"補間されたフレームを {output_csv} に保存しました。")
    print(f"元のフレーム数: {len(df)}, 補間後のフレーム数: {len(result_df)}")

def main():
    # 入力ディレクトリと出力ディレクトリを指定
    input_dir = os.path.join(project_root, 'data', 'camera_csv')
    output_dir = os.path.join(project_root, 'data', 'camera_interpolated')
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # CSVファイルの一覧を取得
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"処理対象のCSVファイルが {input_dir} に見つかりませんでした。")
        return
    
    print(f"処理を開始します。{len(csv_files)}個のCSVファイルを処理します。")
    
    # 各CSVファイルを処理
    for csv_file in csv_files:
        print(f"\n処理中: {os.path.basename(csv_file)}")
        process_single_file(csv_file, output_dir)
    
    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()