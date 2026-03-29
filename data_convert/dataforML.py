#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import subprocess
from pathlib import Path
import ml_utils  # ユーティリティ関数をインポート

# --- パス設定 ---
SCRIPTS_DIR = Path(__file__).parent

def run_step(script_name: str, description: str):
    """
    外部スクリプトを実行する共通インターフェース。
    例外的な直接インポート(wav_to_csv等)もここで吸収する。
    """
    print(f"\n{description}...")
    
    try:
        if script_name == 'wav_to_csv':
            from wav_to_csv import main as wav_main
            wav_main()
        else:
            script_path = SCRIPTS_DIR / f"{script_name}.py"
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if result.returncode != 0:
                print(f"エラー: {script_name} の実行に失敗しました\n{result.stderr}")
                sys.exit(1)
            print(result.stdout)
            
        print(f"完了: {script_name}")
    except Exception as e:
        print(f"例外発生 ({script_name}): {str(e)}")
        sys.exit(1)

def main():
    print("="*50)
    print("機械学習用データ前処理パイプライン開始")
    print("="*50)

    # --- 1. 準備・整合性チェック ---
    ml_utils.setup_directories()           # フォルダ作成
    ml_utils.check_vmd_integrity()         # VMDペアチェック & 隔離
    # ml_utils.clean_intermediate_files()  # 必要に応じてコメントアウト解除

    # --- 2. 変換フェーズ (VMD -> CSV) ---
    run_step('camera_vmd_to_csv',    "Step 1: カメラVMDをCSVに変換")
    run_step('model_vmd_to_csv',     "Step 2: モデルVMDをCSVに変換")
    run_step('wav_to_csv',           "Step 3: 音声データをCSVに変換")

    # --- 3. 補間フェーズ ---
    run_step('interpolate_camera_csv', "Step 4: カメラCSVを補間")
    run_step('interpolate_motion_csv', "Step 5: モーションCSVを補間")

    # --- 4. 同期・加工フェーズ ---
    ml_utils.sync_frame_counts()           # 最短フレームに切り詰め
    run_step('camera_label',           "Step 6: カメラワークラベル付与")

    # --- 5. 分析・正規化フェーズ ---
    ml_utils.run_label_analysis()          # Step 7: 統計出力
    ml_utils.run_normalization()           # Step 8: 特徴量正規化

    print("\n" + "="*50)
    print("全ての処理が正常に完了しました！")
    print("="*50)

if __name__ == "__main__":
    main()