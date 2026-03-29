import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import normalize

# --- パス解決 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / 'data'
CONFIG_PATH = SCRIPT_DIR / "config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def setup_directories():
    """必要なディレクトリを一括作成"""
    print("ディレクトリをセットアップしています...")
    sub_dirs = [
        'camera_csv', 'camera_interpolated', 
        'motion_csv', 'motion_wide', 
        'wav_csv', 'label_csv', 'normalization_params', 'normalized_data'
    ]
    for d in sub_dirs:
        (DATA_ROOT / d).mkdir(parents=True, exist_ok=True)
    print("✓ セットアップ完了")

def check_vmd_integrity():
    """VMDのペアチェックと不整合ファイルの隔離"""
    print("VMD整合性チェック開始...")
    cam_dir = DATA_ROOT / 'camera_vmd'
    mot_dir = DATA_ROOT / 'motion_vmd'
    hinan_dir = DATA_ROOT / 'vmd_hinan'
    
    cam_files = {f.name for f in cam_dir.glob("*.vmd")}
    mot_files = {f.name for f in mot_dir.glob("*.vmd")}
    
    common = cam_files & mot_files
    only_cam = cam_files - mot_files
    only_mot = mot_files - cam_files
    
    if only_cam or only_mot:
        hinan_dir.mkdir(exist_ok=True)
        for f in only_cam: (cam_dir / f).rename(hinan_dir / f)
        for f in only_mot: (mot_dir / f).rename(hinan_dir / f)
        print(f"不整合ファイルを隔離しました: {len(only_cam) + len(only_mot)}件")
    
    print(f"有効なペア: {len(common)}組")

def sync_frame_counts():
    """
    カメラ(interpolated)、モーション(wide)、音声(wav_csv)のフレーム数を最短に合わせ、
    'data/synced' フォルダへ一括出力する。
    後続の camera_label.py や normalization はこの 'synced' フォルダを参照する。
    """
    print("\nフレーム数を同期（最短切り詰め）しています...")
    
    # 1. パス設定
    src_cam = DATA_ROOT / "camera_interpolated"
    src_mot = DATA_ROOT / "motion_wide"
    src_aud = DATA_ROOT / "wav_csv"
    
    # 出力先を 'data/synced' 内の各サブフォルダに設定
    dst_base = DATA_ROOT / "synced"
    dst_cam = dst_base / "camera"
    dst_mot = dst_base / "motion"
    dst_aud = dst_base / "audio"
    
    # ディレクトリ作成
    for d in [dst_cam, dst_mot, dst_aud]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. ファイルリストの突合せ
    # カメラCSVを基準に、対応するモーションと音声があるか確認
    cam_files = sorted(list(src_cam.glob("*.csv")))
    common_targets = []

    for c_path in cam_files:
        fname = c_path.name  # 例: "dance_01.csv"
        m_path = src_mot / fname
        # 音声はファイル名ルールが違う場合（_audio_features付与等）に対応
        a_path = src_aud / fname.replace(".csv", "_audio_features.csv")
        
        if m_path.exists() and a_path.exists():
            common_targets.append({
                'name': fname,
                'cam': c_path,
                'mot': m_path,
                'aud': a_path
            })

    if not common_targets:
        print("同期対象のペア（カメラ・モーション・音声）が見つかりません。")
        return False

    # 3. 同期処理（最短フレームへの切り詰め）
    for target in tqdm(common_targets, desc="Syncing"):
        try:
            # 各CSVの読み込み
            df_c = pd.read_csv(target['cam'])
            df_m = pd.read_csv(target['mot'])
            df_a = pd.read_csv(target['aud'])
            
            # 最短行数を取得
            min_f = min(len(df_c), len(df_m), len(df_a))
            
            # 切り詰め実行
            df_c_sync = df_c.iloc[:min_f]
            df_m_sync = df_m.iloc[:min_f]
            df_a_sync = df_a.iloc[:min_f]
            
            # 'data/synced' フォルダへ保存
            df_c_sync.to_csv(dst_cam / target['name'], index=False)
            df_m_sync.to_csv(dst_mot / target['name'], index=False)
            df_a_sync.to_csv(dst_aud / target['aud'].name, index=False)
            
        except Exception as e:
            print(f"Error syncing {target['name']}: {e}")
            continue

    print(f"同期完了: {len(common_targets)} ファイルを {dst_base} に出力しました。")
    return True

def run_label_analysis():
    """ラベルの分布を表示（デバッグ用）"""
    label_path = DATA_ROOT / "normalized_data/normalized_label.csv"
    if not label_path.exists(): return
    
    df = pd.read_csv(label_path)
    print("\n--- ラベル出現頻度統計 ---")
    cols = [c for c in df.columns if c.startswith(('bin_', 'event_', 'prox_', 'height_', 'tilt_', 'dyn_'))]
    stats = df[cols].mean().sort_values(ascending=False)
    print(stats)
    print("--------------------------\n")

def run_normalization():
    """
    Step 9: 特徴量の正規化を実行。
    内部で normalize.py の高度なロジックを呼び出す。
    """
    print("\n[Step 9] 高度な特徴量正規化を開始します...")
    
    # パス設定
    # sync_frame_counts() が出力した 'data/synced' をソースにする
    synced_root = DATA_ROOT / "synced"
    output_params_dir = DATA_ROOT / "normalization_params"
    
    try:
        # normalize.py のメイン関数を呼び出し
        normalize.normalize_features(
            data_root=synced_root, 
            output_dir=output_params_dir,
            overwrite=True
        )
        print("正規化とデータの統合が完了しました。")
    except Exception as e:
        print(f"正規化中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

def clean_intermediate_files():
    """必要に応じて中間ファイルを消去"""
    pass