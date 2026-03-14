#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
機械学習用データ前処理パイプライン

このスクリプトは、VMDファイルから機械学習用のデータを生成するための一連の前処理を実行します。
以下の処理を順番に実行します：
1. カメラVMDファイルをCSVに変換
2. カメラCSVの補間処理
3. モデルVMDファイルをCSVに変換
4. モーションCSVの補間処理
5. カメラワークラベルの付与
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Optional

# プロジェクトのルートディレクトリを設定
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'  # 1つ上の階層のdataフォルダを指す
PYTHON = sys.executable  # 現在のPythonインタプリタを使用

# 各スクリプトのパス
SCRIPTS = {
    'camera_vmd_to_csv': PROJECT_ROOT / 'camera_vmd_to_csv.py',
    'interpolate_camera_csv': PROJECT_ROOT / 'interpolate_camera_csv.py',
    'model_vmd_to_csv': PROJECT_ROOT / 'model_vmd_to_csv.py',
    'interpolate_motion_csv': PROJECT_ROOT / 'interpolate_motion_csv.py',
    'camera_label': PROJECT_ROOT / 'camera_label.py',
    'label_check': PROJECT_ROOT / 'label_check.py',
    'normalize': PROJECT_ROOT / 'normalize.py'
}

# ディレクトリパスの設定
PATHS = {
    'vmd_dir': DATA_ROOT / 'vmd',
    'camera_vmd_dir': DATA_ROOT / 'camera_vmd',
    'camera_csv_dir': DATA_ROOT / 'camera_csv',
    'camera_interpolated_dir': DATA_ROOT / 'camera_interpolated',
    'motion_vmd_dir': DATA_ROOT / 'motion_vmd',
    'motion_csv_dir': DATA_ROOT / 'motion_csv',
    'motion_wide_dir': DATA_ROOT / 'motion_wide',
    'wav_csv_dir': DATA_ROOT / 'wav_csv',
    'label_csv_dir': DATA_ROOT / 'label_csv',
    'normalization_params_dir': DATA_ROOT / 'normalization_params',
    'analysis_result_dir': PROJECT_ROOT / 'analysis_result'  # 出力ディレクトリはスクリプトと同じ階層に
}

# dataforML.py の run_script 関数に wav_to_csv の処理を追加
def run_script(script_name: str) -> bool:
    """スクリプトを実行する"""
    try:
        if script_name == 'wav_to_csv':
            # wav_to_csvの場合は直接インポートして実行
            from wav_to_csv import main as wav_to_csv_main
            wav_to_csv_main()
            return True
        else:
            # 既存のスクリプト実行処理
            if script_name not in SCRIPTS:
                print(f"エラー: 不明なスクリプト名: {script_name}")
                return False
                
            script_path = SCRIPTS[script_name]
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',  # 明示的にUTF-8を指定
                errors='replace'   # デコードエラーを置換文字で処理
            )
            
            if result.returncode != 0:
                print(f"エラー: {script_name} の実行に失敗しました")
                print(f"エラー出力: {result.stderr}")
                return False
                
            print(result.stdout)
            return True
    except Exception as e:
        print(f"エラー: {script_name} の実行中に例外が発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def ensure_directory(path: Path) -> bool:
    """ディレクトリが存在することを確認し、なければ作成する"""
    
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"ディレクトリの作成に失敗しました: {path}")
        print(f"エラー: {str(e)}")
        return False

def setup_directories() -> bool:
    """必要なディレクトリをセットアップする"""
    print("\nディレクトリをセットアップしています...")
    for name, path in PATHS.items():
        if not ensure_directory(path):
            return False
        print(f"✓ {name}: {path}")
    return True

def clean_directories() -> bool:
    """中間生成物のディレクトリを空にする"""
    print("\n中間ディレクトリをクリーンアップしています...")
    dirs_to_clean = [
        PATHS['camera_csv_dir'],
        PATHS['camera_interpolated_dir'],
        PATHS['motion_csv_dir'],
        PATHS['motion_wide_dir'],
        PATHS['analysis_result_dir'],  # 追加
        PATHS['normalization_params_dir'],  # 追加
        PATHS['label_csv_dir']  # 追加
    ]
    
    for dir_path in dirs_to_clean:
        try:
            if dir_path.exists():
                # ディレクトリ内のファイルを削除
                for item in dir_path.glob('*'):
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            import shutil
                            shutil.rmtree(item)
                        print(f"削除: {item}")
                    except Exception as e:
                        print(f"警告: {item} の削除に失敗しました: {str(e)}")
                print(f"✓ クリーンアップ完了: {dir_path}")
            else:
                print(f"✓ ディレクトリが存在しないためスキップ: {dir_path}")
        except Exception as e:
            print(f"エラー: {dir_path} のクリーンアップに失敗しました: {str(e)}")
            return False
    return True

def ensure_hinan_dir(base_dir: Path) -> Path:
    """hinanディレクトリが存在することを確認し、なければ作成する"""
    hinan_dir = base_dir / 'hinan'
    hinan_dir.mkdir(exist_ok=True)
    return hinan_dir

def check_and_organize_vmd_files() -> bool:
    """VMDファイルの整合性をチェックし、必要な処理を行う
    
    Returns:
        bool: 処理を続行できる場合はTrue、続行できない場合はFalse
    """
    print("\nVMDファイルの整合性をチェックしています...")
    
    # ディレクトリパスを取得
    vmd_dir = PATHS['vmd_dir']
    camera_vmd_dir = PATHS['camera_vmd_dir']
    motion_vmd_dir = PATHS['motion_vmd_dir']
    # hinan_dir = ensure_hinan_dir(vmd_dir)
    
    # ディレクトリが存在することを確認
    if not camera_vmd_dir.exists() or not motion_vmd_dir.exists():
        print(f"エラー: 必要なディレクトリが見つかりません: {camera_vmd_dir} または {motion_vmd_dir}")
        return False
    
    # 各ディレクトリのVMDファイルを取得
    camera_vmds = {f.stem.split('_', 1)[1] if '_' in f.stem else f.stem: f 
                  for f in camera_vmd_dir.glob('*.vmd')}
    motion_vmds = {f.stem.split('_', 1)[1] if '_' in f.stem else f.stem: f 
                  for f in motion_vmd_dir.glob('*.vmd')}
    
    # ペアを見つける
    all_files = set(camera_vmds.keys()) | set(motion_vmds.keys())
    valid_pairs = 0
    
    for name in all_files:
        has_camera = name in camera_vmds
        has_motion = name in motion_vmds
        
        if has_camera and has_motion:
            valid_pairs += 1
        else:
            # 片方しかない場合はhinanフォルダに移動
            if has_camera:
                src = camera_vmds[name]
                dst = camera_vmd_dir / 'camera' / src.name
                dst.parent.mkdir(exist_ok=True)
                src.rename(dst)
                print(f"移動: {src} -> {dst}")
            if has_motion:
                src = motion_vmds[name]
                dst = motion_vmd_dir / 'motion' / src.name
                dst.parent.mkdir(exist_ok=True)
                src.rename(dst)
                print(f"移動: {src} -> {dst}")
    
    print(f"\n有効なVMDファイルペア: {valid_pairs}組")
    print(f"不整合なファイルはhinanディレクトリ に移動しました")
    
    if valid_pairs == 0:
        print("エラー: 有効なVMDファイルのペアが見つかりませんでした")
        return False
    
    return True

def synchronize_frame_counts() -> bool:
    """カメラ、モーション、音声のフレーム数を最短に合わせて同期する"""
    try:
        camera_csvs = list(PATHS['camera_interpolated_dir'].glob('*.csv'))
        motion_csvs = list(PATHS['motion_wide_dir'].glob('*.csv'))
        audio_csvs = list(PATHS['wav_csv_dir'].glob('*_audio_features.csv'))

        camera_dict = {f.stem: f for f in camera_csvs}
        motion_dict = {f.stem: f for f in motion_csvs}
        audio_dict = {f.stem.replace('_audio_features', ''): f for f in audio_csvs}

        all_keys = set(camera_dict.keys()) | set(motion_dict.keys()) | set(audio_dict.keys())

        for name in all_keys:
            frames = {}

            # カメラ
            if name in camera_dict:
                try:
                    df = pd.read_csv(camera_dict[name])
                    frames['camera'] = len(df)
                except Exception as e:
                    print(f"警告: {name} のカメラCSV読み込み失敗: {e}")
                    continue

            # モーション
            if name in motion_dict:
                try:
                    df = pd.read_csv(motion_dict[name])
                    frames['motion'] = len(df)
                except Exception as e:
                    print(f"警告: {name} のモーションCSV読み込み失敗: {e}")
                    continue

            # 音声
            if name in audio_dict:
                try:
                    df = pd.read_csv(audio_dict[name])
                    frames['audio'] = len(df)
                except Exception as e:
                    print(f"警告: {name} の音声CSV読み込み失敗: {e}")
                    continue

            if not frames:
                continue

            # ★ 最短フレーム数に合わせる
            min_frames = min(frames.values())

            # 各データを min_frames に切り詰める
            for data_type, csv_path in [
                ('camera', camera_dict.get(name)),
                ('motion', motion_dict.get(name)),
                ('audio', audio_dict.get(name))
            ]:
                if csv_path is None or not csv_path.exists():
                            continue

                try:
                    df = pd.read_csv(csv_path, low_memory=False)

                    if len(df) > min_frames:
                        df = df.iloc[:min_frames]
                        df.to_csv(csv_path, index=False)
                        print(f"✓ {name} の{data_type}データを切り詰めました ({len(df)+1} → {min_frames})")

                except Exception as e:
                    print(f"警告: {name} の{data_type}データ切り詰め失敗: {e}")

        return True

    except Exception as e:
        print(f"フレーム数同期中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_vmd_files() -> bool:
    """VMDファイルを処理するメインのパイプライン"""
    # 0. VMDファイルの整合性チェック
    print("\n0. VMDファイルの整合性をチェックしています...")
    if not check_and_organize_vmd_files():
        print("エラー: VMDファイルの整合性チェックに失敗しました")
        return False
    
    # 1. 中間ディレクトリをクリーンアップ
    print("\n1. 中間ディレクトリをクリーンアップしています...")
    if not clean_directories():
        print("警告: 一部のディレクトリのクリーンアップに失敗しましたが、処理を続行します")
    
    # 2-6. 既存の処理
    steps = [
        ("2. カメラVMDをCSVに変換しています...", 'camera_vmd_to_csv'),
        ("3. カメラCSVを補間しています...", 'interpolate_camera_csv'),
        ("4. モデルVMDをCSVに変換しています...", 'model_vmd_to_csv'),
        ("5. モーションCSVを補間しています...", 'interpolate_motion_csv'),
        ("6. 音声データをCSVに変換しています...", 'wav_to_csv')  # 新しいステップを追加
    ]
    
    for desc, script in steps:
        print(f"\n{desc}")
        if not run_script(script):
            print(f"エラー: {script} の実行に失敗しました")
            return False
    
    # 6.5 フレーム数を同期
    print("\n6.5 カメラCSV・モーションCSV・音声CSVのフレーム数を同期しています...")
    if not synchronize_frame_counts():
        print("警告: フレーム数の同期に失敗しましたが、処理を続行します")
    
    # 7. カメラワークラベルを付与
    print("\n7. カメラワークラベルを付与しています...")
    if not run_script('camera_label'):
        return False
    
    # 8. ラベルデータの統計情報を出力
    print("\n8. ラベルデータの統計情報を出力しています...")
    try:
        from label_check import analyze_label_distribution
        
        output_dir = PATHS['analysis_result_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        analyze_label_distribution(
            csv_dir=str(PATHS['label_csv_dir']),
            output_dir=str(output_dir),
            threshold=0.5,
            make_plots=False
        )
        print(f"✓ 統計情報を {output_dir} に出力しました")

    except Exception as e:
        print(f"エラー: 統計情報の出力中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 9. 特徴量の正規化を実行
    print("\n9. 特徴量の正規化を実行しています...")
    try:
        from normalize import normalize_features
        
        # データのルートディレクトリ
        data_root = Path("data")  # 必要に応じてパスを調整
        
        # 出力ディレクトリ
        output_dir = data_root / "normalization_params"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 正規化を実行（既存のパラメータは上書き）
        normalize_features(
            data_root=data_root,
            output_dir=output_dir,
            overwrite=True
        )
        print("✓ 特徴量の正規化が完了しました")
    except Exception as e:
        print(f"エラー: 特徴量の正規化中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*50)
    print("全ての処理が正常に完了しました！")
    print("="*50)
    return True

def main():
    """メイン関数"""
    print("="*50)
    print("機械学習用データ前処理パイプラインを開始します")
    print("="*50)
    
    # ディレクトリのセットアップ
    if not setup_directories():
        print("エラー: ディレクトリのセットアップに失敗しました")
        sys.exit(1)
    
    # パイプラインを実行
    if process_vmd_files():
        print("\n" + "="*50)
        print("すべての処理が正常に完了しました！")
        print(f"ラベル付きデータ: {PATHS['label_csv_dir']}")
        print(f"統計情報: {PATHS['analysis_result_dir']}")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("エラー: 処理中にエラーが発生しました")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()