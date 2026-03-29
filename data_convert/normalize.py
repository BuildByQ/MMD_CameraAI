import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

def normalize_features(
    data_root: Union[str, Path],
    output_dir: Union[str, Path],
    overwrite: bool = False
) -> None:
    """
    特徴量の正規化を実行する（カメラ、モーション、音声の特徴量を含む）
    
    Args:
        data_root: データのルートディレクトリ
        output_dir: 出力先ディレクトリ
        overwrite: 既存の正規化パラメータを上書きするかどうか
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 正規化パラメータを読み込むか新規作成
    params_file = output_dir / 'normalization_params.json'
    if params_file.exists() and not overwrite:
        with open(params_file, 'r', encoding='utf-8') as f:
            norm_params = json.load(f)

        # ★ dict → Series に戻す
        for key in ['camera', 'motion', 'audio']:
            if norm_params[key]['mean'] is not None:
                norm_params[key]['mean'] = pd.Series(norm_params[key]['mean'])
                norm_params[key]['std']  = pd.Series(norm_params[key]['std'])
    else:
        norm_params = {
            'camera': _calculate_normalization_params(data_root.parent, 'camera'),
            'motion': _calculate_normalization_params(data_root.parent, 'motion'),
            'audio': _calculate_normalization_params(data_root.parent, 'audio')
        }

        # ★ JSON 保存用に dict に変換する
        json_safe_params = {
            k: {
                'mean': v['mean'].to_dict() if hasattr(v['mean'], 'to_dict') else v['mean'],
                'std':  v['std'].to_dict()  if hasattr(v['std'], 'to_dict')  else v['std']
            }
            for k, v in norm_params.items()
        }

        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(json_safe_params, f, indent=2, ensure_ascii=False)
    
    # 各データの読み込みと正規化
    camera_data = None
    motion_data = None
    audio_data = None
    label_data = None

    synced_root = data_root
    
    try:
        # --- 【ステップ1：ラベル読み込みの独立化】 ---
        # ラベルデータは正規化関数を通さず、単に読み込むだけにします
        label_dir = data_root.parent / 'label_csv'
        if label_dir.exists():
            # _load_and_combine_data は単なるロードなのでこれだけでOK
            label_data = _load_and_combine_data(label_dir, 'label')
            print(f"✓ ラベルデータのロード完了: {len(label_data)}行")

        # カメラデータの読み込みと正規化
        camera_dir = synced_root / 'camera'
        if camera_dir.exists():
            camera_df = _load_and_combine_data(camera_dir, 'camera')
            # ここでカメラ側だけ正規化を適用
            normalized_camera_df = apply_camera_normalization(camera_df, norm_params['camera'])
            
            camera_data = {
                'data': normalized_camera_df,
                'params': norm_params['camera'],
                'output_dir': output_dir
            }
            
            # --- 【ステップ2：結合タイミングの整理】 ---
            # 既にラベルが読み込まれていれば、ここでマージ
            if label_data is not None:
                camera_data['data'] = pd.merge(
                    camera_data['data'],
                    label_data,
                    on=['song_id', 'frame'],
                    how='inner'
                )
            
            output_file = camera_data['output_dir'] / 'normalized_camera.csv'
            camera_data['data'].to_csv(output_file, index=False)
            print(f"✓ 正規化済みcamera(+ラベル)保存完了: {output_file}")
        
        # モーションデータの読み込みと正規化
        motion_dir = synced_root / 'motion'
        if motion_dir.exists():
            motion_df = _load_and_combine_data(motion_dir, 'motion')
            motion_data = {
                'data': apply_motion_normalization(motion_df, norm_params['motion']),
                'params': norm_params['motion'],
                'output_dir': output_dir
            }
            motion_data['output_dir'].mkdir(parents=True, exist_ok=True)
            # 正規化済みデータを保存
            output_file = motion_data['output_dir'] / 'normalized_motion.csv'
            motion_data['data'].to_csv(output_file, index=False)
            print(f"✓ 正規化済みmotion保存完了: {output_file}")
            
        # 音声データの正規化
        audio_data = apply_audio_normalization(synced_root, output_dir, norm_params['audio'])
        
        # データの統合
        _integrate_data(camera_data, motion_data, audio_data, label_data, output_dir)
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def apply_audio_normalization(
    data_root: Union[str, Path],
    output_dir: Union[str, Path],
    norm_params: Dict[str, Any],
    feature_order: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    音声データに正規化を適用する改善版。
    - std=0 対策
    - NaN / inf の除去
    - 列順の固定
    - 正規化後の品質チェック
    """

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = data_root / 'audio'
    if not audio_dir.exists():
        print("警告: 音声データディレクトリが見つかりません")
        return None

    audio_files = list(audio_dir.glob('*_audio_features.csv'))
    if not audio_files:
        print("警告: 音声特徴量CSVファイルが見つかりません")
        return None

    print("\n音声データの正規化を実行します...")
    all_data = []

    mean = norm_params['mean']
    std = norm_params['std'].replace(0, 1e-8)  # ★ 0除算対策

    for audio_file in tqdm(audio_files, desc="音声データを処理中"):
        try:
            df = pd.read_csv(audio_file)

            meta_cols = ['song_id', 'frame']
            meta_data = df[meta_cols].copy()

            feature_cols = [col for col in df.columns if col not in meta_cols]
            features = df[feature_cols].astype('float64')

            # ★ 正規化
            normalized_features = (features - mean[feature_cols]) / std[feature_cols]

            # ★ NaN / inf を除去
            normalized_features = normalized_features.replace([np.inf, -np.inf], np.nan)
            normalized_features = normalized_features.fillna(0.0)

            # ★ 列順を固定（config.json の audio_features を使う）
            if feature_order is not None:
                normalized_features = normalized_features[feature_order]

            # 結合
            normalized_df = pd.concat([meta_data, normalized_features], axis=1)

            all_data.append(normalized_df)

        except Exception as e:
            print(f"エラー: {audio_file} の処理中にエラーが発生しました: {str(e)}")
            continue

    if not all_data:
        print("エラー: 有効な音声データが見つかりませんでした")
        return None

    combined_data = pd.concat(all_data, ignore_index=True)

    # ★ 正規化後の品質チェック
    print("\n=== 正規化後の統計量（サンプル） ===")
    print(combined_data.iloc[:, 2:].describe().loc[['mean', 'std']])

    combined_data.to_csv(output_dir / 'normalized_audio.csv', index=False)

    return {
        'data': combined_data,
        'params': norm_params,
        'output_dir': output_dir
    }

def _calculate_normalization_params(
    data_root: Path,
    data_type: str
) -> Dict[str, Any]:
    """
    正規化パラメータ（mean, std）を計算する改善版。
    音声データの std=0 対策、dtype 統一、Series 化を行う。
    """

    # ディレクトリ設定
    if data_type == 'camera':
        data_dir = data_root / 'camera_interpolated'
        drop_cols = ['frame']
    elif data_type == 'motion':
        data_dir = data_root / 'motion_wide'
        drop_cols = ['frame']
    elif data_type == 'audio':
        data_dir = data_root / 'wav_csv'
        drop_cols = ['song_id', 'frame']
    else:
        raise ValueError(f"サポートされていないデータタイプ: {data_type}")

    if not data_dir.exists():
        return {'mean': None, 'std': None}

    # CSV をすべて読み込む
    all_data = []
    for file in data_dir.glob('*.csv'):
        if data_type == 'audio' and not file.name.endswith('_audio_features.csv'):
            continue

        df = pd.read_csv(file)
        if df.empty:
            continue

        # メタデータを除外
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        # dtype を float64 に統一（音声で混ざることがある）
        df = df.astype('float64')

        all_data.append(df)

    if not all_data:
        return {'mean': None, 'std': None}

    # 結合
    combined = pd.concat(all_data, ignore_index=True)

    # 平均・標準偏差を計算（Series）
    mean = combined.mean()
    std = combined.std()

    # std=0 の列を 1e-8 に置き換え（0除算防止）
    std = std.replace(0, 1e-8)

    return {
        'mean': mean,
        'std': std
    }

def _integrate_data(
    camera_data: Optional[Dict[str, Any]],
    motion_data: Optional[Dict[str, Any]],
    audio_data: Optional[Dict[str, Any]],
    label_data: Optional[Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    正規化されたデータを統合する
    
    Args:
        camera_data: カメラデータ
        motion_data: モーションデータ
        audio_data: 音声データ
        output_dir: 出力先ディレクトリ
    """
    if camera_data is None or motion_data is None:
        print("エラー: カメラデータまたはモーションデータが不足しています")
        return
    
    print("\nデータを統合しています...")
    
    try:
        # カメラデータとモーションデータを結合
        camera_df = camera_data['data']
        motion_df = motion_data['data']

        # ラベルデータとカメラデータを結合
        merged = pd.merge(
            camera_df,
            label_data,
            on=['song_id', 'frame'],
            how='inner'
        )
        
        # 共通のカラム名を確認
        common_columns = set(camera_df.columns) & set(motion_df.columns)
        if 'song_id' not in common_columns or 'frame' not in common_columns:
            raise ValueError("song_id または frame カラムが見つかりません")
        
        # 音声データがある場合は結合
        if audio_data is not None:
            audio_df = audio_data['data']
            # 音声データのカラム名を変更（重複を避けるため）
            audio_df = audio_df.rename(columns={
                col: f"audio_{col}" for col in audio_df.columns 
                if col not in ['song_id', 'frame']
            })
            
            # 3つのデータフレームを結合
            merged = pd.merge(
                camera_df,
                motion_df,
                on=['song_id', 'frame'],
                how='inner'
            )
            
            merged = pd.merge(
                merged,
                audio_df,
                on=['song_id', 'frame'],
                how='inner'
            )
        else:
            # 音声データがない場合はカメラとモーションのみを結合
            merged = pd.merge(
                camera_df,
                motion_df,
                on=['song_id', 'frame'],
                how='inner'
            )
        
        # 結果を保存
        output_file = output_dir / 'normalized_data.csv'
        merged.to_csv(output_file, index=False)
        print(f"統合データを保存しました: {output_file}")
        print(f"合計 {len(merged)} 行のデータを処理しました")
        
    except Exception as e:
        print(f"データの統合中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()

def _load_and_combine_data(
    data_dir: Path,
    data_type: str
) -> pd.DataFrame:
    """
    指定されたディレクトリからデータを読み込み、1つのDataFrameに結合する
    
    Args:
        data_dir: データが含まれるディレクトリ
        data_type: 'camera' または 'motion'
        
    Returns:
        pd.DataFrame: 結合されたデータ
    """
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        raise ValueError(f"{data_dir} にCSVファイルが見つかりませんでした")
    
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # ソングIDを追加（ファイル名から拡張子を除く）
            song_id = csv_file.stem
            df['song_id'] = song_id
            all_data.append(df)
        except Exception as e:
            print(f"警告: {csv_file.name} の読み込み中にエラーが発生しました: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("有効なデータが見つかりませんでした")
    
    # 全データを結合
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def _normalize_camera_data(
    camera_dir: Path,
    output_dir: Path
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """
    カメラデータを正規化する
    
    Args:
        camera_dir: カメラデータが含まれるディレクトリ
        output_dir: 正規化パラメータの出力先ディレクトリ
        
    Returns:
        Dict: 正規化パラメータの辞書
    """
    # 全データを読み込む
    try:
        df = _load_and_combine_data(camera_dir, 'camera')
    except ValueError as e:
        print(f"エラー: {str(e)}")
        return {}
    
    # 正規化パラメータを計算
    stats = {
        'pos_means': {},
        'pos_stds': {},
        'rot_means': {},
        'rot_stds': {}
    }
    
    # 位置情報の列を探す
    pos_columns = [col for col in df.columns if col.endswith(('_x', '_y', '_z')) and not col.startswith('rot_')]
    # 回転情報の列を探す
    rot_columns = [col for col in df.columns if col.startswith('rot_')]
    
    # 位置情報の統計を計算
    for col in pos_columns:
        stats['pos_means'][col] = df[col].mean()
        stats['pos_stds'][col] = df[col].std()
    
    # 回転情報の統計を計算
    for col in rot_columns:
        stats['rot_means'][col] = df[col].mean()
        stats['rot_stds'][col] = df[col].std()
    
    # 正規化パラメータを保存
    params_file = output_dir / 'camera_normalization_params.json'
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    print(f"カメラデータの正規化パラメータを保存しました: {params_file}")
    
    return stats

import numpy as np

def normalize_position_with_log(df, col, ref_distance=30.0):
    """
    対数圧縮＋飽和型の正規化。
    ref_distance に対応する値が 0.5 になるようにスケール調整する。
    """
    # ref_distance を 0.5 にしたい → k = (e - 1) / ref_distance
    k = (np.e - 1.0) / ref_distance

    sign = np.sign(df[col])
    abs_val = np.abs(df[col])

    # スケールをかけてから log1p
    log_val = np.log1p(k * abs_val)

    # -1〜1 に写像
    norm = 1.0 - 1.0 / (1.0 + log_val)
    df_norm_col = sign * norm

    return df_norm_col

def apply_camera_normalization(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    カメラデータに正規化を適用する（角度は sin/cos 変換版）

    - 位置: log 正規化（既存の normalize_position_with_log を使用）
    - 回転:
        ・角速度（speed）は従来通り -180〜180 → -1〜1
        ・角度（angle）は sin/cos に変換し、周期性問題を解消
    """

    df_norm = df.copy()

    # -------------------------
    # 位置データの正規化
    # -------------------------
    for axis in ['x', 'y', 'z']:
        col = f'pos_{axis}'
        if col in df.columns:
            df_norm[col] = normalize_position_with_log(df, col, ref_distance=24.0)

    # -------------------------
    # 回転データ（角速度 + sin/cos）
    # -------------------------
    for axis in ['x', 'y', 'z']:
        col = f'rot_{axis}'
        if col not in df.columns:
            continue

        values = df[col].values.astype(float)

        if len(values) > 1:
            # 1. 角速度（speed）: 周期補正 → -1〜1
            raw_diff = np.diff(values, prepend=values[0])
            angle_speed = ((raw_diff + 180) % 360) - 180
            norm_speed = angle_speed / 180.0
            df_norm[f'{col}_speed'] = norm_speed

            # 2. 角度（angle）: sin/cos 変換
            theta = np.deg2rad(values)  # ラジアンに変換
            df_norm[f'{col}_sin'] = np.sin(theta)
            df_norm[f'{col}_cos'] = np.cos(theta)

        else:
            # フレームが1つしかない場合
            df_norm[f'{col}_speed'] = 0.0
            df_norm[f'{col}_sin'] = np.sin(np.deg2rad(values[0]))
            df_norm[f'{col}_cos'] = np.cos(np.deg2rad(values[0]))

    # -------------------------
    # 距離データの正規化
    # -------------------------
    if 'distance' in df.columns:
        df_norm['distance'] = normalize_position_with_log(df, 'distance', ref_distance=24.0)

    # -------------------------
    # FOV の正規化
    # -------------------------
    if 'fov' in df.columns:
        df_norm['fov'] = (df['fov'] - 1) / 124

    return df_norm

def _normalize_motion_data(
    motion_dir: Path,
    output_dir: Path
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """
    モーションデータを正規化する
    
    Args:
        motion_dir: モーションデータが含まれるディレクトリ
        output_dir: 正規化パラメータの出力先ディレクトリ
        
    Returns:
        Dict: 正規化パラメータの辞書
    """
    # 全データを読み込む
    try:
        df = _load_and_combine_data(motion_dir, 'motion')
    except ValueError as e:
        print(f"エラー: {str(e)}")
        return {}
    
    # 統計情報を収集するための辞書
    stats = {
        'pos_means': {},
        'pos_stds': {},
        'rot_means': {},
        'rot_stds': {}
    }
    
    # 位置情報の列を探す
    pos_columns = [col for col in df.columns if col.endswith(('_pos_x', '_pos_y', '_pos_z'))]
    # 回転情報の列を探す
    rot_columns = [col for col in df.columns if col.endswith(('_rot_x', '_rot_y', '_rot_z', '_rot_w'))]
    
    # 位置情報の統計を計算
    for col in pos_columns:
        stats['pos_means'][col] = df[col].mean()
        stats['pos_stds'][col] = df[col].std()
    
    # 回転情報の統計を計算
    for col in rot_columns:
        stats['rot_means'][col] = df[col].mean()
        stats['rot_stds'][col] = df[col].std()
    
    # 正規化パラメータを保存
    params_file = output_dir / 'motion_normalization_params.json'
    with open(params_file, 'w') as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    print(f"モーションデータの正規化パラメータを保存しました: {params_file}")
    
    return stats

def apply_motion_normalization(df: pd.DataFrame, norm_params: Dict[str, Any]) -> pd.DataFrame:
    """
    モーション（ワールド座標）の正規化。
    現在は全てXYZの座標値なので、log正規化に統一。
    """
    df_norm = df.copy()
    
    # 全てのカラム（song_id, frameを除く）に対して処理
    for col in df.columns:
        if col in ['song_id', 'frame']: continue
        
        # ボーンのワールド座標(x, y, z)を対数圧縮
        # カメラ側の ref_distance=24.0 と合わせることで、空間スケールを統一
        df_norm[col] = normalize_position_with_log(df, col, ref_distance=24.0)
    
    return df_norm

def normalize_all_data(
    data_root: Path,
    output_dir: Path
) -> None:
    """
    全データを正規化し、統合する
    
    Args:
        data_root: データのルートディレクトリ
        output_dir: 出力先ディレクトリ
    """
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 統合データを格納するデータフレーム
    all_camera_data = []
    all_motion_data = []
    
    # カメラデータの正規化と統合
    camera_dir = data_root / 'camera_wide'
    if camera_dir.exists():
        print("カメラデータを処理しています...")
        try:
            # 全カメラデータを読み込む
            camera_df = _load_and_combine_data(camera_dir, 'camera')
            all_camera_data.append(camera_df)
            print(f"カメラデータを読み込みました: {len(camera_df)}行")
        except Exception as e:
            print(f"カメラデータの読み込み中にエラーが発生しました: {str(e)}")
    else:
        print(f"警告: {camera_dir} が見つかりませんでした")
    
    # モーションデータの正規化と統合
    motion_dir = data_root / 'motion_wide'
    if motion_dir.exists():
        print("モーションデータを処理しています...")
        try:
            # 全モーションデータを読み込む
            motion_df = _load_and_combine_data(motion_dir, 'motion')
            all_motion_data.append(motion_df)
            print(f"モーションデータを読み込みました: {len(motion_df)}行")
        except Exception as e:
            print(f"モーションデータの読み込み中にエラーが発生しました: {str(e)}")
    else:
        print(f"警告: {motion_dir} が見つかりませんでした")
    
    # データが存在するか確認
    if not all_camera_data or not all_motion_data:
        print("エラー: カメラデータまたはモーションデータが見つかりませんでした")
        return
    
    # データを結合
    camera_df = pd.concat(all_camera_data, ignore_index=True)
    motion_df = pd.concat(all_motion_data, ignore_index=True)
    
    # 正規化パラメータを計算
    print("正規化パラメータを計算しています...")
    
    # カメラデータの正規化パラメータ
    camera_stats = {
        'pos_means': {},
        'pos_stds': {},
        'rot_means': {},
        'rot_stds': {}
    }
    
    # カメラデータの位置情報と回転情報の列を取得
    camera_pos_columns = [col for col in camera_df.columns if col.endswith(('_x', '_y', '_z')) and not col.startswith('rot_')]
    camera_rot_columns = [col for col in camera_df.columns if col.startswith('rot_')]
    
    # カメラデータの統計を計算
    for col in camera_pos_columns:
        camera_stats['pos_means'][col] = camera_df[col].mean()
        camera_stats['pos_stds'][col] = camera_df[col].std()
    
    for col in camera_rot_columns:
        camera_stats['rot_means'][col] = camera_df[col].mean()
        camera_stats['rot_stds'][col] = camera_df[col].std()
    
    # モーションデータの正規化パラメータ
    motion_stats = {
        'pos_means': {},
        'pos_stds': {},
        'rot_means': {},
        'rot_stds': {}
    }
    
    # モーションデータの位置情報と回転情報の列を取得
    motion_pos_columns = [col for col in motion_df.columns if col.endswith(('_pos_x', '_pos_y', '_pos_z'))]
    motion_rot_columns = [col for col in motion_df.columns if col.endswith(('_rot_x', '_rot_y', '_rot_z', '_rot_w'))]
    
    # モーションデータの統計を計算
    for col in motion_pos_columns:
        motion_stats['pos_means'][col] = motion_df[col].mean()
        motion_stats['pos_stds'][col] = motion_df[col].std()
    
    for col in motion_rot_columns:
        motion_stats['rot_means'][col] = motion_df[col].mean()
        motion_stats['rot_stds'][col] = motion_df[col].std()
    
    # 正規化パラメータを保存
    params_file = output_dir / 'normalization_params.json'
    with open(params_file, 'w') as f:
        json.dump({
            'camera': camera_stats,
            'motion': motion_stats
        }, f, indent=2, cls=NumpyEncoder)
    print(f"正規化パラメータを保存しました: {params_file}")
    
    # データを正規化
    print("データを正規化しています...")
    
    # カメラデータの正規化
    for col in camera_pos_columns:
        mean = camera_stats['pos_means'][col]
        std = camera_stats['pos_stds'][col]
        if std > 0:
            camera_df[col] = (camera_df[col] - mean) / std
    
    for col in camera_rot_columns:
        mean = camera_stats['rot_means'][col]
        std = camera_stats['rot_stds'][col]
        if std > 0:
            camera_df[col] = (camera_df[col] - mean) / std
    
    # モーションデータの正規化
    for col in motion_pos_columns:
        mean = motion_stats['pos_means'][col]
        std = motion_stats['pos_stds'][col]
        if std > 0:
            motion_df[col] = (motion_df[col] - mean) / std
    
    for col in motion_rot_columns:
        mean = motion_stats['rot_means'][col]
        std = motion_stats['rot_stds'][col]
        if std > 0:
            motion_df[col] = (motion_df[col] - mean) / std
    
    # カメラデータとモーションデータを結合
    print("カメラデータとモーションデータを結合しています...")
    
    # song_id と frame で結合
    merged_df = pd.merge(
        camera_df,
        motion_df,
        on=['song_id', 'frame'],
        how='inner',  # 両方のデータセットに存在するフレームのみを保持
        suffixes=('_camera', '_motion')
    )
    
    # 結果を保存
    output_file = output_dir / 'normalized_data.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"正規化されたデータを保存しました: {output_file}")
    print(f"合計 {len(merged_df)} 行のデータを処理しました")
    
    print("\n" + "="*50)
    print("データの正規化と統合が完了しました")
    print("="*50)

def _normalize_audio_features(
    audio_dir: Path,
    output_dir: Path,
    sr: int = 22050,
    hop_length: int = 512
) -> Dict[str, Union[float, np.ndarray]]:
    """音声特徴量の正規化"""
    # ここに音声特徴量の正規化処理を実装
    return {
        'mean': 0.0,  # 実際の平均値に置き換え
        'std': 1.0,   # 実際の標準偏差に置き換え
        'sr': sr,
        'hop_length': hop_length
    }

class NumpyEncoder(json.JSONEncoder):
    """NumPy のデータ型を JSON にシリアライズするためのエンコーダ"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

if __name__ == "__main__":
    # デバッグ用のパラメータをここで設定
    data_root = Path("E:\\ML\\MMD_CameraAI_wind\\data")  # 実際のパスに置き換えてください
    output_dir = data_root / "normalization_params"  # None の場合は data_root/normalization_params に保存
    overwrite = True   # 既存のパラメータを上書きするかどうか
    
    # 正規化を実行
    normalize_features(
        data_root=data_root,
        output_dir=output_dir,
        overwrite=overwrite
    )