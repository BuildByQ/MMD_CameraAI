import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import glob

# プロジェクトのルートディレクトリを設定
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# ボーン選択の設定をインポート
from data_convert.interpolate_motion_csv import BoneSelector

def get_selected_bones() -> Set[str]:
    """interpolate_motion_csv.pyで選択されるボーンのセットを取得"""
    # BoneSelectorのデフォルト設定を使用
    selector = BoneSelector()
    return set(selector.get_selected_bones())

def analyze_bone_positions(csv_dir: str, selected_bones: Set[str]) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    指定されたディレクトリ内のCSVファイルを分析し、各ボーンの位置情報の使用状況を返す
    
    Args:
        csv_dir: 分析対象のCSVファイルが含まれるディレクトリ
        selected_bones: 選択対象のボーン名のセット
        
    Returns:
        Tuple[DataFrame, Dict]: 集計結果のDataFrameと詳細データの辞書
    """
    # 結果を格納する辞書
    bone_stats = {}
    
    # 位置情報のカラム
    pos_columns = ['pos_x', 'pos_y', 'pos_z']
    
    # CSVファイルの一覧を取得
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # 選択されたボーンのみにフィルタリング
            df = df[df['bone_name'].isin(selected_bones)]
            
            # ボーンごとに処理
            for bone_name, group in df.groupby('bone_name'):
                if bone_name not in bone_stats:
                    bone_stats[bone_name] = {
                        'total_frames': 0,
                        'non_zero_frames': 0,
                        'files_with_bone': 0,
                        'files_with_non_zero': 0,
                        'frame_counts': []
                    }
                
                # フレーム数をカウント
                frame_count = len(group)
                bone_stats[bone_name]['total_frames'] += frame_count
                bone_stats[bone_name]['files_with_bone'] += 1
                
                # 位置情報が0でないフレームをカウント
                non_zero_mask = (group[pos_columns] != 0).any(axis=1)
                non_zero_count = non_zero_mask.sum()
                
                bone_stats[bone_name]['non_zero_frames'] += non_zero_count
                bone_stats[bone_name]['frame_counts'].append({
                    'file': os.path.basename(file_path),
                    'total_frames': frame_count,
                    'non_zero_frames': int(non_zero_count),
                    'non_zero_ratio': float(non_zero_count) / frame_count if frame_count > 0 else 0
                })
                
                if non_zero_count > 0:
                    bone_stats[bone_name]['files_with_non_zero'] += 1
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # 結果をDataFrameに変換
    result_data = []
    for bone_name, stats in bone_stats.items():
        total_frames = stats['total_frames']
        non_zero_frames = stats['non_zero_frames']
        files_with_bone = stats['files_with_bone']
        files_with_non_zero = stats['files_with_non_zero']
        
        result_data.append({
            'bone_name': bone_name,
            'total_frames': total_frames,
            'non_zero_frames': non_zero_frames,
            'non_zero_ratio': non_zero_frames / total_frames if total_frames > 0 else 0,
            'files_with_bone': files_with_bone,
            'files_with_non_zero': files_with_non_zero,
            'files_non_zero_ratio': files_with_non_zero / files_with_bone if files_with_bone > 0 else 0
        })
    
    df_result = pd.DataFrame(result_data)
    df_result = df_result.sort_values('non_zero_ratio', ascending=False)
    
    return df_result, bone_stats

def save_detailed_analysis(bone_stats: Dict, output_dir: str) -> None:
    """
    詳細な分析結果をCSVファイルに保存
    
    Args:
        bone_stats: ボーンの統計情報
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 各ボーンの詳細を保存
    for bone_name, stats in bone_stats.items():
        if not stats['frame_counts']:
            continue
            
        # ファイルごとの詳細
        df_detail = pd.DataFrame(stats['frame_counts'])
        detail_file = os.path.join(output_dir, f"bone_{bone_name}_details.csv")
        df_detail.to_csv(detail_file, index=False, float_format='%.4f')

def main():
    # 選択されたボーンを取得
    selected_bones = get_selected_bones()
    print(f"分析対象のボーン数: {len(selected_bones)}")
    print("選択されたボーン:", ", ".join(sorted(selected_bones)))
    
    # 入出力ディレクトリを設定
    input_dir = os.path.join(project_root, 'data', 'motion_csv')
    output_dir = os.path.join(project_root, 'results', 'bone_position_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nボーンの位置情報を分析中...")
    df_result, bone_stats = analyze_bone_positions(input_dir, selected_bones)
    
    # 結果をCSVに保存
    output_file = os.path.join(output_dir, 'selected_bones_position_analysis.csv')
    df_result.to_csv(output_file, index=False, float_format='%.6f')
    
    # 詳細な分析結果も保存
    save_detailed_analysis(bone_stats, os.path.join(output_dir, 'selected_details'))
    
    print(f"\n分析結果を保存しました: {output_file}")
    
    # 結果のサマリーを表示
    print("\n=== 位置情報の使用状況サマリー ===")
    print(f"総ボーン数: {len(df_result)}")
    print(f"位置情報を使用しているボーン: {len(df_result[df_result['non_zero_ratio'] > 0])}")
    print(f"位置情報を使用していないボーン: {len(df_result[df_result['non_zero_ratio'] == 0])}")
    
    # 位置情報を使用しているボーンの上位10件を表示
    print("\n=== 位置情報をよく使用するボーン（上位10件）===")
    print(df_result[df_result['non_zero_ratio'] > 0].head(10)[
        ['bone_name', 'non_zero_ratio', 'files_non_zero_ratio', 'files_with_bone']
    ].to_string(index=False))

if __name__ == "__main__":
    main()