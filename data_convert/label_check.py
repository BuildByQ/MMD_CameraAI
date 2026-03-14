import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
# 日本語フォントの設定
import matplotlib as mpl
import matplotlib.font_manager as fm

# 日本語フォントの設定
font_path = None
# 利用可能な日本語フォントを探す
for font in fm.findSystemFonts():
    if 'meiryo' in str(font).lower() or 'ms gothic' in str(font).lower() or 'yugothic' in str(font).lower():
        font_path = font
        break

if font_path:
    # フォントを設定
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['font.family'] = font_prop.get_name()
    mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
else:
    # 日本語フォントが見つからない場合は警告を非表示にする
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

def analyze_label_distribution(
    csv_dir: str, 
    output_dir: Optional[str] = None,
    threshold: float = 0.5,
    make_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    # 関数の実装...
    """ラベルCSVファイルの分布を分析する
    
    Args:
        csv_dir: ラベルCSVが格納されているディレクトリ
        output_dir: グラフや統計情報を保存するディレクトリ（Noneの場合は保存しない）
    
    Returns:
        Dict[str, pd.DataFrame]: ファイル名をキー、統計情報DataFrameを値とする辞書
    """
    csv_dir_path = Path(csv_dir)
    csv_files = list(csv_dir_path.glob('*.csv'))
    
    if not csv_files:
        print(f"警告: {csv_dir} にCSVファイルが見つかりませんでした")
        return {}
    
    results = {}
    all_dfs = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'frame' in df.columns:
                df = df.drop(columns=['frame'])
            
            # 統計情報を計算
            stats = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose()
            stats['non_zero_ratio'] = (df > 0).mean()
            stats['max_frame'] = df.idxmax()
            stats['file'] = csv_file.name
            
            # 閾値を超えるフレーム数を計算
            thresholds = [0.25, 0.5, 0.75]
            for threshold in thresholds:
                stats[f'frames_above_{int(threshold*100)}%'] = (df > threshold).sum()
                stats[f'ratio_above_{int(threshold*100)}%'] = (df > threshold).mean()
            
            # 統計情報をコンソールに表示
            print(f"\n=== {csv_file.name} の統計情報 ===")
            display_columns = ['count', 'mean', 'std', 'min', 'max', 'non_zero_ratio']
            for t in thresholds:
                display_columns.extend([f'frames_above_{int(t*100)}%', f'ratio_above_{int(t*100)}%'])
            print(stats[display_columns])
            
            results[csv_file.name] = stats
            all_dfs.append(df)
            
            # グラフをプロット
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                
                # グラフのプロット
                plt.figure(figsize=(12, 6))
                for col in df.columns:
                    plt.plot(df.index, df[col], label=col)
                
                plt.title(f'Label Distribution: {csv_file.name}')
                plt.xlabel('Frame')
                plt.ylabel('Score')
                plt.legend()
                plt.grid(True)
                
                output_path = output_dir_path / f'{csv_file.stem}_distribution.png'
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
                
                # 個別の統計情報をCSVで保存
                stats_output_path = output_dir_path / f'{csv_file.stem}_stats.csv'
                stats.to_csv(stats_output_path, float_format='%.4f')
                print(f"統計情報を保存しました: {stats_output_path}")
                
        except Exception as e:
            print(f"エラー: {csv_file} の処理中にエラーが発生しました: {e}")
            continue
    
    # 全CSVを結合した統計情報を計算
    if all_dfs and output_dir:
        try:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_stats = combined_df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose()
            combined_stats['non_zero_ratio'] = (combined_df > 0).mean()
            
            # 閾値を超えるフレーム数を計算（全ファイル結合後）
            for threshold in thresholds:
                combined_stats[f'frames_above_{int(threshold*100)}%'] = (combined_df > threshold).sum()
                combined_stats[f'ratio_above_{int(threshold*100)}%'] = (combined_df > threshold).mean()
            
            print("\n=== 全ファイルを結合した統計情報 ===")
            display_columns = ['count', 'mean', 'std', 'min', 'max', 'non_zero_ratio']
            for t in thresholds:
                display_columns.extend([f'frames_above_{int(t*100)}%', f'ratio_above_{int(t*100)}%'])
            print(combined_stats[display_columns])
            
            # 結合した統計情報をCSVで保存
            output_dir_path = Path(output_dir)
            combined_stats_path = output_dir_path / 'combined_stats.csv'
            combined_stats.to_csv(combined_stats_path, float_format='%.4f')
            print(f"\n全ファイルを結合した統計情報を保存しました: {combined_stats_path}")
            
            # 全ファイルの統計情報を1つのCSVにまとめて保存
            all_stats = pd.concat(results.values())
            all_stats_path = output_dir_path / 'all_stats.csv'
            all_stats.to_csv(all_stats_path, float_format='%.4f')
            print(f"全ファイルの統計情報を保存しました: {all_stats_path}")
            
        except Exception as e:
            print(f"全ファイルの統計情報の計算中にエラーが発生しました: {e}")
    
    return results

def find_high_score_frames(csv_path: str, 
                          label: Optional[str] = None, 
                          threshold: float = 0.5, 
                          top_n: int = 10) -> pd.DataFrame:
    """指定したラベルのスコアが高いフレームを検索する
    
    Args:
        csv_path: ラベルCSVのパス
        label: 対象のラベル名（Noneの場合は全ラベル）
        threshold: スコアの閾値
        top_n: 上位何件を取得するか
    
    Returns:
        pd.DataFrame: 検索結果のDataFrame
    """
    try:
        df = pd.read_csv(csv_path)
        
        if 'frame' not in df.columns:
            df['frame'] = df.index
            
        if label and label in df.columns:
            high_scores = df[df[label] >= threshold]
            return high_scores.nlargest(top_n, label)
        
        elif label is None:
            results = []
            for col in df.columns:
                if col != 'frame':
                    high_scores = df[df[col] >= threshold].nlargest(top_n, col)
                    results.append(high_scores)
            return pd.concat(results).sort_values('frame')
            
    except Exception as e:
        print(f"エラー: {csv_path} の処理中にエラーが発生しました: {e}")
        return pd.DataFrame()
    
    return pd.DataFrame()

def interactive_analysis(csv_dir: str):
    """対話的にラベルを分析する"""
    csv_dir_path = Path(csv_dir)
    csv_files = list(csv_dir_path.glob('*.csv'))
    
    if not csv_files:
        print(f"警告: {csv_dir} にCSVファイルが見つかりませんでした")
        return
    
    print("\n利用可能なCSVファイル:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file.name}")
    
    try:
        choice = int(input("\n分析するファイル番号を入力してください (0: すべて): "))
        if choice < 0 or choice > len(csv_files):
            print("無効な選択です")
            return
            
        if choice == 0:
            # 全ファイルを分析
            output_dir = input("グラフの出力先ディレクトリを入力してください (デフォルト: label_analysis): ")
            output_dir = output_dir or "label_analysis"
            analyze_label_distribution(csv_dir, output_dir)
            print(f"分析結果を {output_dir} に保存しました")
        else:
            # 単一ファイルを分析
            selected_file = csv_files[choice - 1]
            print(f"\n{selected_file.name} を分析中...")
            
            # ラベル一覧を表示
            df = pd.read_csv(selected_file)
            labels = [col for col in df.columns if col != 'frame']
            
            print("\n利用可能なラベル:")
            for i, label in enumerate(labels, 1):
                print(f"{i}. {label}")
            
            label_choice = input("\nラベル番号を入力してください (Enter: 全ラベル): ")
            
            if label_choice:
                try:
                    label_idx = int(label_choice) - 1
                    if 0 <= label_idx < len(labels):
                        label = labels[label_idx]
                        threshold = float(input(f"スコアの閾値を入力 (0.0-1.0, デフォルト: 0.5): ") or "0.5")
                        top_n = int(input(f"表示する上位件数を入力 (デフォルト: 10): ") or "10")
                        
                        result = find_high_score_frames(selected_file, label, threshold, top_n)
                        print(f"\n{label} のスコアが高いフレーム (閾値: {threshold}):")
                        print(result.to_string(index=False))
                    else:
                        print("無効なラベル番号です")
                except ValueError:
                    print("無効な入力です")
            else:
                # 全ラベルを表示
                threshold = float(input("スコアの閾値を入力 (0.0-1.0, デフォルト: 0.5): ") or "0.5")
                top_n = int(input("表示する上位件数を入力 (デフォルト: 10): ") or "10")
                
                result = find_high_score_frames(selected_file, None, threshold, top_n)
                print(f"\n全ラベルのスコアが高いフレーム (閾値: {threshold}):")
                print(result.to_string(index=False))
                
    except ValueError:
        print("無効な入力です")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ラベルCSVを分析するツール')
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')
    
    # 分布分析コマンド
    dist_parser = subparsers.add_parser('dist', help='ラベルの分布を分析する')
    dist_parser.add_argument('csv_dir', help='ラベルCSVが格納されているディレクトリ')
    dist_parser.add_argument('--output-dir', default='label_analysis', 
                           help='グラフの出力先ディレクトリ (デフォルト: label_analysis)')
    
    # 高スコア検索コマンド
    find_parser = subparsers.add_parser('find', help='高スコアのフレームを検索する')
    find_parser.add_argument('csv_file', help='ラベルCSVファイルのパス')
    find_parser.add_argument('--label', help='対象のラベル名 (指定しない場合は全ラベル)')
    find_parser.add_argument('--threshold', type=float, default=0.5,
                           help='スコアの閾値 (デフォルト: 0.5)')
    find_parser.add_argument('--top-n', type=int, default=10,
                           help='表示する上位件数 (デフォルト: 10)')
    
    # インタラクティブモード
    subparsers.add_parser('interactive', help='対話モードで分析する')
    
    args = parser.parse_args()
    
    if args.command == 'dist':
        analyze_label_distribution(args.csv_dir, args.output_dir)
    elif args.command == 'find':
        result = find_high_score_frames(
            args.csv_file, 
            args.label, 
            args.threshold, 
            args.top_n
        )
        print(result.to_string(index=False))
    elif args.command == 'interactive':
        csv_dir = input("ラベルCSVが格納されているディレクトリを入力してください: ")
        interactive_analysis(csv_dir)
    else:
        parser.print_help()
