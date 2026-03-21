import pandas as pd
from pathlib import Path

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent
DIR_STEP1 = PROJECT_ROOT.parent / 'output' / 'Step1'
DIR_STEP2 = PROJECT_ROOT.parent / 'output' / 'Step2'
DIR_LABEL = PROJECT_ROOT.parent / 'predict_01to02'

def run_optimize(song_id):
    """
    指定されたsong_idのファイルを読み込み、間引き処理を行って保存する
    """
    # 1. ファイルの読み込み
    path_denorm = DIR_STEP1 / f"step1_denorm_{song_id}.csv"
    path_label = DIR_LABEL / "director_instruction.csv"
    
    if not path_denorm.exists() or not path_label.exists():
        print(f"ファイルが見つかりません: {song_id}")
        return

    df_denorm = pd.read_csv(path_denorm)
    df_label_all = pd.read_csv(path_label)
    
    # 演出指示から該当する曲のデータだけ抽出
    df_label = df_label_all[df_label_all['song_id'] == song_id].copy()

    print(f"--- [Step 2] 最適化開始: {song_id} ---")

    # 2. データの結合
    # フレーム番号で紐付け。演出指示側の必要なフラグだけ持ってくる
    merged = pd.merge(
        df_denorm, 
        df_label[['frame', 'bin_event_cut', 'bin_sub_phase_boundary']], 
        on='frame', 
        how='left'
    ).fillna(0) # ラベルがない区間は0埋め

# 3. ノイズ（30f固定）を除去した、純粋な演出連動ロジック
    condition = (
        (merged['bin_event_cut'] == 1) | 
        (merged['bin_sub_phase_boundary'] == 1) |
        (merged['frame'] == 0) |                   # 最初のフレーム (0.0)
        (merged.index == len(merged) - 1)          # 最後のフレーム
    )
    
    # カット(bin_event_cut)の直前フレーム(i-1)も追加
    # これにより「カットの終点」と「次のカットの始点」が1フレーム差で記録される
    cut_indices = merged.index[merged['bin_event_cut'] == 1].tolist()
    pre_cut_indices = [i - 1 for i in cut_indices if i > 0]
    
    # すべての必須インデックスを統合（30fごとの剰余計算を排除）
    selected_indices = sorted(list(set(merged.index[condition].tolist() + pre_cut_indices)))
    selected_indices = [i for i in selected_indices if i == 0 or i >= 5]
    
    df_opt = merged.iloc[selected_indices].copy()
    
    # 4. 浮動小数点によるノイズを防ぐため、念のためframeを整数に（VMD仕様に合わせる）
    df_opt['frame'] = df_opt['frame'].astype(int)

    # 出力カラムの整理
    output_cols = ['frame', 'pos_x', 'pos_y', 'pos_z', 'distance', 'rot_x', 'rot_y', 'rot_z', 'fov']
    df_opt = df_opt[output_cols]
    DIR_STEP2.mkdir(parents=True, exist_ok=True)
    output_path = DIR_STEP2 / f"step2_opt_{song_id}.csv"
    df_opt.to_csv(output_path, index=False)
    
    print(f"保存完了: {output_path.name} ({len(df_denorm)}f -> {len(df_opt)}f)")

if __name__ == "__main__":
    # 単体実行デバッグ用
    target_id = "ring_my_bell" 
    run_optimize(target_id)