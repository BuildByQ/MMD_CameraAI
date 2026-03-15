import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PRED01_ROOT = PROJECT_ROOT.parent / 'predict_01'
PRED01TO02_ROOT = PROJECT_ROOT.parent / 'predict_01to02'

# 出力先フォルダがなければ作成する
PRED01TO02_ROOT.mkdir(parents=True, exist_ok=True)

ML_ROOT = PROJECT_ROOT.parent / 'ml'  # 1つ上の階層のmlフォルダを指す

# 制御パラメータ
PHYSICAL_CUT_THRESH = 0.15
SUB_PHASE_THRESH = 0.7
MIN_PROB_ADOPT = 0.15
NEAR_MISS_RANGE = 0.1
CONTEXT_PENALTY_COEFF = 0.8

# 1. 設定読み込み
def load_config(config_path="config.json"):
    """
    設定ファイル（config.json）を読み込んで辞書として返す関数。
    """
    # ML_ROOT が定義されている前提で絶対パスを作成
    full_path = ML_ROOT / config_path

    if not full_path.exists():
        # ML_ROOT にない場合はカレントディレクトリも探す
        full_path = Path(config_path)
        if not full_path.exists():
            print(f"警告: 設定ファイルが見つかりません: {config_path}。バイアスなしで実行します。")
            return {}

    with open(full_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"設定を読み込みました: {full_path}")
    return config

def apply_director_polish(section, prob_cols, last_selected, is_first_frame_of_cut, bias_config, label_groups):
    """
    bias_config: {"bin_target_leg_focused": 0.2, ...} のような辞書を想定
    """
    polished = section.copy()
    
    # bin_ カラムの初期化と新設フラグの処理
    bin_cols = [c.replace('prob_', 'bin_') for c in prob_cols]
    for bc in bin_cols:
        polished[bc] = 0
    polished['bin_event_cut'] = 0
    polished['bin_sub_phase_boundary'] = 0
    
    if is_first_frame_of_cut:
        # ilocを使用して最初のフレームにフラグを立てる
        polished.iloc[0, polished.columns.get_loc('bin_event_cut')] = 1

    # サブフェーズ境界の特定
    vec_data = section[prob_cols].values
    vec_diffs = np.insert(np.linalg.norm(vec_data[1:] - vec_data[:-1], axis=1), 0, 0)
    change_indices = np.where(vec_diffs > SUB_PHASE_THRESH)[0]
    
    for p in change_indices:
        polished.iloc[p, polished.columns.get_loc('bin_sub_phase_boundary')] = 1

    start_idx = section.index[0]
    sub_boundaries = [start_idx] + [section.index[p] for p in change_indices] + [section.index[-1] + 1]

    current_last_selected = last_selected.copy()

    # サブフェーズごとの支配的ラベル選出
    for j in range(len(sub_boundaries) - 1):
        s, e = sub_boundaries[j], sub_boundaries[j+1]
        phase = polished.loc[s:e-1]
        if len(phase) == 0: continue
        
        new_selections = set()

        for group_name, members in label_groups.items():
            valid_cols = []
            
            if group_name == "target":
                # --- ターゲットグループ専用：日本語名を含む全バリエーションを回収 ---
                for bone_name in members:
                    # 例: "頭" が含まれ、かつ prob_ で始まるカラムをすべて抽出
                    # (prob_bin_target_頭_focused_strict 等をまとめて拾う)
                    matched = [c for c in phase.columns if (c.startswith("prob_") and bone_name in c)]
                    valid_cols.extend(matched)
            else:
                # --- その他のグループ：前方一致または完全一致で回収 ---
                for m in members:
                    # jsonに 'prob_' が付いていてもいなくても対応できるように
                    col_name = m if m.startswith("prob_") else f"prob_{m}"
                    if col_name in phase.columns:
                        valid_cols.append(col_name)

            if not valid_cols: continue

            # 以降のスコアリングと排他制御は共通
            avg_probs = {}
            for c in valid_cols:
                p = phase[c].mean()
                
                # バイアス引き当て (jsonのキーは prob_ 無しを想定)
                bias_key = c.replace('prob_', '')
                p += bias_config.get(bias_key, 0.0)
                
                if c in current_last_selected:
                    p *= CONTEXT_PENALTY_COEFF
                avg_probs[c] = p

            if not avg_probs: continue
            best_col = max(avg_probs, key=avg_probs.get)
            
            if avg_probs[best_col] > MIN_PROB_ADOPT:
                # フラグ立て (prob_ -> bin_)
                # 例: prob_bin_target_頭... -> bin_bin_target_頭...
                t_bin_col = best_col.replace('prob_', 'bin_', 1)
                if t_bin_col in polished.columns:
                    polished.loc[s:e-1, t_bin_col] = 1
                new_selections.add(best_col)
        
        current_last_selected = new_selections

    return polished, current_last_selected

def main():
    config = load_config("config.json")
    bias_config = config.get("bias_settings", {})
    label_groups = config.get("label_groups", {})

    input_path = os.path.join(PRED01_ROOT, "prediction_full.csv")
    output_path = os.path.join(PRED01TO02_ROOT, "director_instruction.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: 入力ファイルが見つかりません: {input_path}")
        return

    df_all = pd.read_csv(input_path, encoding="utf-8-sig")
    all_refined_sections = []

    # --- 曲ごとにグループ化してループ ---
    for sid, df in df_all.groupby("song_id", sort=False):
        print(f"演出ブラッシュアップ中: {sid} ({len(df)} frames)")
        
        # 物理カット検出（曲の中での相対的な位置で計算）
        peaks = (df['prob_event_cut'] > df['prob_event_cut'].shift(1)) & \
                (df['prob_event_cut'] > df['prob_event_cut'].shift(-1))
        
        cut_indices = df.index[peaks & (df['prob_event_cut'] > PHYSICAL_CUT_THRESH)].tolist()
        boundaries = [df.index[0]] + cut_indices + [df.index[-1] + 1]

        prob_cols = [c for c in df.columns if c.startswith('prob_') and c != 'prob_event_cut']
        global_last_selected = set()

        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            section = df.loc[start:end-1]
            if len(section) == 0: continue
            
            is_first = (i == 0) # 曲の最初、またはカットの最初
            refined, global_last_selected = apply_director_polish(
                section, prob_cols, global_last_selected, is_first, bias_config, label_groups
            )
            all_refined_sections.append(refined)

    # 全曲分を結合して保存
    final_df = pd.concat(all_refined_sections)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"パイプライン処理完了: {output_path}")

if __name__ == "__main__":
    main()