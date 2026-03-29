import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent
PRED01_ROOT = PROJECT_ROOT.parent / 'predict_01'
PRED01TO02_ROOT = PROJECT_ROOT.parent / 'predict_01to02'
ML_ROOT = PROJECT_ROOT.parent / 'ml'

PRED01TO02_ROOT.mkdir(parents=True, exist_ok=True)

# 1. 設定読み込み（閾値の外部化対応）
def load_config(config_path="config.json"):
    full_path = ML_ROOT / config_path
    if not full_path.exists():
        full_path = Path(config_path)
        if not full_path.exists():
            print(f"警告: 設定ファイルが見つかりません。デフォルト値で実行します。")
            return {}

    with open(full_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"設定を読み込みました: {full_path}")
    return config

def apply_director_polish(section, prob_cols, last_selected, is_first_frame_of_cut, bias_config, label_groups, thresholds):
    """
    この関数の中で「カット」「サブフェーズ」「ラベル」のすべてを確定させる
    """
    PHYSICAL_CUT_THRESH = thresholds.get("physical_cut", 0.05)
    SUB_PHASE_THRESH    = thresholds.get("sub_phase", 0.3)
    MIN_PROB_ADOPT      = thresholds.get("min_prob_adopt", 0.1)
    CONTEXT_PENALTY_COEFF = thresholds.get("context_penalty", 0.8)
    
    polished = section.copy()
    
    # --- [A] 初期化：bin_ カラムをすべて 0 に（一括リセット） ---
    bin_cols = [c.replace('prob_', 'bin_') for c in prob_cols]
    if 'bin_event_cut' not in bin_cols: bin_cols.append('bin_event_cut')
    if 'bin_sub_phase_boundary' not in bin_cols: bin_cols.append('bin_sub_phase_boundary')
    
    for bc in bin_cols:
        polished[bc] = 0

    # --- [B] 物理カット(bin_event_cut)の特定 ---
    # セクション内での「山頂」を判定
    p_cut = section['prob_event_cut']
    peaks = (p_cut > p_cut.shift(1)) & (p_cut >= p_cut.shift(-1))
    cut_indices = section.index[peaks & (p_cut > PHYSICAL_CUT_THRESH)].tolist()
    
    # 算出した地点にフラグを立てる
    polished.loc[cut_indices, 'bin_event_cut'] = 1
    
    # 曲の先頭なら強制的にカット扱い
    if is_first_frame_of_cut:
        polished.iloc[0, polished.columns.get_loc('bin_event_cut')] = 1

    # --- [C] サブフェーズ境界(bin_sub_phase_boundary)の特定 ---
    vec_data = section[prob_cols].values
    vec_diffs = np.insert(np.linalg.norm(vec_data[1:] - vec_data[:-1], axis=1), 0, 0)
    change_indices = np.where(vec_diffs > SUB_PHASE_THRESH)[0]
    
    for p in change_indices:
        polished.iloc[p, polished.columns.get_loc('bin_sub_phase_boundary')] = 1

    # --- 支配的ラベル選出ロジック (現状維持) ---
    start_idx = section.index[0]
    sub_boundaries = [start_idx] + [section.index[p] for p in change_indices] + [section.index[-1] + 1]
    current_last_selected = last_selected.copy()

    for j in range(len(sub_boundaries) - 1):
        s, e = sub_boundaries[j], sub_boundaries[j+1]
        phase = polished.loc[s:e-1]
        if len(phase) == 0: continue
        new_selections = set()
        for group_name, members in label_groups.items():
            valid_cols = []
            if group_name == "target":
                for bone_name in members:
                    matched = [c for c in phase.columns if (c.startswith("prob_") and bone_name in c)]
                    valid_cols.extend(matched)
            else:
                for m in members:
                    col_name = m if m.startswith("prob_") else f"prob_{m}"
                    if col_name in phase.columns:
                        valid_cols.append(col_name)

            if not valid_cols: continue
            avg_probs = {}
            for c in valid_cols:
                p = phase[c].mean()
                bias_key = c.replace('prob_', '')
                p += bias_config.get(bias_key, 0.0)
                if c in current_last_selected:
                    p *= CONTEXT_PENALTY_COEFF
                avg_probs[c] = p

            if not avg_probs: continue
            best_col = max(avg_probs, key=avg_probs.get)
            if avg_probs[best_col] > MIN_PROB_ADOPT:
                t_bin_col = best_col.replace('prob_', 'bin_', 1)
                if t_bin_col in polished.columns:
                    polished.loc[s:e-1, t_bin_col] = 1
                new_selections.add(best_col)
        current_last_selected = new_selections

    return polished, current_last_selected

def main():
    config = load_config("config.json")
    thresh_cfg = config.get("thresholds", {})
    bias_config = config.get("bias_settings", {})
    label_groups = config.get("label_groups", {})

    # --- 1. 入力フォルダ内のCSVをすべて取得 ---
    # predict01が出力した「prediction_*.csv」を対象にする
    input_files = list(PRED01_ROOT.glob("predict_*.csv"))
    
    if not input_files:
        print(f"Error: 入力ファイルが {PRED01_ROOT} に見つかりません。")
        return

    print(f"{len(input_files)} 件のファイルを処理します...")

    for file_path in input_files:
        print(f"処理中: {file_path.name}")
        
        # --- 2. ファイルごとに読み込み ---
        df_single_song = pd.read_csv(file_path, encoding="utf-8-sig")
        
        prob_cols = [c for c in df_single_song.columns if c.startswith('prob_') and c != 'prob_event_cut']
        all_refined_sections = []

        # 元々 song_id で groupby していたロジックを、ファイル単位のループに統合
        # (ファイル内に複数曲混ざっている可能性も考慮して groupby は残すのが安全)
        for sid, df in df_single_song.groupby("song_id", sort=False):
            refined, _ = apply_director_polish(
                df, prob_cols, set(), True, bias_config, label_groups, thresh_cfg
            )
            all_refined_sections.append(refined)

        if not all_refined_sections:
            continue

        # --- 3. 曲ごとの出力ファイル名を生成 ---
        # 例: prediction_songA.csv -> director_songA.csv
        out_name = file_path.name.replace("prediction_", "director_")
        output_path = PRED01TO02_ROOT / out_name

        # 結合して保存
        final_df = pd.concat(all_refined_sections)
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # --- ログ出力 (ファイルごと) ---
        num_cuts = final_df['bin_event_cut'].sum()
        print(f"  出力完了: {output_path.name} (物理カット: {int(num_cuts)})")

    print("-" * 50)
    print(f"全 {len(input_files)} 件のパイプライン処理が完了しました。")
    print(f"出力先: {PRED01TO02_ROOT}")

if __name__ == "__main__":
    main()