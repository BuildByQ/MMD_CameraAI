import pandas as pd
import numpy as np
import os
import json  # 追加
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PRED01_ROOT = PROJECT_ROOT.parent / 'predict_01'  # 1つ上の階層のmlフォルダを指す
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
        for group_name, cols in label_groups.items(): # 引数の label_groups を使用
            valid_cols = [c for c in cols if c in phase.columns]
            if not valid_cols: continue
            
            avg_probs = {}
            for c in valid_cols:
                # 1. AIが算出したベースの平均確率
                p = phase[c].mean()
                
                # 2. JSONバイアス（足し算）の適用
                # カラム名が prob_ か bin_ かに関わらず bias_config から取得
                bias = bias_config.get(c, bias_config.get(c.replace('prob_', 'bin_'), 0.0))
                p += bias
                
                # 3. 継続性ペナルティの適用
                if c in current_last_selected:
                    p *= CONTEXT_PENALTY_COEFF
                
                avg_probs[c] = p
            
            # 代表ラベルの選出（バイアス適用後の値で判定）
            if not avg_probs: continue
            best_col = max(avg_probs, key=avg_probs.get)
            
            if avg_probs[best_col] > MIN_PROB_ADOPT:
                polished.loc[s:e-1, best_col.replace('prob_', 'bin_')] = 1
                new_selections.add(best_col)
                
                # 近似ラベル（複数点灯）の判定もバイアス適用後の値で行う
                for c, p in avg_probs.items():
                    if c != best_col and (avg_probs[best_col] - p) < NEAR_MISS_RANGE:
                        if p > MIN_PROB_ADOPT:
                            polished.loc[s:e-1, c.replace('prob_', 'bin_')] = 1
                            new_selections.add(c)
        
        current_last_selected = new_selections

    return polished, current_last_selected

def main():
    # 1. JSON設定の読み込み
    config = load_config("config.json")
    # bias_settings セクションを取得（なければ空の辞書）
    bias_config = config.get("bias_settings", {})

    input_path = os.path.join(PRED01_ROOT, "prediction_full.csv")
    output_path = os.path.join(PRED01_ROOT, "prediction_stage3_cleaned.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: 入力ファイルが見つかりません: {input_path}")
        return

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    
    # 物理カット（イベントカット）のピーク検出
    peaks = (df['prob_event_cut'] > df['prob_event_cut'].shift(1)) & \
            (df['prob_event_cut'] > df['prob_event_cut'].shift(-1))
    cut_indices = df.index[peaks & (df['prob_event_cut'] > PHYSICAL_CUT_THRESH)].tolist()
    boundaries = [0] + cut_indices + [len(df)]

    prob_cols = [c for c in df.columns if c.startswith('prob_') and c != 'prob_event_cut']
    label_groups = config.get("label_groups", {})
    cleaned_sections = []
    global_last_selected = set()

    # セクションごとに演出を磨き上げる
    for i in range(len(boundaries)-1):
        start, end = boundaries[i], boundaries[i+1]
        section = df.iloc[start:end]
        if len(section) == 0: continue
        
        is_first = (i == 0) or (start in cut_indices)
        # bias_config を引数に追加
        refined, global_last_selected = apply_director_polish(
            section, prob_cols, global_last_selected, is_first, bias_config, label_groups
        )
        cleaned_sections.append(refined)

    # 結果の結合と保存
    final_df = pd.concat(cleaned_sections)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Success! Stage 3 cleaned file created: {output_path}")

if __name__ == "__main__":
    main()