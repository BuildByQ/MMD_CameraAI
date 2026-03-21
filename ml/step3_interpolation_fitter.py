import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time
import os

# --- ディレクトリ設定 ---
PROJECT_ROOT = Path(__file__).parent
STEP1_DIR = PROJECT_ROOT.parent / 'output' / 'Step1'
STEP2_DIR = PROJECT_ROOT.parent / 'output' / 'Step2'
OUTPUT_DIR = PROJECT_ROOT.parent / 'output' / 'Step2.5'

# MMD標準の直線パラメータ
MMD_LINEAR = [20, 20, 107, 107]

SEARCH_PATTERNS = [
    [20, 20, 107, 107], [73, 18, 43, 105], [101, 18, 127, 92], 
    [73, 18, 64, 117], [64, 64, 64, 64], [20, 0, 107, 127]
]

def get_mmd_bezier_value(t, p):
    x1, y1, x2, y2 = p[0]/127.0, p[1]/127.0, p[2]/127.0, p[3]/127.0
    low, high = 0.0, 1.0
    for _ in range(20):
        mid = (low + high) / 2
        vx = 3 * x1 * mid * (1-mid)**2 + 3 * x2 * mid**2 * (1-mid) + mid**3
        if vx < t: low = mid
        else: high = mid
    t_bezier = (low + high) / 2
    vy = 3 * y1 * t_bezier * (1-t_bezier)**2 + 3 * y2 * t_bezier**2 * (1-t_bezier) + t_bezier**3
    return vy

def fit_worker(args):
    col_name, raw_segment = args
    n = len(raw_segment)
    if n <= 2: return MMD_LINEAR
    
    start_v, end_v = raw_segment[0], raw_segment[-1]
    diff = end_v - start_v
    if abs(diff) < 1e-6: return MMD_LINEAR

    norm_raw = (raw_segment - start_v) / diff
    t_steps = np.linspace(0, 1, n)

    def calculate_mse(p):
        sim = np.array([get_mmd_bezier_value(t, p) for t in t_steps])
        return np.mean((norm_raw - sim)**2)

    best_mse = float('inf')
    best_pts = list(MMD_LINEAR)

    for p_init in SEARCH_PATTERNS:
        mse = calculate_mse(p_init)
        if mse < best_mse:
            best_mse = mse
            best_pts = list(p_init)
            if best_mse < 1e-10: return best_pts

    for step in [20, 4, 1]:
        improved = True
        while improved:
            improved = False
            for i in [1, 3, 0, 2]:
                current_val = best_pts[i]
                for delta in [-step, step]:
                    test_val = int(np.clip(current_val + delta, 0, 127))
                    if test_val == current_val: continue
                    test_p = list(best_pts); test_p[i] = test_val
                    mse = calculate_mse(test_p)
                    if mse < best_mse:
                        best_mse = mse; best_pts = test_p; improved = True
    return best_pts

def process_single_file(step2_file):
    step1_file = STEP1_DIR / step2_file.name.replace('step2_opt_', 'step1_denorm_')
    if not step1_file.exists(): return

    df_raw = pd.read_csv(step1_file)
    df_key = pd.read_csv(step2_file)
    df_raw['frame'] = df_raw['frame'].astype(int)
    df_raw = df_raw.set_index('frame')
    key_indices = df_key['frame'].astype(int).tolist()
    
    target_cols = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'distance', 'fov']
    
    # --- Frame 0 の初期化（空白回避） ---
    for col in target_cols:
        for suffix, val in zip(['_x1', '_y1', '_x2', '_y2'], MMD_LINEAR):
            df_key.loc[0, f"{col}{suffix}"] = val

    tasks = []
    for i in range(len(key_indices)-1):
        s_frame, e_frame = key_indices[i], key_indices[i+1]
        for col in target_cols:
            if s_frame in df_raw.index and e_frame in df_raw.index:
                segment = df_raw.loc[s_frame:e_frame, col].values
                tasks.append((col, segment, i))

    with Pool(cpu_count()) as p:
        results = p.map(fit_worker, [(t[0], t[1]) for t in tasks])

    for task, res in zip(tasks, results):
        col, _, task_idx = task
        target_row_idx = task_idx + 1 # 区間の終端（次のキーフレーム）に書き込む
        df_key.loc[target_row_idx, f"{col}_x1"] = res[0]
        df_key.loc[target_row_idx, f"{col}_y1"] = res[1]
        df_key.loc[target_row_idx, f"{col}_x2"] = res[2]
        df_key.loc[target_row_idx, f"{col}_y2"] = res[3]

    output_path = OUTPUT_DIR / step2_file.name.replace('step2_opt_', 'step2.5_ready_')
    df_key.to_csv(output_path, index=False)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    step2_files = list(STEP2_DIR.glob('*.csv'))
    for f in step2_files:
        process_single_file(f)
    print("全プロセス完了!")

if __name__ == "__main__":
    main()