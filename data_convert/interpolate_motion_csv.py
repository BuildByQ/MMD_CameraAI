import json
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import norm
from pymeshio.vmd import reader as vmd_reader
from pymeshio.pmx import reader as pmx_reader
from interpolate import interpolate_bezier_frame, get_bezier_t
from concurrent.futures import ProcessPoolExecutor
import functools
from tqdm import tqdm  # pip install tqdm が必要です

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
ML_ROOT = PROJECT_ROOT / 'ml'

MODEL_ROOT = DATA_ROOT / 'mmd_model'
CONFIG_PATH = ML_ROOT / "config.json"

def quat_to_matrix(q):
    """クォータニオン(x, y, z, w)から4x4行列を生成"""
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-8: return np.eye(4, dtype=np.float32)
    x, y, z, w = x/n, y/n, z/n, w/n
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 - 2.0 * (yy + zz); m[0, 1] = 2.0 * (xy - wz); m[0, 2] = 2.0 * (xz + wy)
    m[1, 0] = 2.0 * (xy + wz); m[1, 1] = 1.0 - 2.0 * (xx + zz); m[1, 2] = 2.0 * (yz - wx)
    m[2, 0] = 2.0 * (xz - wy); m[2, 1] = 2.0 * (yz + wx); m[2, 2] = 1.0 - 2.0 * (xx + yy)
    return m

class Bone:
    def __init__(self, name, index, parent_index, local_position):
        self.name = name
        self.index = index
        self.parent_index = parent_index
        self.parent = None
        self.children = []
        self.local_position = np.array(local_position, dtype=np.float32)
        self.vmd_offset = np.zeros(3, dtype=np.float32)
        self.local_rotation = np.array([0, 0, 0, 1], dtype=np.float32)
        self.world_matrix = np.eye(4, dtype=np.float32)
        self.world_position = np.zeros(3, dtype=np.float32)

    def update_matrix(self):
        R = quat_to_matrix(self.local_rotation)
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = self.local_position + self.vmd_offset
        local_m = T @ R
        if self.parent:
            self.world_matrix = self.parent.world_matrix @ local_m
        else:
            self.world_matrix = local_m
        self.world_position = self.world_matrix[:3, 3]

class FKModel:
    def __init__(self, pmx):
        self.bones = {}
        self.movable_bones = set()
        
        # Bone構造構築（初期ポーズから相対座標を計算）
        for i, b in enumerate(pmx.bones):
            if b.parent_index >= 0:
                p_pos = pmx.bones[b.parent_index].position
                rel_pos = [b.position.x - p_pos.x, b.position.y - p_pos.y, b.position.z - p_pos.z]
            else:
                rel_pos = [b.position.x, b.position.y, b.position.z]
            
            bone = Bone(b.name, i, b.parent_index, rel_pos)
            self.bones[b.name] = bone
            if b.flag & 0x0002: self.movable_bones.add(b.name)
            
        # 親子関係のリンク
        for bone in self.bones.values():
            if bone.parent_index >= 0:
                parent_name = pmx.bones[bone.parent_index].name
                bone.parent = self.bones[parent_name]
                bone.parent.children.append(bone)

    def update(self):
        # PMXのインデックス順に更新すれば、必ず親から計算される
        for b in sorted(self.bones.values(), key=lambda x: x.index):
            b.update_matrix()

def interpolate_bone_frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    全ボーンのキーフレーム間をベジェ曲線(座標)とSlerp(回転)で補間する
    """
    # モーション用の補間パラメータマッピング (VMDバイナリ順: x1, x2, y1, y2)
    param_groups = {
        'pos_x': ['X_x1', 'X_x2', 'X_y1', 'X_y2'],
        'pos_y': ['Y_x1', 'Y_y1', 'Y_x2', 'Y_y2'],
        'pos_z': ['Z_x1', 'Z_x2', 'Z_y1', 'Z_y2'],
        'rotation': ['R_x1', 'R_x2', 'R_y1', 'R_y2']  # 回転の進捗計算用
    }

    result_rows = []
    
    # ボーンごとにグループ化して処理
    for bone_name, bone_data in df.groupby('bone_name'):
        bone_data = bone_data.sort_values('frame')
        
        for i in range(len(bone_data) - 1):
            curr = bone_data.iloc[i].to_dict()
            nxt = bone_data.iloc[i+1].to_dict()
            
            start_f = int(curr['frame'])
            end_f = int(nxt['frame'])
            
            # 開始フレームを追加
            result_rows.append(curr)
            
            # キーフレーム間の補間
            for f_num in range(start_f + 1, end_f):
                # 時間の進捗率 (0.0 - 1.0)
                t = (f_num - start_f) / (end_f - start_f)
                
                # --- 1. 座標のベジェ補間 ---
                # interpolate_bezier_frame を使用 (内部で nxt からパラメータを取得)
                # ※この関数が pos_x, pos_y, pos_z を計算して返すと想定
                interp_frame = interpolate_bezier_frame(curr, nxt, t, param_groups)
                
                # --- 2. 回転のSlerp補間 (ベジェ進捗 alpha を使用) ---
                # 回転用のベジェパラメータを取得 (nxtから)
                rx1, rx2, ry1, ry2 = [nxt[p] / 127.0 for p in param_groups['rotation']]
                
                # ベジェ曲線上の「値の進捗 y」を算出
                t_bezier = get_bezier_t(rx1, ry1, rx2, ry2, t)
                alpha = 3 * ry1 * t_bezier * (1-t_bezier)**2 + 3 * ry2 * t_bezier**2 * (1-t_bezier) + t_bezier**3
                
                # クォータニオンのSlerp計算
                q1 = np.array([curr['rot_x'], curr['rot_y'], curr['rot_z'], curr['rot_w']])
                q2 = np.array([nxt['rot_x'], nxt['rot_y'], nxt['rot_z'], nxt['rot_w']])
                
                if np.dot(q1, q2) < 0: q2 = -q2  # 最短経路
                
                dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
                theta = math.acos(dot) * alpha # 線形の alpha ではなくベジェの alpha を使う
                
                q_rel = (q2 - q1 * dot)
                norm_val = np.linalg.norm(q_rel)
                if norm_val > 1e-9:
                    q_rel /= norm_val
                    res_q = q1 * math.cos(theta) + q_rel * math.sin(theta)
                else:
                    res_q = q1 # 変化なし
                
                res_q /= np.linalg.norm(res_q)
                
                # 値を格納
                interp_frame['frame'] = f_num
                interp_frame['bone_name'] = bone_name
                interp_frame['rot_x'], interp_frame['rot_y'], interp_frame['rot_z'], interp_frame['rot_w'] = res_q
                
                result_rows.append(interp_frame)
                
        # 最後のキーフレームを追加
        result_rows.append(bone_data.iloc[-1].to_dict())

    # データフレームに変換して重複除去
    result_df = pd.DataFrame(result_rows).drop_duplicates(subset=['frame', 'bone_name'])
    return result_df.sort_values(['frame', 'bone_name']).reset_index(drop=True)

def generate_world_coords_dataset(pmx_data, vmd_csv_path, json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # JSONからボーン名のリストをそのまま取得
    target_bones = config.get("motion_features", [])

    model = FKModel(pmx_data)
    df_raw = pd.read_csv(vmd_csv_path)
    print(f"  - 補間中: {Path(vmd_csv_path).name}")
    df_full = interpolate_bone_frames(df_raw)
    
    dataset = []
    frames = sorted(df_full['frame'].unique())
    for f in frames:
        frame_data = df_full[df_full['frame'] == f]
        for _, row in frame_data.iterrows():
            if row['bone_name'] in model.bones:
                b = model.bones[row['bone_name']]
                b.local_rotation = np.array([row['rot_x'], row['rot_y'], row['rot_z'], row['rot_w']])
                if row['bone_name'] in model.movable_bones:
                    b.vmd_offset = np.array([row['pos_x'], row['pos_y'], row['pos_z']])
        
        model.update()
        
        res = {'frame': f}
        for b_name in target_bones:
            pos = model.bones[b_name].world_position if b_name in model.bones else [0,0,0]
            res[f"{b_name}_w_x"], res[f"{b_name}_w_y"], res[f"{b_name}_w_z"] = pos

        virtual_bone_defs = config.get("virtual_bone_definitions", {})
        for parent_name, settings in virtual_bone_defs.items():
            # JSONの定義（[0.0, 0.0, 10.0] など）を取得
            offset = settings.get("offset", [0.0, 0.0, 5.0])
            suffix = settings.get("suffix", "前方")
            virtual_name = f"{parent_name}{suffix}" # 例: "頭前方"

            if parent_name in model.bones:
                parent_bone = model.bones[parent_name]
                
                # ローカルでのオフセット値をワールド行列で変換
                # offset[0]:x, offset[1]:y, offset[2]:z
                front_local = np.array([offset[0], offset[1], offset[2], 1.0], dtype=np.float32)
                front_world = parent_bone.world_matrix @ front_local
                
                res[f"{virtual_name}_w_x"] = front_world[0]
                res[f"{virtual_name}_w_y"] = front_world[1]
                res[f"{virtual_name}_w_z"] = front_world[2]
            else:
                # 親ボーンが存在しない場合のフォールバック
                res[f"{virtual_name}_w_x"] = 0
                res[f"{virtual_name}_w_y"] = 0
                res[f"{virtual_name}_w_z"] = 0
        dataset.append(res)
    
    pd.DataFrame(dataset).to_csv(output_path, index=False)

def load_config(config_path="config.json"):
    path = ML_ROOT / config_path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pmx(path):
    pmx = pmx_reader.read_from_file(str(path))
    print(f"PMX読み込み成功: {path.name} (ボーン数: {len(pmx.bones)})")
    return pmx

def process_single_file(filename, input_dir, output_dir, pmx_data, target_bones, virtual_bone_defs):
    """1つのCSVファイルを補間・ワールド座標変換する単位（並列実行用）"""
    input_path = input_dir / filename
    output_path = output_dir / filename
    
    # 既存チェック（Step実行の効率化）
    if output_path.exists():
        return f"Skip: {filename}"

    try:
        # 1. 読み込み
        df_raw = pd.read_csv(input_path)
        
        # 2. ベジェ補間実行 (ローカル姿勢の確定)
        # ※前述の「ベジェ + Slerp」を実装した interpolate_bone_frames を使用
        df_full = interpolate_bone_frames(df_raw)
        
        # 3. FK計算 & データ抽出
        model = FKModel(pmx_data)
        dataset = []
        frames = sorted(df_full['frame'].unique())
        
        for f in frames:
            frame_data = df_full[df_full['frame'] == f]
            for _, row in frame_data.iterrows():
                b_name = row['bone_name']
                if b_name in model.bones:
                    b = model.bones[b_name]
                    b.local_rotation = np.array([row['rot_x'], row['rot_y'], row['rot_z'], row['rot_w']])
                    if b_name in model.movable_bones:
                        b.vmd_offset = np.array([row['pos_x'], row['pos_y'], row['pos_z']])
            
            model.update()
            
            res = {'frame': f}
            # 目線ボーン等、必要なボーンの抽出
            for b_name in target_bones:
                pos = model.bones[b_name].world_position if b_name in model.bones else [0,0,0]
                res[f"{b_name}_w_x"], res[f"{b_name}_w_y"], res[f"{b_name}_w_z"] = pos

            # バーチャルボーン（頭前方など）の計算
            for parent_name, settings in virtual_bone_defs.items():
                offset = settings.get("offset", [0.0, 0.0, 5.0])
                suffix = settings.get("suffix", "前方")
                if parent_name in model.bones:
                    parent_bone = model.bones[parent_name]
                    front_local = np.array([offset[0], offset[1], offset[2], 1.0], dtype=np.float32)
                    front_world = parent_bone.world_matrix @ front_local
                    res[f"{parent_name}{suffix}_w_x"] = front_world[0]
                    res[f"{parent_name}{suffix}_w_y"] = front_world[1]
                    res[f"{parent_name}{suffix}_w_z"] = front_world[2]
            
            dataset.append(res)
            
        pd.DataFrame(dataset).to_csv(output_path, index=False)
        return f"Success: {filename}"
        
    except Exception as e:
        return f"Error: {filename} ({e})"

def main():
    print(f"[Step 5] Motion Interpolation & World Coords 開始")
    
    # 1. JSON設定の読み込み
    if not CONFIG_PATH.exists():
        print(f"設定ファイルが見つかりません: {CONFIG_PATH}")
        return
        
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 2. パスと設定の確定
    input_dir = DATA_ROOT / 'motion_csv'
    output_dir = DATA_ROOT / 'motion_wide'
    pmx_path = MODEL_ROOT / "model.pmx"
    
    target_bones = config.get("motion_features", [])
    virtual_bone_defs = config.get("virtual_bone_definitions", {})

    output_dir.mkdir(parents=True, exist_ok=True)
    pmx_data = pmx_reader.read_from_file(str(pmx_path))

    # 3. 処理対象の取得
    csv_files = [f.name for f in input_dir.glob('*.csv')]
    if not csv_files:
        print(f"処理対象のCSVが見つかりません: {input_dir}")
        return

    # 4. 並列実行
    print(f"並列処理を開始します... (コア数に合わせて実行 / 対象: {len(csv_files)}件)")
    with ProcessPoolExecutor() as executor:
        worker = functools.partial(
            process_single_file,
            input_dir=input_dir,
            output_dir=output_dir,
            pmx_data=pmx_data,
            target_bones=target_bones,
            virtual_bone_defs=virtual_bone_defs
        )
        
        # list(executor.map(...)) を tqdm でラップします
        results = list(tqdm(executor.map(worker, csv_files), total=len(csv_files), desc="Processing CSVs"))
    
    # 5. 結果の要約表示
    for res in results:
        if "Success" not in res: # 異常系のみ表示
            print(res)
    print(f"全工程が完了しました。")

if __name__ == "__main__":
    main()