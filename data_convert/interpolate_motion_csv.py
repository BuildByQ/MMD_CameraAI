import json
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import norm
from pymeshio.vmd import reader as vmd_reader
from pymeshio.pmx import reader as pmx_reader

# パス設定
PROJECT_ROOT = Path(__file__).parent
ML_ROOT = PROJECT_ROOT.parent / 'ml'
MODEL_ROOT = PROJECT_ROOT.parent / 'data' / 'mmd_model'
project_root_str = str(Path(__file__).parent.parent)

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
    result_rows = []
    for bone_name, bone_data in df.groupby('bone_name'):
        bone_data = bone_data.sort_values('frame')
        for i in range(len(bone_data) - 1):
            curr, nxt = bone_data.iloc[i], bone_data.iloc[i+1]
            result_rows.append(curr)
            diff = int(nxt['frame'] - curr['frame'])
            if diff > 1:
                q1 = np.array([curr['rot_x'], curr['rot_y'], curr['rot_z'], curr['rot_w']])
                q2 = np.array([nxt['rot_x'], nxt['rot_y'], nxt['rot_z'], nxt['rot_w']])
                if np.dot(q1, q2) < 0: q2 = -q2
                for f in range(1, diff):
                    alpha = f / diff
                    interp = curr.copy()
                    interp['frame'] = curr['frame'] + f
                    for ax in ['x', 'y', 'z']:
                        interp[f'pos_{ax}'] = (1-alpha)*curr[f'pos_{ax}'] + alpha*nxt[f'pos_{ax}']
                    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
                    theta = math.acos(dot) * alpha
                    q_rel = (q2 - q1 * dot)
                    q_rel /= (norm(q_rel) + 1e-9)
                    res_q = q1 * math.cos(theta) + q_rel * math.sin(theta)
                    interp['rot_x'], interp['rot_y'], interp['rot_z'], interp['rot_w'] = res_q / norm(res_q)
                    result_rows.append(interp)
        result_rows.append(bone_data.iloc[-1])
    return pd.DataFrame(result_rows).sort_values(['frame', 'bone_name'])

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

def main():
    input_dir = os.path.join(project_root_str, 'data', 'motion_csv')
    output_dir = os.path.join(project_root_str, 'data', 'motion_wide')
    json_path = ML_ROOT / "config.json"
    pmx_path = MODEL_ROOT / "model.pmx"

    pmx_data = load_pmx(pmx_path)
    os.makedirs(output_dir, exist_ok=True)

    print("ワールド座標データセットの生成を開始します...")
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            generate_world_coords_dataset(pmx_data, input_path, json_path, output_path)
            print(f"  [完了] {filename}")

    print(f"全工程が完了しました。出力先: {output_dir}")

if __name__ == "__main__":
    main()