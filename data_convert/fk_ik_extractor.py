import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from math import radians
from numpy.linalg import norm
from pymeshio.vmd import reader as vmd_reader
from pymeshio.pmx import reader as pmx_reader
import math


PROJECT_ROOT = Path(__file__).parent
ML_ROOT = PROJECT_ROOT.parent / 'ml'  # 1つ上の階層のmlフォルダを指す
MODEL_ROOT = PROJECT_ROOT.parent / 'data' / 'mmd_model'  # 1つ上の階層のmlフォルダを指す


# 1. 設定読み込み
def load_config(config_path="config.json"):
    """
    設定ファイル（config.json）を読み込んで辞書として返す関数。
    train01.py の全処理がこの設定を参照する。
    """
    config_path = ML_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"設定を読み込みました: {config_path}")
    return config


# ① PMX 読み込み
def load_pmx(path):
    """
    PMXファイルを読み込み、pymeshioのpmxオブジェクトを返す。
    """
    from pymeshio.pmx import reader as pmx_reader
    from pathlib import Path

    path = Path(path)

    print(f"PMX読み込み開始: {path.resolve()}")
    if not path.exists():
        raise FileNotFoundError(f"PMXファイルが存在しません: {path.resolve()}")

    pmx = pmx_reader.read_from_file(str(path))

    if pmx is None:
        raise RuntimeError(f"PMX の読み込みに失敗しました: {path.resolve()}")

    # IK の数を正しく数える
    ik_count = sum(1 for b in pmx.bones if b.ik is not None)

    print(f"PMX読み込み成功: {path.resolve()}")
    print(f"ボーン数: {len(pmx.bones)} / IK数: {ik_count}")
    return pmx

class Bone:
    def __init__(self, name, index, parent_index, local_position):
        self.name = name
        self.index = index
        self.parent_index = parent_index
        self.parent = None
        self.children = [] # 追加
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


# ============================================================
# クォータニオン・行列ユーティリティ
# ============================================================

def normalize_quat(q):
    q = np.array(q, dtype=np.float32)
    n = norm(q)
    if n < 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return q / n


def quat_to_matrix(q):
    """クォータニオン(x, y, z, w)から4x4行列を生成"""
    x, y, z, w = q
    # 正規化
    n = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 - 2.0 * (yy + zz)
    m[0, 1] = 2.0 * (xy - wz)
    m[0, 2] = 2.0 * (xz + wy)
    m[1, 0] = 2.0 * (xy + wz)
    m[1, 1] = 1.0 - 2.0 * (xx + zz)
    m[1, 2] = 2.0 * (yz - wx)
    m[2, 0] = 2.0 * (xz - wy)
    m[2, 1] = 2.0 * (yz + wx)
    m[2, 2] = 1.0 - 2.0 * (xx + yy)
    return m

def matrix_to_quat(m):
    """
    回転行列（4x4）→ クォータニオン
    """
    R = m[:3, :3]
    t = np.trace(R)

    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # 対角成分の最大値で場合分け
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            w = (R[2, 1] - R[1, 2]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
            w = (R[0, 2] - R[2, 0]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            w = (R[1, 0] - R[0, 1]) / s

    return normalize_quat([x, y, z, w])

# ② Bone 構造生成（FK+IK 用）
def build_bones(pmx):
    bones_list = []  # インデックス順保持用
    bones_dict = {}  # 名前引き用
    ik_list = []
    movable_bones = set()

    # --- 1. Bone インスタンス生成（相対座標への変換） ---
    for i, b in enumerate(pmx.bones):
        # PMXのb.positionは世界座標。親との差分をとって相対座標にする
        if b.parent_index >= 0:
            parent_b = pmx.bones[b.parent_index]
            rel_pos = [
                b.position.x - parent_b.position.x,
                b.position.y - parent_b.position.y,
                b.position.z - parent_b.position.z
            ]
        else:
            # 親がいない（ルート）場合は世界座標のまま
            rel_pos = [b.position.x, b.position.y, b.position.z]

        bone = Bone(
            name=b.name,
            index=i,
            parent_index=b.parent_index,
            local_position=rel_pos
        )
        bones_list.append(bone)
        bones_dict[b.name] = bone

        # PMX フラグで「移動可能（VMDのposキーを持つ）」ボーンを判定
        if b.flag & 0x0002: # 0x0002 が「移動可能」フラグです
            movable_bones.add(b.name)

    # --- 2. 親子リンク ---
    for bone in bones_list:
        if bone.parent_index >= 0:
            parent_bone = bones_list[bone.parent_index]
            bone.parent = parent_bone
            parent_bone.children.append(bone)

    # --- 3. IK情報の構築 ---
    for b in pmx.bones:
        if b.ik is not None:
            ik = IK(
                target=pmx.bones[b.ik.target_index].name,
                effector=b.name,
                chain=[pmx.bones[link.bone_index].name for link in b.ik.link],
                iterations=b.ik.loop,
                angle_limit=b.ik.limit_radian
            )
            ik_list.append(ik)

    print(f"Bone構造生成完了: Bone数={len(bones_dict)}, 移動可能ボーン={len(movable_bones)}")
    return bones_dict, ik_list, movable_bones

class FKModel:
    def __init__(self, pmx):
        self.bones = {}
        self.movable_bones = set()
        # インデックス順にボーンを生成して保持
        sorted_pmx_bones = sorted(enumerate(pmx.bones), key=lambda x: x[0])
        
        for i, b in sorted_pmx_bones:
            if b.parent_index >= 0:
                p_pos = pmx.bones[b.parent_index].position
                rel_pos = [b.position.x - p_pos.x, b.position.y - p_pos.y, b.position.z - p_pos.z]
            else:
                rel_pos = [b.position.x, b.position.y, b.position.z]
            
            bone = Bone(b.name, i, b.parent_index, rel_pos)
            self.bones[b.name] = bone
            if b.flag & 0x0002: self.movable_bones.add(b.name)
            
        # 親子リンク
        for bone in self.bones.values():
            if bone.parent_index >= 0:
                parent_name = pmx.bones[bone.parent_index].name
                bone.parent = self.bones[parent_name]
                bone.parent.children.append(bone)

    def update(self):
        # インデックス順に更新することで親から子へ正しく伝播
        for b in sorted(self.bones.values(), key=lambda x: x.index):
            b.update_matrix()

def interpolate_bone_frames(df: pd.DataFrame) -> pd.DataFrame:
    """全ボーンの全フレームをSlerp/Lerp補間"""
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
    # JSONからボーン名リスト抽出
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    target_bones = []
    for feat in config.get("motion_features", []):
        name = feat.rsplit('_', 2)[0]
        if name not in target_bones: target_bones.append(name)

    model = FKModel(pmx_data)
    df_raw = pd.read_csv(vmd_csv_path)
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
            res[f"{b_name}_x"], res[f"{b_name}_y"], res[f"{b_name}_z"] = pos
        dataset.append(res)
    pd.DataFrame(dataset).to_csv(output_path, index=False)

def create_model(bones, ik_list, movable_bones):
    return FKModel(bones, ik_list, movable_bones)

def update_model(model):
    model.update_bone()

class IK:
    """
    PMX の IK 情報を Python で扱いやすい形に変換したクラス。
    """
    def __init__(self, target, effector, chain, iterations, angle_limit):
        self.target = target
        self.effector = effector
        self.chain = chain
        self.iterations = iterations
        self.angle_limit = angle_limit  # True/False（角度制限の有無）

def load_vmd(path):
    vmd = vmd_reader.read_from_file(str(path))
    if vmd is None:
        raise RuntimeError(f"VMD の読み込みに失敗しました: {path}")
    print(f"VMD 読み込み成功: {path}")
    return vmd

def apply_vmd_frame(model, vmd_dict, frame):
    for bone_name, keyframes in vmd_dict.items():
        kf = next((k for k in keyframes if k.frame == frame), None)
        # ※実際は現在のframe以下の最大キーを取得する線形補間が必要ですが、一旦一致のみで実装
        if kf is None or bone_name not in model.bones:
            continue

        bone = model.bones[bone_name]
        bone.set_local_rotation([kf.q.x, kf.q.y, kf.q.z, kf.q.w])

        if bone_name in model.movable_bones:
            # 初期座標(local_position)を壊さず、追加移動分としてセット
            bone.vmd_offset = np.array([kf.pos.x, kf.pos.y, kf.pos.z], dtype=np.float32)
        else:
            bone.vmd_offset = np.zeros(3, dtype=np.float32)

def get_max_frame(vmd_dict):
    max_frame = 0
    for keyframes in vmd_dict.values():
        if keyframes:
            max_frame = max(max_frame, max(k.frame for k in keyframes))
    return max_frame

def process_motion(model, vmd):
    vmd_dict = build_vmd_dict(vmd)
    max_frame = get_max_frame(vmd_dict)

    results = []

    for frame in range(max_frame + 1):
        apply_vmd_frame(model, vmd_dict, frame)
        update_model(model)

        frame_data = {name: bone.world_position.copy()
                      for name, bone in model.bones.items()}
        results.append(frame_data)

    return results

def decode_bone_name(b):
    # VMD のボーン名は Shift-JIS の bytes
    return b.decode("shift_jis", errors="ignore")

def build_vmd_dict(vmd):
    bone_dict = {}

    for kf in vmd.motions:
        name = decode_bone_name(kf.name)

        if name not in bone_dict:
            bone_dict[name] = []

        bone_dict[name].append(kf)

    # フレーム番号順にソート
    for name in bone_dict:
        bone_dict[name].sort(key=lambda k: k.frame)

    return bone_dict

def main():
    # 0. 設定とパスの準備
    config = load_config("config.json")
    pmx_path = MODEL_ROOT / "model.pmx"
    # ここは「VMD生ファイル」ではなく「キーフレームCSV」のパスを指定します
    vmd_csv_path = MODEL_ROOT / "motion_keyframes.csv" 
    json_path = ML_ROOT / "config.json"
    output_path = MODEL_ROOT / "world_coordinates_train.csv"

    # ① PMX データの読み込み
    # generate_world_coords_dataset の内部で FKModel(pmx_data) を作るので、
    # ここでは pmx_reader で読み込んだオブジェクトを渡します。
    pmx_data = load_pmx(pmx_path)

    # ② ワールド座標データセットの生成実行
    print("ワールド座標データセットの生成を開始します...")
    generate_world_coords_dataset(
        pmx_data=pmx_data,
        vmd_csv_path=vmd_csv_path,
        json_path=json_path,
        output_path=output_path
    )

    print(f"全工程が完了しました。出力先: {output_path}")

if __name__ == "__main__":
    main()