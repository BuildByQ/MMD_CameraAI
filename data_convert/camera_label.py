import pandas as pd
import numpy as np
import os
import json
import math
from pathlib import Path

def load_config():
    # スクリプトと同階層の config.json を読み込む
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        # mlディレクトリ直下を探すフォールバック
        config_path = Path(__file__).parent.parent / "ml" / "config.json"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

CONFIG = load_config()
VIRTUAL_BONE_DEFS = CONFIG.get("virtual_bone_definitions", {})
ANCHOR_DEFS = CONFIG.get("anchor_definitions", {})

class MMDCameraProjector:
    def __init__(self, aspect_ratio=16/9):
        self.aspect_ratio = aspect_ratio
        # 視野角補間用の多項式係数
        self.mmd_values = np.array([10, 20, 40, 60, 80, 100, 120])
        self.actual_values = np.array([20.0, 39.6, 73.0, 99.4, 119.6, 136.0, 149.3])
        self.fov_poly_coeffs = np.polyfit(self.mmd_values, self.actual_values, 3)

    def mmd_fov_to_actual_degrees(self, mmd_fov: float) -> float:
        mmd_fov = max(0, min(180, float(mmd_fov)))
        return max(0, min(180, np.polyval(self.fov_poly_coeffs, mmd_fov)))

    def get_camera_eye_pos(self, cam):
        target_pos = cam['pos']
        dist = cam['dist'] 
        init_offset = np.array([0, 0, dist])

        rx, ry, rz = np.radians(cam['rot'])
        Ry = np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        combined_rotation = Ry @ Rx @ Rz
        rotated_offset = combined_rotation @ init_offset
        return target_pos + rotated_offset

    def get_screen_projections(self, cam, bones):
        """ボーン座標をスクリーン(0.0~1.0)に投影（デバッグ・可視化用）"""
        eye_pos = self.get_camera_eye_pos(cam)
        target_pos = cam['pos']
        
        up = np.array([0, 1, 0])
        z_axis = (target_pos - eye_pos) / (np.linalg.norm(target_pos - eye_pos) + 1e-6)
        x_axis = np.cross(up, z_axis) / (np.linalg.norm(np.cross(up, z_axis)) + 1e-6)
        y_axis = np.cross(z_axis, x_axis)
        
        view_matrix = np.eye(4)
        view_matrix[0, :3], view_matrix[0, 3] = x_axis, -np.dot(x_axis, eye_pos)
        view_matrix[1, :3], view_matrix[1, 3] = y_axis, -np.dot(y_axis, eye_pos)
        view_matrix[2, :3], view_matrix[2, 3] = z_axis, -np.dot(z_axis, eye_pos)

        actual_fov = self.mmd_fov_to_actual_degrees(cam['fov'])
        f = 1.0 / math.tan(math.radians(actual_fov) / 2.0)
        near, far = 0.1, 10000.0
        
        proj_matrix = np.zeros((4, 4))
        proj_matrix[0, 0] = f / self.aspect_ratio
        proj_matrix[1, 1] = f
        proj_matrix[2, 2] = far / (far - near)
        proj_matrix[2, 3] = -far * near / (far - near)
        proj_matrix[3, 2] = 1.0

        vp_matrix = proj_matrix @ view_matrix
        projections = {}
        for name, bone in bones.items():
            world_pos = np.append(bone['pos'], 1.0)
            clip_pos = vp_matrix @ world_pos
            if clip_pos[3] != 0:
                ndc_x = clip_pos[0] / clip_pos[3]
                ndc_y = clip_pos[1] / clip_pos[3]
                projections[name] = {
                    'x': (ndc_x + 1.0) / 2.0,
                    'y': (1.0 - ndc_y) / 2.0,
                    'in_view': (abs(ndc_x) <= 1.0 and abs(ndc_y) <= 1.0 and clip_pos[3] > 0)
                }
        return projections

def detect_target(cam, eye_pos, bones):
    """
    JSONのanchor_definitionsに基づき、和名ボーンからラベル名を自動生成して判定。
    命名規則: bin_target_{ボーン名}_focused
    """
    targets = {}
    vec_cam_aim = cam['pos'] - eye_pos
    v_aim = vec_cam_aim / (np.linalg.norm(vec_cam_aim) + 1e-6)

    for anchor_key, info in ANCHOR_DEFS.items():
        target_bone_names = info.get("bones", [])
        
        for b_name in target_bone_names:
            # 仮想ボーン（前方）の名前を取得
            suffix = VIRTUAL_BONE_DEFS.get(b_name, {}).get("suffix", "前方")
            front_name = f"{b_name}{suffix}"

            if b_name in bones and front_name in bones:
                p_base = bones[b_name]['pos']
                p_front = bones[front_name]['pos']

                # --- 1. 正面・背面判定 ---
                v_char_f = (p_front - p_base) / (np.linalg.norm(p_front - p_base) + 1e-6)
                v_to_cam = (eye_pos - p_base) / (np.linalg.norm(eye_pos - p_base) + 1e-6)
                cos_pos = np.dot(v_char_f, v_to_cam)
                
                targets[f"bin_target_{b_name}_front"] = 1 if cos_pos > 0.5 else 0
                targets[f"bin_target_{b_name}_back"]  = 1 if cos_pos < -0.5 else 0

                # --- 2. 注視判定 ---
                v_to_bone = (p_base - eye_pos) / (np.linalg.norm(p_base - eye_pos) + 1e-6)
                cos_aim = np.dot(v_aim, v_to_bone)

                targets[f"bin_target_{b_name}_focused_strict"] = 1 if cos_aim > 0.985 else 0
                targets[f"bin_target_{b_name}_focused"]        = 1 if cos_aim > 0.93 else 0
                targets[f"bin_target_{b_name}_focused_loose"]  = 1 if cos_aim > 0.82 else 0

                # --- 3. 特殊: 目線 (頭のみ) ---
                if b_name == '頭':
                    targets['event_eye_contact'] = 1 if (cos_pos > 0.85 and cos_aim > 0.985) else 0
            else:
                # ボーン欠損時は全フラグを0で埋める
                targets[f"bin_target_{b_name}_front"] = 0
                targets[f"bin_target_{b_name}_back"] = 0
                targets[f"bin_target_{b_name}_focused"] = 0
                targets[f"bin_target_{b_name}_focused_strict"] = 0
                targets[f"bin_target_{b_name}_focused_loose"] = 0
                if b_name == '頭': targets['event_eye_contact'] = 0

    return targets

def detect_attitude(cam, eye_pos, bones):
    """高さ・角度・ロール判定（和名直接参照）"""
    flags = {}
    cam_y = eye_pos[1]
    
    # JSONの定義に基づき高さを判定（デフォルト値はMMD標準値）
    head_y = bones.get('頭', {}).get('pos', [0, 15, 0])[1]
    waist_y = bones.get('下半身', {}).get('pos', [0, 8, 0])[1]
    margin = 3.0 
    
    flags['height_high'] = 1 if cam_y > head_y - margin else 0
    flags['height_mid']  = 1 if waist_y - margin < cam_y < head_y + margin else 0
    flags['height_low']  = 1 if cam_y < waist_y + margin else 0

    rot_x = cam['rot'][0]
    tilt_threshold = 15.0
    flags['tilt_up']   = 1 if rot_x > tilt_threshold else 0
    flags['tilt_flat'] = 1 if abs(rot_x) <= tilt_threshold + 5.0 else 0
    flags['tilt_down'] = 1 if rot_x < -tilt_threshold else 0

    rot_z = cam['rot'][2]
    flags['is_rolled'] = 1 if abs(rot_z) > 5.0 else 0
    return flags

def detect_proximity(cam, eye_pos, bones):
    """距離感判定（光軸中央にあるボーンを基準に算出）"""
    vec_cam_aim = cam['pos'] - eye_pos
    v_aim = vec_cam_aim / (np.linalg.norm(vec_cam_aim) + 1e-6)
    fov_factor = np.tan(np.radians(15.0)) / np.tan(np.radians(cam['fov'] / 2.0))

    # anchor_definitions に登録されている全てのボーンを判定候補にする
    candidates = []
    for info in ANCHOR_DEFS.values():
        candidates.extend(info.get("bones", []))
    
    max_score = -1.0
    best_eff_dist = None

    for b_name in set(candidates): # 重複排除
        if b_name not in bones: continue
        p_bone = bones[b_name]['pos']
        vec_to_bone = p_bone - eye_pos
        dist_to_bone = np.linalg.norm(vec_to_bone)
        v_to_bone = vec_to_bone / (dist_to_bone + 1e-6)
        
        cos_aim = np.dot(v_aim, v_to_bone)
        if cos_aim <= 0: continue
        
        score = (cos_aim ** 4) * (50.0 / (dist_to_bone + 5.0))
        if score > max_score:
            max_score = score
            best_eff_dist = dist_to_bone / fov_factor

    flags = {'prox_close': 0, 'prox_mid': 0, 'prox_far': 0}
    if best_eff_dist is None:
        flags['prox_far'] = 1
    else:
        if best_eff_dist < 22.0: flags['prox_close'] = 1
        if 18.0 <= best_eff_dist < 55.0: flags['prox_mid'] = 1
        if best_eff_dist >= 48.0: flags['prox_far'] = 1
    return flags

def detect_dynamics(cam, prev_cam, eye_pos, prev_eye_pos):
    """カメラの動き判定。初回フレームやカット時も全列を0埋めで返す。"""
    
    # --- 1. 全ての出力列を0で初期化 ---
    results = {
        'event_cut': 0,
        'dyn_move_high': 0, 'dyn_move_mid': 0, 'dyn_move_low': 0,
        'dyn_rot_high': 0, 'dyn_rot_mid': 0, 'dyn_rot_low': 0,
        'dyn_zoom_in': 0, 'dyn_zoom_out': 0,
        'dyn_zoom_high': 0, 'dyn_zoom_mid': 0, 'dyn_zoom_low': 0
    }

    # 初回フレームの場合は、低速フラグだけ立てて即座に返す
    if prev_cam is None:
        results['dyn_move_low'] = 1
        results['dyn_rot_low'] = 1
        results['dyn_zoom_low'] = 1
        return results

    # --- 2. 変化量の計算 ---
    move_dist = np.linalg.norm(eye_pos - prev_eye_pos)
    rot_diff = np.abs(cam['rot'] - prev_cam['rot']).sum()
    
    # カット判定
    is_cut = (move_dist > 10.0 or rot_diff > 20.0)
    results['event_cut'] = 1 if is_cut else 0

    # カット直後のフレームも、前フレームとの連続性がないため0埋め（初期状態）で返す
    if is_cut:
        results['dyn_move_low'] = 1
        results['dyn_rot_low'] = 1
        results['dyn_zoom_low'] = 1
        return results

    # --- 3. 通常時の判定（値を上書き） ---
    # 移動速度
    results['dyn_move_high'] = 1 if move_dist > 1.2 else 0
    results['dyn_move_mid']  = 1 if 0.2 < move_dist < 1.5 else 0
    results['dyn_move_low']  = 1 if move_dist < 0.4 else 0

    # 回転速度
    results['dyn_rot_high'] = 1 if rot_diff > 2.5 else 0
    results['dyn_rot_mid']  = 1 if 0.3 < rot_diff < 3.0 else 0
    results['dyn_rot_low']  = 1 if rot_diff < 0.6 else 0

    # ズーム（FOV変化）
    fov_delta = cam['fov'] - prev_cam['fov']
    results['dyn_zoom_in']  = 1 if fov_delta < -0.05 else 0
    results['dyn_zoom_out'] = 1 if fov_delta > 0.05 else 0
    
    zoom_speed = abs(fov_delta)
    results['dyn_zoom_high'] = 1 if zoom_speed > 1.0 else 0
    results['dyn_zoom_mid']  = 1 if 0.1 < zoom_speed <= 1.0 else 0
    results['dyn_zoom_low']  = 1 if zoom_speed <= 0.1 else 0

    return results

def load_motion_data(camera_csv_path, motion_csv_path):
    df_cam = pd.read_csv(camera_csv_path)
    df_mot = pd.read_csv(motion_csv_path)
    df_combined = pd.merge(df_cam, df_mot, on='frame', suffixes=('_cam', '_mot')).sort_values('frame')

    # configから必要な全ボーン名を収集
    needed_bones = set()
    for info in ANCHOR_DEFS.values():
        for b in info.get("bones", []):
            needed_bones.add(b)
            # 仮想ボーン名も計算して追加
            suffix = VIRTUAL_BONE_DEFS.get(b, {}).get("suffix", "前方")
            needed_bones.add(f"{b}{suffix}")

    combined_data = []
    for _, row in df_combined.iterrows():
        cam_info = row.to_dict()
        cam_info['pos'] = np.array([row['pos_x'], row['pos_y'], row['pos_z']])
        cam_info['rot'] = np.array([row['rot_x'], row['rot_y'], row['rot_z']])
        cam_info['dist'] = row['distance']
        
        bones_info = {}
        for b_name in needed_bones:
            try:
                bones_info[b_name] = {
                    'pos': np.array([row[f'{b_name}_w_x'], row[f'{b_name}_w_y'], row[f'{b_name}_w_z']])
                }
            except KeyError: continue
        
        combined_data.append({'frame': int(row['frame']), 'camera': cam_info, 'bones': bones_info})
    return combined_data

def process_camera_csv(camera_csv_path, motion_csv_path, output_dir):
    data = load_motion_data(camera_csv_path, motion_csv_path)
    projector = MMDCameraProjector()
    processed_labels = []
    prev_cam, prev_eye_pos = None, None

    for frame_data in data:
        cam = frame_data['camera']
        bones = frame_data['bones']
        eye_pos = projector.get_camera_eye_pos(cam)
        
        # 各種判定の実行
        labels = {
            'frame': frame_data['frame'],
            **detect_target(cam, eye_pos, bones),
            **detect_attitude(cam, eye_pos, bones),
            **detect_proximity(cam, eye_pos, bones),
            **detect_dynamics(cam, prev_cam, eye_pos, prev_eye_pos)
        }
        processed_labels.append(labels)
        prev_cam, prev_eye_pos = cam, eye_pos

    df_output = pd.DataFrame(processed_labels)
    output_path = Path(output_dir) / Path(camera_csv_path).name
    df_output.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path.name}")

def main():
    project_root = Path(__file__).parent.parent
    camera_dir = project_root / "data" / "camera_interpolated"
    motion_dir = project_root / "data" / "motion_wide"
    output_dir = project_root / "data" / "label_csv"
    
    os.makedirs(output_dir, exist_ok=True)
    camera_files = sorted(list(camera_dir.glob("*.csv")))
    
    for cam_csv in camera_files:
        mot_csv = motion_dir / cam_csv.name
        if mot_csv.exists():
            process_camera_csv(str(cam_csv), str(mot_csv), str(output_dir))

if __name__ == "__main__":
    main()