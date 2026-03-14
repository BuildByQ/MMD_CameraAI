import pandas as pd
import numpy as np
import os
from pathlib import Path
import math

# --- 設定項目 (必要に応じて調整) ---
# ボーンCSVの列命名規則: "head_x", "head_y", "head_z", "neck_x", ... と想定
TARGET_BONES = ['head', 'neck', 'upper_body', 'lower_body', 'hand_l', 'hand_r']

import math
import numpy as np


# ボーン名と、それに対応させるラベル用キーのマップ
# 右側の名前（ラベル用）は後で英語にしたほうがMLで扱いやすいですが、
# 今はCSVの列名に合わせます。
BONE_MAP = {
    '頭': 'head',
    '首': 'neck',
    '上半身': 'upper_body',
    '下半身': 'lower_body',
    'センター': 'center',
    '右手首': 'hand_r',
    '左手首': 'hand_l',
    '右ひじ': 'elbow_r',  # 追加
    '左ひじ': 'elbow_l',  # 追加
    '右足首': 'foot_r',
    '左足首': 'foot_l',
    '右ひざ': 'knee_r',   # 追加
    '左ひざ': 'knee_l',   # 追加
    "head_front": "head_front", # 仮想ボーンもマップに加える
    "upper_body_front": "upper_body_front",
    "lower_body_front": "lower_body_front"
}

class MMDCameraProjector:
    def __init__(self, aspect_ratio=16/9):
        self.aspect_ratio = aspect_ratio
        # 共有いただいた多項式係数
        self.mmd_values = np.array([10, 20, 40, 60, 80, 100, 120])
        self.actual_values = np.array([20.0, 39.6, 73.0, 99.4, 119.6, 136.0, 149.3])
        self.fov_poly_coeffs = np.polyfit(self.mmd_values, self.actual_values, 3)

    def mmd_fov_to_actual_degrees(self, mmd_fov: float) -> float:
        mmd_fov = max(0, min(180, float(mmd_fov)))
        return max(0, min(180, np.polyval(self.fov_poly_coeffs, mmd_fov)))

    def get_camera_eye_pos(self, cam):
        target_pos = cam['pos']
        
        # デバッグ用の値（後で cam['dist'] に戻してください）
        # dist が -45.0 なら、init_offset は [0, 0, 45] となり、注視点の後ろになります。
        # ここでは監督の定義「角度0で [0, 10, -45]」に従い、
        # dist が -45.0 の時に init_offset を [0, 0, -dist] とすることで [0, 0, 45] になってしまうのを防ぎます。
        dist = cam['dist'] 

        # 1. 初期位置ベクトル
        # 角度[0,0,0], dist=-45 の時、[0, 10, -45] にしたいのであれば、
        # そのまま dist を Z に入れれば [0, 0, -45] になります。
        init_offset = np.array([0, 0, dist])

        # 2. 回転行列の定義
        rx, ry, rz = np.radians(cam['rot'])
        
        Ry = np.array([
            [ np.cos(ry), 0,-np.sin(ry)],
            [ 0,          1, 0         ],
            [ np.sin(ry), 0, np.cos(ry)]
        ])
        Rx = np.array([
            [1, 0,           0          ],
            [0, np.cos(rx), -np.sin(rx) ],
            [0, np.sin(rx),  np.cos(rx) ]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [0,           0,          1]
        ])

        # 3. 回転の合成順序を修正 (MMDカメラは通常 Y * X * Z)
        # 行列積 A @ B @ C は C -> B -> A の順で適用されます。
        # 「Z軸ロール -> X軸ピッチ -> Y軸ヨー」の順で回転させる場合：
        combined_rotation = Ry @ Rx @ Rz
        
        rotated_offset = combined_rotation @ init_offset
        
        # 4. 最終座標
        eye_pos = target_pos + rotated_offset
        return eye_pos

    def get_screen_projections(self, cam, bones):
        """ボーン座標をスクリーン(0.0~1.0)に投影"""
        eye_pos = self.get_camera_eye_pos(cam)
        target_pos = cam['pos']
        
        # 1. ビュー行列 (LookAt)
        # 上方向ベクトルは通常 Y軸(0,1,0)
        up = np.array([0, 1, 0])
        z_axis = (target_pos - eye_pos) / np.linalg.norm(target_pos - eye_pos)
        x_axis = np.cross(up, z_axis) / np.linalg.norm(np.cross(up, z_axis))
        y_axis = np.cross(z_axis, x_axis)
        
        view_matrix = np.eye(4)
        view_matrix[0, :3], view_matrix[0, 3] = x_axis, -np.dot(x_axis, eye_pos)
        view_matrix[1, :3], view_matrix[1, 3] = y_axis, -np.dot(y_axis, eye_pos)
        view_matrix[2, :3], view_matrix[2, 3] = z_axis, -np.dot(z_axis, eye_pos)

        # 2. プロジェクション行列
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
                # 0~1に変換
                # --- 修正後 ---
                projections[name] = {
                    'x': (ndc_x + 1.0) / 2.0,
                    'y': (1.0 - ndc_y) / 2.0,
                    'z': clip_pos[2],
                    'world_pos': bone['pos'],  # 元の 3次元 [x, y, z] を入れる
                    'in_view': (abs(ndc_x) <= 1.0 and abs(ndc_y) <= 1.0 and clip_pos[3] > 0)
                }
        return projections


def load_motion_data(camera_csv_path: str, motion_csv_path: str):
    df_cam = pd.read_csv(camera_csv_path)
    df_mot = pd.read_csv(motion_csv_path)

    df_combined = pd.merge(df_cam, df_mot, on='frame', suffixes=('_cam', '_mot'))
    df_combined = df_combined.sort_values('frame').reset_index(drop=True)

    combined_data = []
    
    for _, row in df_combined.iterrows():
        # --- ここを修正：rowを辞書に変換し、必要な配列を上書きして保持 ---
        cam_info = row.to_dict() 
        
        # 既存のロジックが壊れないよう、numpy配列版も保持
        cam_info['pos'] = np.array([row['pos_x'], row['pos_y'], row['pos_z']])
        cam_info['rot'] = np.array([row['rot_x'], row['rot_y'], row['rot_z']])
        # fov, distance は row.to_dict() で既に入っているのでそのままでOK
        # ここが重要：'distance' を 'dist' として参照できるようにする
        cam_info['dist'] = row['distance']
        # 'fov' は元々同じ名前なのでそのままでOK
        
        bones_info = {}
        for mmd_name, label_name in BONE_MAP.items():
            try:
                bones_info[label_name] = {
                    'pos': np.array([
                        row[f'{mmd_name}_w_x'], 
                        row[f'{mmd_name}_w_y'], 
                        row[f'{mmd_name}_w_z']
                    ])
                }
            except KeyError:
                continue
        
        combined_data.append({
            'frame': int(row['frame']),
            'camera': cam_info,
            'bones': bones_info
        })
        
    return combined_data

def process_camera_csv(camera_csv_path: str, motion_csv_path: str, output_dir: str):
    # 1. データのロード
    data = load_motion_data(camera_csv_path, motion_csv_path)
    projector = MMDCameraProjector()
    # (今後 analyzer クラスにまとめても良いですが、一旦関数として呼び出す想定)
    processed_labels = []

    # --- 前フレーム保持用変数 ---
    prev_cam = None
    prev_eye_pos = None
    
    for i in range(len(data)):
        if i == 1645:
            print('debug')
        
        frame_data = data[i]
        cam = frame_data['camera']
        bones = frame_data['bones']
        
        # --- A. 物理的な座標の算出 ---
        # 真のカメラ位置 (Eye Position)
        eye_pos = projector.get_camera_eye_pos(cam)
        
        # スクリーン投影 (各ボーンが画面のどこにいるか)
        projections = projector.get_screen_projections(cam, bones)
        
        # --- B. ラベル判定関数の呼び出し ---
        
        # 1. ターゲット判定 (前回のロジック)
        # target_labels = detect_targets(projections)
        
        # 2. アティチュード判定 (今回追加したもの)
        # 引数に cam, eye_pos, bones を渡す
        attitude_labels = detect_attitude(cam, eye_pos, bones)

        # 引数に cam, eye_pos, bones を渡す
        target_labels = detect_target(cam, eye_pos, bones)
        
        # 3. 距離感判定 (今回追加したもの)
        # 引数に projections を渡す
        proximity_labels = detect_proximity(cam, eye_pos, bones)

        # 4. ダイナミクス判定 (前フレームの情報を使用)
        dynamics_labels = detect_dynamics(cam, prev_cam, eye_pos, prev_eye_pos)
        
        # 判定が終わったら、現在の状態を「前フレーム」として保存
        prev_cam = cam
        prev_eye_pos = eye_pos
        
        # --- C. データの統合 ---
        label_dict = {
            'frame': frame_data['frame'],
            **target_labels,
            **attitude_labels,
            **proximity_labels,
            **dynamics_labels
        }
        processed_labels.append(label_dict)

    # --- D. CSV出力処理 ---
    # リストを一気にDataFrameに変換
    df_output = pd.DataFrame(processed_labels)
    
    # 元のファイル名を利用して出力パスを作成 (例: camera_001.csv -> label_001.csv)
    input_filename = Path(camera_csv_path).name
    # もしファイル名を変更したい場合はここで調整
    output_path = Path(output_dir) / input_filename
    
    # 保存実行 (index=False で行番号の出力を防ぐ)
    df_output.to_csv(output_path, index=False)
    
    print(f"  ✓ Label CSV saved: {output_path} ({len(df_output)} frames)")

def detect_attitude(cam, eye_pos, bones):
    """
    カメラの高さ(Height)と角度(Tilt/Roll)を判定
    """
    flags = {}
    
    # --- A. Height (カメラのY座標 vs キャラクターの各部位) ---
    cam_y = eye_pos[1]
    # ボーンが取得できない場合のフォールバック（MMD標準的な高さ）
    head_y = bones.get('head', {}).get('pos', [0, 15, 0])[1]
    waist_y = bones.get('lower_body', {}).get('pos', [0, 8, 0])[1]
    foot_y = bones.get('foot_r', {}).get('pos', [0, 0, 0])[1]

    # 閾値（この範囲内ならフラグを立てる / 重なりを許容）
    margin = 3.0 
    
    flags['height_high'] = 1 if cam_y > head_y - margin else 0
    flags['height_mid']  = 1 if waist_y - margin < cam_y < head_y + margin else 0
    flags['height_low']  = 1 if cam_y < waist_y + margin else 0

    # --- B. Tilt (垂直角: カメラの回転X) ---
    # MMDのカメラ回転Xは、下向きがマイナス、上向きがプラス（または逆）なため、符号に注意
    rot_x = cam['rot'][0]
    tilt_threshold = 15.0 # 15度以上で傾きと判定
    
    flags['tilt_up']   = 1 if rot_x > tilt_threshold else 0
    flags['tilt_flat'] = 1 if abs(rot_x) <= tilt_threshold + 5.0 else 0 # 遊びを持たせる
    flags['tilt_down'] = 1 if rot_x < -tilt_threshold else 0

    # --- C. Roll (回転Z) ---
    rot_z = cam['rot'][2]
    flags['is_rolled'] = 1 if abs(rot_z) > 5.0 else 0 # 5度以上の傾き

    return flags

def detect_proximity(cam, eye_pos, bones):
    """
    光軸ベクトルとの親和性から『主役ボーン』を特定し、
    その実質距離（FOV補正込み）で距離感を判定する。
    """
    # 1. 信頼できる光軸ベクトル（カメラから注視点へ向かう向き）
    # MMDの仕様上、注視点の奥行きはズレていても「向き」は正確
    vec_cam_aim = cam['pos'] - eye_pos
    v_aim = vec_cam_aim / (np.linalg.norm(vec_cam_aim) + 1e-6)

    # 2. FOVによるズーム補正係数の計算（FOV 30度を基準 [ズーム倍率1.0] とする）
    # tan(15deg) / tan(現在のFOV / 2)
    fov_factor = np.tan(np.radians(15.0)) / np.tan(np.radians(cam['fov'] / 2.0))

    # 3. 判定対象のボーンと基礎重要度
    target_bones = {
        'head': 1.5, 'upper_body': 1.2, 'lower_body': 1.0,
        'elbow_r': 0.9, 'elbow_l': 0.9, 'hand_r': 0.6, 'hand_l': 0.6,
        'knee_r': 0.9, 'knee_l': 0.9, 'foot_r': 0.6, 'foot_l': 0.6
    }

    max_score = -1.0
    best_eff_dist = None

    # 4. 「今、カメラが向いている方向（光軸）」に最も近いボーンを主役として選別
    for name, base_weight in target_bones.items():
        if name not in bones:
            continue
        
        p_bone = bones[name]['pos']
        vec_to_bone = p_bone - eye_pos
        dist_to_bone = np.linalg.norm(vec_to_bone)
        v_to_bone = vec_to_bone / (dist_to_bone + 1e-6)

        # A. 光軸との一致度 (内積: 1.0に近いほど画面中央)
        # 2D投影座標の (0.5, 0.5) からの距離を測るのと同義
        cos_aim = np.dot(v_aim, v_to_bone)
        
        # カメラの真後ろにあるボーンは除外
        if cos_aim <= 0:
            continue

        # B. 物理距離の優位性 (近いほど高得点)
        dist_score = 50.0 / (dist_to_bone + 5.0) 
        
        # C. 総合スコア（中央度 × 近さ × 部位重要度）
        # ※中央度には少し高い累乗(例: 4乗)をかけると、より「中央のものを優先」しやすくなります
        score = (cos_aim ** 4) * dist_score * base_weight
        
        if score > max_score:
            max_score = score
            # 採用されたボーンの「実質的な寄り具合」を算出 (物理距離 / ズーム倍率)
            best_eff_dist = dist_to_bone / fov_factor

    # 5. 判定フラグの導出
    flags = {'prox_close': 0, 'prox_mid': 0, 'prox_far': 0}

    if best_eff_dist is None:
        flags['prox_far'] = 1
        return flags

    # 閾値判定 (監督の設定値を維持)
    if best_eff_dist < 22.0:
        flags['prox_close'] = 1
    
    if 18.0 <= best_eff_dist < 55.0:
        flags['prox_mid'] = 1
        
    if best_eff_dist >= 48.0:
        flags['prox_far'] = 1

    return flags

def detect_dynamics(cam, prev_cam, eye_pos, prev_eye_pos):
    # 補間列の定義
    interpolation_cols = [
        'X_x1', 'X_x2', 'X_y1', 'X_y2', 'Y_x1', 'Y_x2', 'Y_y1', 'Y_y2',
        'Z_x1', 'Z_x2', 'Z_y1', 'Z_y2', 'R_x1', 'R_x2', 'R_y1', 'R_y2',
        'L_x1', 'L_x2', 'L_y1', 'L_y2', 'V_x1', 'V_x2', 'V_y1', 'V_y2'
    ]
    
    # --- 1. 状態判定 ---
    is_keyframe = any(cam.get(col, 0) != 0 for col in interpolation_cols)
    
    move_dist_raw = 0
    rot_diff_raw = 0
    if prev_eye_pos is not None:
        move_dist_raw = np.linalg.norm(eye_pos - prev_eye_pos)
        rot_diff_raw = np.abs(cam['rot'] - prev_cam['rot']).sum()

    is_cut = is_keyframe and (move_dist_raw > 10.0 or rot_diff_raw > 20.0)

    # 初回またはカット時の早期リターン
    if prev_cam is None or is_cut:
        return {
            'event_cut': 1 if is_cut else 0,
            'dyn_move_low': 1, 'dyn_move_mid': 0, 'dyn_move_high': 0,
            'dyn_rot_low': 1, 'dyn_rot_mid': 0, 'dyn_rot_high': 0,
            'dyn_zoom_in': 0, 'dyn_zoom_out': 0,
            'dyn_zoom_low': 1, 'dyn_zoom_mid': 0, 'dyn_zoom_high': 0
        }

    # --- 2. 通常の計算 ---
    results = {'event_cut': 0}

    # 移動
    results['dyn_move_high'] = 1 if move_dist_raw > 1.2 else 0
    results['dyn_move_mid']  = 1 if 0.2 < move_dist_raw < 1.5 else 0
    results['dyn_move_low']  = 1 if move_dist_raw < 0.4 else 0

    # 回転
    results['dyn_rot_high'] = 1 if rot_diff_raw > 2.5 else 0
    results['dyn_rot_mid']  = 1 if 0.3 < rot_diff_raw < 3.0 else 0
    results['dyn_rot_low']  = 1 if rot_diff_raw < 0.6 else 0

    # ズーム
    fov_delta = cam['fov'] - prev_cam['fov']
    actual_dist_delta = abs(cam['dist']) - abs(prev_cam['dist'])
    zoom_trend = fov_delta + (actual_dist_delta * 0.5)

    results['dyn_zoom_in']  = 1 if zoom_trend < -0.05 else 0
    results['dyn_zoom_out'] = 1 if zoom_trend > 0.05 else 0
    
    zoom_speed = abs(zoom_trend)
    results['dyn_zoom_high'] = 1 if zoom_speed > 1.0 else 0
    results['dyn_zoom_mid']  = 1 if 0.1 < zoom_speed <= 1.0 else 0
    results['dyn_zoom_low']  = 1 if zoom_speed <= 0.1 else 0

    return results

def get_camera_front_vector(rot_deg):
    """
    注視点座標を使わず、回転値(rot)からカメラの真の視線方向ベクトルを出す
    """
    rx, ry, rz = np.radians(rot_deg)
    
    # MMDカメラの回転（Y->X->Z）から、初期方向(0,0,1)を回転させたベクトルを計算
    vx =  np.sin(ry) * np.cos(rx)
    vy = -np.sin(rx)
    vz =  np.cos(ry) * np.cos(rx)
    
    vec = np.array([vx, vy, vz])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.array([0, 0, 1])

def detect_target(cam, eye_pos, bones):
    """
    カメラの光軸に基づいた注視判定を、独立したバイナリフラグとして算出。
    MLモデルが正規化なしで「注視の精度」を学習できるように設計。
    """
    targets = {}
    
    # 1. 光軸ベクトル（カメラから注視点へ）
    vec_cam_aim = cam['pos'] - eye_pos
    v_aim = vec_cam_aim / (np.linalg.norm(vec_cam_aim) + 1e-6)
    
    check_sets = [
        ('head', 'head_front', 'target_head'),
        ('upper_body', 'upper_body_front', 'target_body'),
        ('lower_body', 'lower_body_front', 'target_leg')
    ]

    for base_key, front_key, label_prefix in check_sets:
        if base_key in bones and front_key in bones:
            p_base = bones[base_key]['pos']
            p_front = bones[front_key]['pos']
            
            # --- A. 相対位置判定 (正面・背面) ---
            vec_char_f = p_front - p_base
            v_char_f = vec_char_f / (np.linalg.norm(vec_char_f) + 1e-6)
            vec_to_cam = eye_pos - p_base
            v_to_cam = vec_to_cam / (np.linalg.norm(vec_to_cam) + 1e-6)
            
            cos_pos = np.dot(v_char_f, v_to_cam)
            targets[f'{label_prefix}_front'] = 1 if cos_pos > 0.5 else 0
            targets[f'{label_prefix}_back']  = 1 if cos_pos < -0.5 else 0

            # --- B. 注視判定 (独立した3段階のバイナリフラグ) ---
            vec_to_bone = p_base - eye_pos
            v_to_bone = vec_to_bone / (np.linalg.norm(vec_to_bone) + 1e-6)
            cos_aim = np.dot(v_aim, v_to_bone)

            # 1. Strict: 約10度以内 (ド真ん中の決定的なカット)
            targets[f'{label_prefix}_focused_strict'] = 1 if cos_aim > 0.985 else 0
            
            # 2. Focused: 約21.5度以内 (狙っている状態。19.2度をカバーし、跳ねを抑制)
            targets[f'{label_prefix}_focused'] = 1 if cos_aim > 0.93 else 0
            
            # 3. Loose: 約35度以内 (そちらの方向へカメラを向けている)
            targets[f'{label_prefix}_focused_loose'] = 1 if cos_aim > 0.82 else 0

            # --- C. 複合フラグ (目線判定) ---
            if label_prefix == 'target_head':
                # 正面を向いており、かつカメラがド真ん中(Strict)で捉えている時
                targets['event_eye_contact'] = 1 if (cos_pos > 0.85 and targets[f'{label_prefix}_focused_strict'] == 1) else 0
        else:
            # ボーン欠損時のセーフティ
            targets[f'{label_prefix}_front'] = 0
            targets[f'{label_prefix}_back'] = 0
            targets[f'{label_prefix}_focused_strict'] = 0
            targets[f'{label_prefix}_focused'] = 0
            targets[f'{label_prefix}_focused_loose'] = 0
            if label_prefix == 'target_head':
                targets['event_eye_contact'] = 0
                
    return targets

def main():
    """メイン処理：ディレクトリ内のファイルをスキャンして回す"""
    project_root = Path(__file__).parent.parent
    # あなたの構成に合わせたパス設定
    camera_csv_dir = project_root / "data" / "camera_interpolated"
    motion_csv_dir = project_root / "data" / "motion_wide"
    output_dir = project_root / "data" / "label_csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    camera_csv_files = sorted(list(camera_csv_dir.glob("*.csv")))
    print(f"処理を開始します。{len(camera_csv_files)}個のファイルを解析します。")
    
    processed_count = 0
    for camera_csv in camera_csv_files:
        try:
            # camera_001.csv -> 001.csv -> 001_motion.csv (または同名)
            # 提示された設計に合わせて、カメラCSVと同名のモーションCSVを探す
            motion_csv = motion_csv_dir / camera_csv.name 
            
            if not motion_csv.exists():
                print(f" 警告: モーションファイル欠損: {motion_csv.name}")
                continue
                
            print(f"解析中: {camera_csv.name}")
            process_camera_csv(str(camera_csv), str(motion_csv), str(output_dir))
            processed_count += 1
            
        except Exception as e:
            print(f" エラー {camera_csv.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    print(f"\n完了: {processed_count}個のファイルを処理しました。")

if __name__ == "__main__":
    main()