import os
import glob
import struct
import csv

def is_model_vmd(vmd_path):
    """
    VMDファイルがボーンモーションを含むかどうかを判定
    """
    with open(vmd_path, "rb") as f:
        f.read(30)  # header
        f.read(20)  # model name

        # ボーンモーション数を確認
        bone_count = struct.unpack("<I", f.read(4))[0]
        return bone_count > 0

def model_vmd_to_csv(vmd_path, csv_path, sort_by_frame=True):
    # まず種別チェック
    if not is_model_vmd(vmd_path):
        raise ValueError(f"{vmd_path} はボーンモーションを含まないため、model_vmd_to_csv.pyでは処理できません。")

    with open(vmd_path, "rb") as f:
        f.read(30)  # header
        f.read(20)  # model name

        bone_count = struct.unpack("<I", f.read(4))[0]
        rows = []
        for _ in range(bone_count):
            bone_name = f.read(15).decode("shift_jis", errors="ignore").strip("\x00")
            frame = struct.unpack("<I", f.read(4))[0]
            pos = struct.unpack("<3f", f.read(12))
            rot = struct.unpack("<4f", f.read(16))  # quaternion
            interp = list(f.read(64))  # 補間データ

            row = [bone_name, frame, *pos, *rot, *interp]
            rows.append(row)

        # morph section skip
        morph_count = struct.unpack("<I", f.read(4))[0]
        f.seek(23 * morph_count, 1)

        # camera section skip
        cam_count = struct.unpack("<I", f.read(4))[0]
        f.seek(61 * cam_count, 1)

    if sort_by_frame:
        rows.sort(key=lambda r: r[1])  # frame列でソート

    # 補間パラメータのマッピング
    interp_headers = [f"interp_{i}" for i in range(64)]
    interp_mapping = {
        0: 'X_x1', 4: 'X_y1', 8: 'X_x2', 12: 'X_y2',  # X軸補間
        1: 'Y_x1', 5: 'Y_y1', 9: 'Y_x2', 13: 'Y_y2',  # Y軸補間
        2: 'Z_x1', 6: 'Z_y1', 10: 'Z_x2', 14: 'Z_y2', # Z軸補間
        3: 'R_x1', 7: 'R_y1', 11: 'R_x2', 15: 'R_y2'  # 回転補間
    }
    # ヘッダーを更新
    for i, name in interp_mapping.items():
        if i < len(interp_headers):
            interp_headers[i] = name
    header = ["bone_name", "frame",
            "pos_x", "pos_y", "pos_z",
            "rot_x", "rot_y", "rot_z", "rot_w"] + interp_headers

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

def batch_model_vmd_to_csv(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for vmd_file in glob.glob(os.path.join(input_dir, "*.vmd")):
        base = os.path.splitext(os.path.basename(vmd_file))[0]
        csv_file = os.path.join(output_dir, base + ".csv")
        print(f"Converting {vmd_file} → {csv_file}")
        try:
            model_vmd_to_csv(vmd_file, csv_file)
        except ValueError as e:
            print(f"スキップ: {e}")

batch_model_vmd_to_csv("./data/motion_vmd", "./data/motion_csv")