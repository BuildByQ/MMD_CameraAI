import os
import glob
import struct
import math
import csv

def is_camera_vmd(vmd_path):
    """
    VMDファイルがカメラモーションを含むかどうかを判定
    """
    with open(vmd_path, "rb") as f:
        f.read(30)  # header
        f.read(20)  # model name

        # ボーン数を読み飛ばす
        bone_count = struct.unpack("<I", f.read(4))[0]
        f.seek(111 * bone_count, 1)

        # モーフ数を読み飛ばす
        morph_count = struct.unpack("<I", f.read(4))[0]
        f.seek(23 * morph_count, 1)

        # カメラモーション数を確認
        camera_count = struct.unpack("<I", f.read(4))[0]
        return camera_count > 0

def vmd_to_csv(vmd_path, csv_path, sort_by_frame=True, dedupe=None):
    # まず種別チェック
    if not is_camera_vmd(vmd_path):
        raise ValueError(f"{vmd_path} はカメラモーションを含まないため、camera_vmd_to_csv.pyでは処理できません。")

    with open(vmd_path, "rb") as f:
        f.read(30)  # header
        f.read(20)  # model name

        bone_count = struct.unpack("<I", f.read(4))[0]
        f.seek(111 * bone_count, 1)

        morph_count = struct.unpack("<I", f.read(4))[0]
        f.seek(23 * morph_count, 1)

        camera_count = struct.unpack("<I", f.read(4))[0]

        rows = []
        for _ in range(camera_count):
            frame = struct.unpack("<I", f.read(4))[0]
            distance = struct.unpack("<f", f.read(4))[0]
            pos = struct.unpack("<3f", f.read(12))
            rot_rad = struct.unpack("<3f", f.read(12))
            rot = tuple(r * 180.0 / math.pi for r in rot_rad)
            interp = list(f.read(24))  # 補間データ（暫定）
            fov = struct.unpack("<I", f.read(4))[0]
            f.read(1)  # flag

            row = [frame, *pos, *rot, distance, fov, *interp]
            rows.append(row)

    # 並べ替え
    if sort_by_frame:
        rows.sort(key=lambda r: r[0])

    # 重複フレーム処理
    if dedupe in ("keep_first", "keep_last"):
        seen = {}
        for r in rows:
            frame = r[0]
            if dedupe == "keep_first":
                if frame not in seen:
                    seen[frame] = r
            else:  # keep_last
                seen[frame] = r
        rows = list(seen.values())
        rows.sort(key=lambda r: r[0])

    # CSV出力
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = [
            "frame", "pos_x", "pos_y", "pos_z",
            "rot_x", "rot_y", "rot_z",
            "distance", "fov",
            # X軸補間パラメータ
            "X_x1", "X_x2", "X_y1", "X_y2",
            # Y軸補間パラメータ
            "Y_x1", "Y_x2", "Y_y1", "Y_y2",
            # Z軸補間パラメータ
            "Z_x1", "Z_x2", "Z_y1", "Z_y2",
            # 回転補間パラメータ
            "R_x1", "R_x2", "R_y1", "R_y2",
            # 距離補間パラメータ
            "L_x1", "L_x2", "L_y1", "L_y2",
            # 視野角補間パラメータ
            "V_x1", "V_x2", "V_y1", "V_y2"
        ]
        writer.writerow(header)
        writer.writerows(rows)

def batch_vmd_to_csv(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for vmd_file in glob.glob(os.path.join(input_dir, "*.vmd")):
        base = os.path.splitext(os.path.basename(vmd_file))[0]
        csv_file = os.path.join(output_dir, base + ".csv")
        print(f"Converting {vmd_file} → {csv_file}")
        try:
            vmd_to_csv(vmd_file, csv_file)
        except ValueError as e:
            print(f"スキップ: {e}")

batch_vmd_to_csv("./data/camera_vmd", "./data/camera_csv")