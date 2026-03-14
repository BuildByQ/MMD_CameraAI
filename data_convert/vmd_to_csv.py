import struct
import csv
import sys
from pathlib import Path
import math

def read_vmd_camera(file_path, output_csv):
    with open(file_path, "rb") as f:
        # ヘッダ読み込み
        header = f.read(30).decode("shift_jis", errors="ignore")
        model_name = f.read(20).decode("shift_jis", errors="ignore")
        print(f"Header: {header.strip()}, Model: {model_name.strip()}")

        # カメラデータ数
        camera_count = struct.unpack("<I", f.read(4))[0]
        camera_count = struct.unpack("<I", f.read(4))[0]
        camera_count = struct.unpack("<I", f.read(4))[0]
        print(f"Camera keyframes: {camera_count}")

        rows = []
        for _ in range(camera_count):
            frame = struct.unpack("<I", f.read(4))[0]
            distance = struct.unpack("<f", f.read(4))[0]
            pos = struct.unpack("<3f", f.read(12))   # X,Y,Z
            rot_rad = struct.unpack("<3f", f.read(12))   # Pitch, Yaw, Roll (radian)
            rot = tuple(r * 180.0 / math.pi for r in rot_rad)
            # fov = struct.unpack("<I", f.read(4))[0]

            # 補間曲線 (ここは仕様に基づきさらに読み込む必要あり)
            interp = f.read(144)  # 仮に24バイト確保（詳細は仕様参照）

            rows.append([frame, *pos, *rot, distance, fov])

        # CSV出力
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["frame","pos_x","pos_y","pos_z",
                             "rot_x","rot_y","rot_z",
                             "distance","fov"])
            writer.writerows(rows)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vmd_to_csv.py input.vmd output.csv")
    else:
        read_vmd_camera(sys.argv[1], sys.argv[2])