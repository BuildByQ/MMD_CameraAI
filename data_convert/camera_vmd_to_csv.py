#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import struct
import math
import csv
from pathlib import Path

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT.parent / 'data' / 'camera_vmd'
OUTPUT_DIR = PROJECT_ROOT.parent / 'data' / 'camera_csv'

def is_camera_vmd(vmd_path):
    """VMDファイルがカメラモーションを含むかどうかを判定"""
    with open(vmd_path, "rb") as f:
        f.read(30)  # header
        f.read(20)  # model name
        
        # ボーン・モーフをシーク
        try:
            bone_count = struct.unpack("<I", f.read(4))[0]
            f.seek(111 * bone_count, 1)
            morph_count = struct.unpack("<I", f.read(4))[0]
            f.seek(23 * morph_count, 1)
            camera_count = struct.unpack("<I", f.read(4))[0]
            return camera_count > 0
        except (struct.error, OverflowError):
            return False

def vmd_to_csv(vmd_path, csv_path):
    """
    1つのVMDファイルを解析してCSVに書き出す
    重複フレームは後勝ち (keep_last)
    """
    with open(vmd_path, "rb") as f:
        f.read(30) # header
        f.read(20) # model name
        bone_count = struct.unpack("<I", f.read(4))[0]
        f.seek(111 * bone_count, 1)
        morph_count = struct.unpack("<I", f.read(4))[0]
        f.seek(23 * morph_count, 1)
        camera_count = struct.unpack("<I", f.read(4))[0]

        rows_dict = {}
        for _ in range(camera_count):
            frame = struct.unpack("<I", f.read(4))[0]
            distance = struct.unpack("<f", f.read(4))[0]
            pos = struct.unpack("<3f", f.read(12))
            rot_rad = struct.unpack("<3f", f.read(12))
            rot = [r * 180.0 / math.pi for r in rot_rad]
            interp = list(f.read(24))
            fov = struct.unpack("<I", f.read(4))[0]
            f.read(1)
            
            row = [frame, *pos, *rot, distance, fov, *interp]
            rows_dict[frame] = row

    sorted_rows = [rows_dict[f] for f in sorted(rows_dict.keys())]

    header = [
        "frame", "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "distance", "fov",
        'X_x1', 'X_x2', 'X_y1', 'X_y2', # X軸補間
        'Y_x1', 'Y_x2', 'Y_y1', 'Y_y2', # Y軸補間
        'Z_x1', 'Z_x2', 'Z_y1', 'Z_y2', # Z軸補間
        'R_x1', 'R_x2', 'R_y1', 'R_y2', # 回転補間
        'L_x1', 'L_x2', 'L_y1', 'L_y2', # 距離補間
        'V_x1', 'V_x2', 'V_y1', 'V_y2'  # FOV補間
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(sorted_rows)

def main():
    """ディレクトリ内の全VMDをチェックし、未処理分のみ変換"""
    print(f"[Step 2] Camera VMD -> CSV 変換チェック開始")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    vmd_files = list(INPUT_DIR.glob("*.vmd"))
    if not vmd_files:
        print(f"VMDファイルが見つかりません: {INPUT_DIR}")
        return

    processed_count = 0
    skipped_count = 0

    for vmd_file in vmd_files:
        csv_file = OUTPUT_DIR / (vmd_file.stem + ".csv")
        
        # --- 追加機能: 既存チェック ---
        if csv_file.exists():
            skipped_count += 1
            # 大量のログが出過ぎないよう、スキップ時は表示を簡略化
            continue
            
        if not is_camera_vmd(vmd_file):
            print(f"非カメラVMDのためスキップ: {vmd_file.name}")
            continue
            
        try:
            vmd_to_csv(vmd_file, csv_file)
            print(f"変換完了: {vmd_file.name}")
            processed_count += 1
        except Exception as e:
            print(f"エラー ({vmd_file.name}): {e}")

    print(f"\n処理結果: 新規変換 {processed_count} 件 / スキップ {skipped_count} 件")

if __name__ == "__main__":
    main()