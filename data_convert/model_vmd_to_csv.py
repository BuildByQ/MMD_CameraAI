import struct
import csv
from pathlib import Path

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT.parent / 'data' / 'motion_vmd'
OUTPUT_DIR = PROJECT_ROOT.parent / 'data' / 'motion_csv'

def is_model_vmd(vmd_path):
    """VMDファイルがボーンモーションを含むかどうかを判定"""
    try:
        with open(vmd_path, "rb") as f:
            f.read(30)  # header
            f.read(20)  # model name
            bone_count = struct.unpack("<I", f.read(4))[0]
            return bone_count > 0
    except (struct.error, OSError):
        return False

def model_vmd_to_csv(vmd_path, csv_path):
    """
    1つのモデルVMDを解析してCSVに変換。
    ※ボーン名は Shift-JIS でデコード。
    """
    with open(vmd_path, "rb") as f:
        f.read(30)  # header
        f.read(20)  # model name
        bone_count = struct.unpack("<I", f.read(4))[0]

        rows = []
        for _ in range(bone_count):
            # ボーン名 (15 bytes)
            bone_name = f.read(15).decode("shift_jis", errors="ignore").strip("\x00")
            frame = struct.unpack("<I", f.read(4))[0]
            pos = struct.unpack("<3f", f.read(12))      # x, y, z
            rot = struct.unpack("<4f", f.read(16))      # quaternion (x, y, z, w)
            interp = list(f.read(64))                  # 補間データ (64 bytes)

            rows.append([bone_name, frame, *pos, *rot, *interp])

    # フレーム順でソート
    rows.sort(key=lambda r: r[1])

    # ヘッダー構築 (補間パラメータ 64個分)
    interp_headers = [f"interp_{i}" for i in range(64)]
    # MMDの仕様に基づき、先頭16個を意味のある名前に置換
    interp_mapping = {
        0: 'X_x1', 4: 'X_y1', 8: 'X_x2', 12: 'X_y2',
        1: 'Y_x1', 5: 'Y_y1', 9: 'Y_x2', 13: 'Y_y2',
        2: 'Z_x1', 6: 'Z_y1', 10: 'Z_x2', 14: 'Z_y2',
        3: 'R_x1', 7: 'R_y1', 11: 'R_x2', 15: 'R_y2'
    }
    for i, name in interp_mapping.items():
        interp_headers[i] = name

    header = ["bone_name", "frame", "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w"] + interp_headers

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)

def main():
    """未処理のモーションVMDを一括変換"""
    print(f"[Step 4] Model VMD -> CSV 変換チェック開始")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    vmd_files = list(INPUT_DIR.glob("*.vmd"))
    if not vmd_files:
        print(f"VMDファイルが見つかりません: {INPUT_DIR}")
        return

    processed_count = 0
    skipped_count = 0

    for vmd_file in vmd_files:
        csv_file = OUTPUT_DIR / (vmd_file.stem + ".csv")
        
        # --- 既存チェック ---
        if csv_file.exists():
            skipped_count += 1
            continue
            
        if not is_model_vmd(vmd_file):
            print(f"⏭️  非モデルVMDのためスキップ: {vmd_file.name}")
            continue
            
        try:
            model_vmd_to_csv(vmd_file, csv_file)
            print(f"変換完了: {vmd_file.name}")
            processed_count += 1
        except Exception as e:
            print(f"エラー ({vmd_file.name}): {e}")

    print(f"\n処理結果: 新規変換 {processed_count} 件 / スキップ {skipped_count} 件")

if __name__ == "__main__":
    main()