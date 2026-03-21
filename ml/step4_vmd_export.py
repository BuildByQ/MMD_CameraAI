import struct
import math
import pandas as pd
from pathlib import Path

# --- ディレクトリ設定 (main.pyからの相対パス、または絶対パス) ---
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT.parent / 'output' / 'Step3' 
OUTPUT_DIR = PROJECT_ROOT.parent / 'output' / 'Step4_vmd' 

def run_vmd_export(song_id):
    """
    main.py から呼び出されるメイン関数。
    特定の song_id に基づいて CSV を VMD に変換する。
    """
    # 入力ファイルパス (Step 3 の出力ファイル名に合わせる)
    csv_path = INPUT_DIR / f"step3_ready_{song_id}.csv"
    
    # 出力ファイルパス
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vmd_path = OUTPUT_DIR / f"{song_id}.vmd"

    if not csv_path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # 補間パラメータの列定義
    interp_targets = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'distance', 'fov']
    suffixes = ['_x1', '_x2', '_y1', '_y2']

    with open(vmd_path, "wb") as f:
        # 1. ヘッダー (30 bytes)
        f.write("Vocaloid Motion Data 0002\0".encode('shift-jis').ljust(30, b'\0'))

        # 2. モデル名 (20 bytes)
        f.write("カメラ・照明".encode('shift-jis').ljust(20, b'\0'))

        # 3. ボーン数, 4. モーフ数 (各4 bytes, カメラなので 0)
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

        # 5. カメラキーフレーム数 (4 bytes)
        f.write(struct.pack("<I", len(df)))

        # 6. カメラレコード
        for _, row in df.iterrows():
            f.write(struct.pack("<I", int(row['frame'])))
            f.write(struct.pack("<f", float(row['distance'])))
            f.write(struct.pack("<3f", float(row['pos_x']), float(row['pos_y']), float(row['pos_z'])))
            
            # 回転 (度数法 -> ラジアン)
            rot_rad = [float(row[f'rot_{c}']) * math.pi / 180.0 for c in ['x', 'y', 'z']]
            f.write(struct.pack("<3f", *rot_rad))

            # 補間データ (24 bytes)
            interp_bytes = []
            for target in interp_targets:
                for s in suffixes:
                    col_name = f"{target}{s}"
                    val = row[col_name] if col_name in row and pd.notna(row[col_name]) else None
                    if val is None:
                        # デフォルト値 (直線)
                        default_val = 107 if ('x2' in s or 'y2' in s) else 20
                        interp_bytes.append(int(default_val))
                    else:
                        interp_bytes.append(int(val))
            f.write(bytes(interp_bytes))

            # 視野角 fov, パースフラグ
            f.write(struct.pack("<I", int(row['fov'])))
            f.write(struct.pack("<b", 0))

        # 7. 照明・セルフ影 (0固定)
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

    return str(vmd_path)

if __name__ == "__main__":
    # 単体テスト用
    try:
        res = run_vmd_export("ring_my_bell")
        print(f"ステップ４成功: {res}")
    except Exception as e:
        print(f"ステップ４失敗: {e}")