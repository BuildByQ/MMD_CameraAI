import os
import glob
import csv
from collections import Counter, defaultdict
from model_vmd_to_csv import model_vmd_to_csv  # 先ほど作成した変換関数を利用

def convert_all_vmd_to_csv(input_dir, output_dir):
    """指定ディレクトリ内のvmdをすべてcsv化"""
    os.makedirs(output_dir, exist_ok=True)
    csv_files = []
    for vmd_file in glob.glob(os.path.join(input_dir, "*.vmd")):
        base = os.path.splitext(os.path.basename(vmd_file))[0]
        csv_file = os.path.join(output_dir, base + "_motion.csv")
        print(f"Converting {vmd_file} → {csv_file}")
        model_vmd_to_csv(vmd_file, csv_file)
        csv_files.append(csv_file)
    return csv_files

def aggregate_bone_occurrence(csv_files, output_csv):
    """
    全csvを集計し、ボーンごとの出現率を出力
    出力列: bone_name, count, file_count, ratio
    """
    bone_counter = Counter()
    bone_file_presence = defaultdict(set)

    for csv_file in csv_files:
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bone = row["bone_name"]
                bone_counter[bone] += 1
                bone_file_presence[bone].add(csv_file)

    total_files = len(csv_files)
    rows = []
    for bone, count in bone_counter.items():
        file_count = len(bone_file_presence[bone])
        ratio = file_count / total_files if total_files > 0 else 0
        rows.append([bone, count, file_count, f"{ratio:.2f}"])

    # 出力
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bone_name", "total_count", "file_count", "ratio"])
        writer.writerows(sorted(rows, key=lambda r: (-r[2], r[0])))

    print(f"集計結果を {output_csv} に保存しました。")

if __name__ == "__main__":
    input_dir = "data/motion_vmd"       # vmdファイル置き場
    output_dir = "data/motion_csv"      # csv出力先
    output_csv = "results/bone_occurrence.csv"

    csv_files = convert_all_vmd_to_csv(input_dir, output_dir)
    aggregate_bone_occurrence(csv_files, output_csv)