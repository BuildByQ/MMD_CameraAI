import torch
import pandas as pd
import numpy as np
import json
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from scipy.interpolate import interp1d
from pathlib import Path

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data' / 'normalization_params'
PRED01TO02_ROOT = PROJECT_ROOT.parent / 'predict_01to02'
PRED02_ROOT = PROJECT_ROOT.parent / 'predict_02'
MODEL_SAVE_PATH = PRED02_ROOT / 'model02_weights.pth'
CONFIG_PATH = MODEL_SAVE_PATH.parent / "model02_config.json"
INPUT_INS_PATH = PRED01TO02_ROOT / 'director_instruction.csv'
INPUT_MOT_PATH = DATA_ROOT / 'normalized_motion.csv'

class CameraLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # ターゲットの数に合わせて自動で変わるようにする
        self.fc = nn.Linear(hidden_size * 2, output_size) 

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_model(model_path, config):
    """
    保存された重みをロードし、推論モードに切り替えたモデルを返す。
    """
    print(f"モデルを構築し、重みをロードしています: {model_path}")

    # 1. Configからモデルの構造（次元）を取得
    input_size = len(config["x_cols"])
    output_size = len(config["y_cols"])
    
    # 2. モデルのインスタンス化（CameraLSTMクラスが定義済みである前提）
    # hidden_sizeなどは学習時のデフォルト値、あるいはconfigに含めていればそれを使う
    model = CameraLSTM(
        input_size=input_size, 
        output_size=output_size,
        hidden_size=128,  # 学習時と合わせる
        num_layers=2      # 学習時と合わせる
    )

    # 3. 重みファイルの存在確認とロード
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1. ファイル全体をロード（辞書として読み込まれる）
        checkpoint = torch.load(model_path, map_location=device)
        
        # 2. 辞書の中から「重みデータ」のキー（model_state_dict）だけを抽出して適用
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("   - 重みデータを正常に抽出しました。")
        else:
            # 万が一、重みデータだけが保存されている古い形式だった場合のフォールバック
            model.load_state_dict(checkpoint)
            print("   - 重みデータを直接ロードしました。")
            
        model.to(device)
        model.eval()
        return model, device

    except Exception as e:
        print(f"モデルのロードに失敗しました。詳細な理由は以下の通りです:")
        import traceback
        traceback.print_exc() # これでエラーの「詳細（どの層が合わないか）」が見えます
        sys.exit(1)

def load_config(config_path):
    """
    学習側で生成された model02_config.json を読み込み、
    推論に必要なパラメータ（列の順番、出力数など）を返す。
    """
    print(f"設定ファイルを読み込んでいます: {config_path}")
    
    path = Path(config_path)
    if not path.exists():
        print(f"エラー: 設定ファイルが見つかりません。学習を先に完了させてください。")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 推論に必須のキーが存在するかチェック
        required_keys = ["x_cols", "y_cols", "seq_len"]
        for key in required_keys:
            if key not in config:
                print(f"エラー: config 内に必須キー '{key}' が不足しています。")
                sys.exit(1)
        
        print(f"   - 入力特徴量 (X): {len(config['x_cols'])} 列")
        print(f"   - 出力ターゲット (Y): {len(config['y_cols'])} 列")
        print(f"   - シーケンス長: {config['seq_len']}")
        
        return config

    except Exception as e:
        print(f"エラー: JSONの読み込み中に問題が発生しました: {e}")
        sys.exit(1)

def load_and_prepare_data(ins_path, mot_path, config):
    print(f"データをロード中...")

    df_ins = pd.read_csv(ins_path)
    df_mot = pd.read_csv(mot_path)

    # --- 【重要】結合キーに song_id を追加 ---
    # これにより、同じフレーム番号でも曲が違えば別物として扱われます
    df = pd.merge(df_ins, df_mot, on=['song_id', 'frame'], how='inner')
    
    # 結合後に「曲ごと」→「フレーム順」にソートを確実に
    df = df.sort_values(['song_id', 'frame']).reset_index(drop=True)

    # 接頭辞の正規化
    new_columns = {col: col.replace('bin_bin_', 'bin_') for col in df.columns if col.startswith('bin_bin_')}
    df = df.rename(columns=new_columns)

    if 'bin_event_cut' not in df.columns and 'event_cut' in df.columns:
        df = df.rename(columns={'event_cut': 'bin_event_cut'})
    
    # バリデーション
    missing_cols = [c for c in config["x_cols"] if c not in df.columns]
    if missing_cols:
        print(f"エラー: 列が不足しています: {missing_cols}")
        sys.exit(1)

    return df

def inference(model, df, config, device):
    """
    スライディングウィンドウを用いて、全フレームのカメラ座標を予測する。
    """
    x_cols = config["x_cols"]
    seq_len = config["seq_len"]
    
    # 1. 学習時と同じ列、順番でデータを抽出し、numpy行列に変換
    X_all = df[x_cols].values.astype(np.float32)
    num_frames = len(X_all)
    
    print(f"推論を開始します (全 {num_frames} フレーム / Window: {seq_len})")
    
    all_preds = []
    
    # 2. 推論ループ
    with torch.no_grad():
        for i in range(num_frames):
            # --- スライディングウィンドウの作成 ---
            if i < seq_len - 1:
                # 窓が足りない最初期フレームの処理（パディング）
                # 0フレーム目のデータをコピーして、seq_len分まで埋める
                window = np.tile(X_all[0], (seq_len, 1))
                # 実際のデータを後ろ側に流し込む
                window[-(i+1):] = X_all[:i+1]
            else:
                # 通常時の窓切り出し（直近 seq_len フレーム）
                window = X_all[i - (seq_len - 1) : i + 1]
            
            # --- テンソル変換とデバイス転送 ---
            # [seq_len, feature] -> [batch(1), seq_len, feature]
            input_tensor = torch.tensor(window).unsqueeze(0).to(device)
            
            # --- AIによる予測 ---
            pred = model(input_tensor) # 出力は [1, 14]
            
            # 結果をCPUに戻してリストへ格納
            all_preds.append(pred.squeeze().cpu().numpy())
            
            # 進捗表示（1000フレームごとなど）
            if (i + 1) % 1000 == 0:
                print(f"   - {i + 1} フレーム完了...")

    return np.array(all_preds)

def main():
    print("第2モデル 推論パイプライン開始")

    config = load_config(CONFIG_PATH)
    model, device = load_model(MODEL_SAVE_PATH, config)

    # 1. データのロード（この時点で正しい曲×フレームの組み合わせになる）
    df_merged = load_and_prepare_data(INPUT_INS_PATH, INPUT_MOT_PATH, config)

    # 2. 【重要】曲ごとにループして推論・保存
    for sid, df_song in df_merged.groupby("song_id", sort=False):
        print(f"\n曲: {sid} のカメラ座標を生成中... ({len(df_song)} frames)")
        
        # 曲ごとに独立して推论（ウィンドウが曲を跨がない）
        predictions = inference(model, df_song, config, device)

        # 3. 結果の保存
        # 曲名を含んだ出力パスを作成
        output_filename = f"prediction_{sid}.csv"
        output_path = PRED02_ROOT / output_filename
        
        # 保存用のDataFrame作成
        df_out = pd.DataFrame(predictions, columns=config["y_cols"])
        df_out.insert(0, 'frame', df_song['frame'].values)
        df_out.insert(0, 'song_id', sid)
        
        df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"保存完了: {output_path}")

    print(f"\n全曲のデバッグ実行完了")

if __name__ == "__main__":
    main()