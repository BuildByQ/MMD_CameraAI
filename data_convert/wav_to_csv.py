import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any
from scipy.signal import resample

# --- パス設定 ---
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'
INPUT_DIR = DATA_ROOT / 'wav'
OUTPUT_DIR = DATA_ROOT / 'wav_csv'

def extract_audio_features(
    audio_path: str,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    n_mels: int = 128,
    max_frames: Optional[int] = None
) -> Dict[str, Any]:
    """
    音声からML用の特徴量（MFCC, Mel, Chroma等）を抽出する
    """
    if hop_length is None:
        hop_length = int(sr / 30)  # 30fps同期用

    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Load Error {audio_path}: {e}")
        return None

    if y.ndim > 1:
        y = librosa.to_mono(y)

    features = {}

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    for i in range(n_mfcc):
        features[f'mfcc_{i}'] = mfcc[i]

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    for i in range(n_mels):
        features[f'mel_{i}'] = mel_db[i]

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    for i in range(12):
        features[f'chroma_{i}'] = chroma[i]

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    for i in range(contrast.shape[0]):
        features[f'contrast_{i}'] = contrast[i]

    # 単一値系
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]

    # NaN / inf 処理
    for k, v in features.items():
        features[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    # フレーム数調整 (max_frames指定時)
    if max_frames is not None:
        for k in features:
            v = features[k]
            if len(v) > max_frames:
                features[k] = v[:max_frames]
            elif len(v) < max_frames:
                features[k] = np.pad(v, (0, max_frames - len(v)), mode='edge')

    return features

def process_audio_file(input_path: Path, output_dir: Path) -> bool:
    """1つの音声ファイルを変換。既存ならスキップ。"""
    song_id = input_path.stem
    output_path = output_dir / f"{song_id}_audio_features.csv"

    # --- 既存チェック ---
    if output_path.exists():
        return True # スキップ成功扱い

    features = extract_audio_features(str(input_path))
    if features is None:
        return False

    df = pd.DataFrame(features)
    df['frame'] = df.index
    df['song_id'] = song_id
    
    # 列順整理
    cols = ['song_id', 'frame'] + [c for c in df.columns if c not in ['song_id', 'frame']]
    df[cols].to_csv(output_path, index=False)
    return True

def main():
    """ディレクトリ内のWAVを一括処理"""
    print(f"[Step 6] WAV -> Audio Features CSV 変換チェック開始")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    wav_files = list(INPUT_DIR.glob('*.wav'))
    if not wav_files:
        print(f"WAVファイルが見つかりません: {INPUT_DIR}")
        return

    processed = 0
    skipped = 0
    
    for wav_file in tqdm(wav_files, desc="Audio Extraction"):
        # 既存チェックを内部で行う
        if (OUTPUT_DIR / f"{wav_file.stem}_audio_features.csv").exists():
            skipped += 1
            continue
            
        if process_audio_file(wav_file, OUTPUT_DIR):
            processed += 1
            
    print(f"\n処理結果: 新規抽出 {processed} 件 / スキップ {skipped} 件")

if __name__ == "__main__":
    main()