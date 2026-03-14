import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any
import argparse
from scipy.signal import resample

# プロジェクトのルートディレクトリを設定
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'  # 1つ上の階層のdataフォルダを指す

# ディレクトリパスの設定
PATHS = {
    'vmd_dir': DATA_ROOT / 'vmd',
    'camera_vmd_dir': DATA_ROOT / 'camera_vmd',
    'camera_csv_dir': DATA_ROOT / 'camera_csv',
    'camera_interpolated_dir': DATA_ROOT / 'camera_interpolated',
    'motion_vmd_dir': DATA_ROOT / 'motion_vmd',
    'motion_csv_dir': DATA_ROOT / 'motion_csv',
    'motion_wide_dir': DATA_ROOT / 'motion_wide',
    'label_csv_dir': DATA_ROOT / 'label_csv',
    'analysis_result_dir': PROJECT_ROOT / 'analysis_result'  # 出力ディレクトリはスクリプトと同じ階層に
}

def get_mmd_frame_count_from_wav(audio_path: str) -> int:
    y, sr = librosa.load(audio_path, sr=None)  # 元のサンプリングレートで読み込む
    duration_sec = len(y) / sr
    mmd_frames = int(duration_sec * 30)  # MMD は 30fps
    return mmd_frames

def resample_audio_features_to_camera_frames(features: dict, target_frames: int) -> dict:
    """
    音声特徴量をカメラフレーム数にリサンプリングする。

    Args:
        features: {特徴量名: numpy配列} の辞書
        target_frames: カメラ側のフレーム数（例：3000）

    Returns:
        リサンプリング後の特徴量辞書
    """
    resampled = {}

    for key, values in features.items():
        # values は 1次元の numpy array（音声フレーム数）
        # target_frames に合わせてリサンプリング
        resampled[key] = resample(values, target_frames)

    return resampled

def extract_audio_features(
    audio_path: str,
    sr: int = 22050,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = None,
    n_mels: int = 40,
    max_frames: Optional[int] = None
) -> Dict[str, Any]:
    """
    高品質な音声特徴量を抽出する改善版。
    カメラ制御生成タスクに最適化。
    """

    # --- 30fps に同期する hop_length を自動計算 ---
    # 30fps → 1フレーム = 1/30 秒
    # hop_length = sr / fps
    if hop_length is None:
        hop_length = int(sr / 30)

    # --- 音声読み込み ---
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading {audio_path}: {str(e)}")
        return None

    if y.ndim > 1:
        y = librosa.to_mono(y)

    features = {}

    # --- MFCC（20次元） ---
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length
    )
    for i in range(n_mfcc):
        features[f'mfcc_{i}'] = mfcc[i]

    # --- Melスペクトログラム（40次元） ---
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    for i in range(n_mels):
        features[f'mel_{i}'] = mel_db[i]

    # --- Chroma（12次元） ---
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    for i in range(12):
        features[f'chroma_{i}'] = chroma[i]

    # --- Spectral Contrast（7次元） ---
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    for i in range(contrast.shape[0]):
        features[f'contrast_{i}'] = contrast[i]

    # --- 単一値系特徴量 ---
    features.update({
        'spectral_centroid': librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0],
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0],
        'spectral_rolloff': librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
        )[0],
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(
            y, frame_length=n_fft, hop_length=hop_length
        )[0]
    })

    # --- NaN / inf を除去 ---
    for k, v in features.items():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        features[k] = v

    # --- max_frames に合わせてパディング ---
    if max_frames is not None:
        for key in features:
            values = features[key]
            if len(values) > max_frames:
                features[key] = values[:max_frames]
            elif len(values) < max_frames:
                pad_value = values[-1] if len(values) > 0 else 0.0
                padding = np.full(max_frames - len(values), pad_value)
                features[key] = np.concatenate([values, padding])

    return features

def process_audio_file(
    input_path: str,
    output_dir: str,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 735,
    n_mels: int = 128,
    max_frames: Optional[int] = None
) -> bool:
    """
    Process a single audio file and save features to CSV.
    
    Args:
        input_path: Path to input WAV file
        output_dir: Directory to save output CSV
        sr: Target sample rate
        n_mfcc: Number of MFCCs
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        max_frames: Maximum number of frames to extract
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract features
        features = extract_audio_features(
            input_path, sr, n_mfcc, n_fft, hop_length, n_mels, max_frames
        )
        
        if features is None:
            return False
        
        # リサンプリング
        # mmd_frames = get_mmd_frame_count_from_wav(input_path)
        # features = resample_audio_features_to_camera_frames(features, mmd_frames)

        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        # Add frame numbers
        df['frame'] = df.index
        
        # Get song ID from filename
        song_id = Path(input_path).stem
        df['song_id'] = song_id
        
        # Reorder columns
        cols = ['song_id', 'frame'] + [c for c in df.columns if c not in ['song_id', 'frame']]
        df = df[cols]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"{song_id}_audio_features.csv")
        df.to_csv(output_path, index=False)
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_directory(
    input_dir: str,
    output_dir: str,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 735,
    n_mels: int = 128,
    max_frames: Optional[int] = None
) -> None:
    """
    Process all WAV files in a directory.
    
    Args:
        input_dir: Directory containing WAV files
        output_dir: Directory to save output CSVs
        sr: Target sample rate
        n_mfcc: Number of MFCCs
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        max_frames: Maximum number of frames to extract
    """
    # Find all WAV files
    wav_files = list(Path(input_dir).glob('**/*.wav'))
    
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return
    
    print(f"Found {len(wav_files)} WAV files to process")
    
    # Process each file
    success_count = 0
    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        if process_audio_file(
            str(wav_file), output_dir, sr, n_mfcc, n_fft, hop_length, n_mels, max_frames
        ):
            success_count += 1
    
    print(f"\nSuccessfully processed {success_count}/{len(wav_files)} files")
    print(f"CSV files saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert WAV files to CSV with audio features')
    parser.add_argument('--input', type=str, default="data\\wav",
                        help='Input directory containing WAV files or path to a single WAV file')
    parser.add_argument('--output', type=str, default="data\\wav_csv",
                        help='Output directory to save CSV files')
    parser.add_argument('--camera_csv', type=str, default="data\\camera_csv",
                        help='Output directory to save CSV files')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Target sample rate (default: 22050)')
    parser.add_argument('--n_mfcc', type=int, default=13,
                        help='Number of MFCCs to extract (default: 13)')
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT window size (default: 2048)')
    parser.add_argument('--hop_length', type=int, default=735,
                        help='Hop length for STFT (default: 735)')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of mel bands (default: 128)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to extract (default: None)')
    
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single file
        if not args.input.lower().endswith('.wav'):
            print("Error: Input file must be a WAV file")
            return
        
        process_audio_file(
            args.input, args.output,
            sr=args.sr,
            n_mfcc=args.n_mfcc,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            max_frames=args.max_frames
        )
    else:
        # Process directory
        process_directory(
            args.input, args.output,
            sr=args.sr,
            n_mfcc=args.n_mfcc,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            max_frames=args.max_frames
        )

if __name__ == "__main__":
    main()