import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import librosa
from tqdm import tqdm

class CameraAIDataLoader:
    def __init__(self, data_root: Union[str, Path], sr: int = 22050, hop_length: int = 512):
        """
        カメラAIデータローダーの初期化
        
        Args:
            data_root: データのルートディレクトリ
            sr: 音声のサンプリングレート
            hop_length: MFCCのホップ長
        """
        self.data_root = Path(data_root)
        self.sr = sr
        self.hop_length = hop_length
        self.audio_dir = self.data_root / 'audio'
        self.camera_dir = self.data_root / 'camera_interpolated'
        self.motion_dir = self.data_root / 'motion_wide'
        self.label_dir = self.data_root / 'label_csv'
        
        # ディレクトリの存在確認
        self._check_dirs()
        
        # データセットのメタデータをロード
        self.metadata = self._load_metadata()
    
    def _check_dirs(self) -> None:
        """必要なディレクトリが存在するか確認"""
        required_dirs = [self.audio_dir, self.camera_dir, self.motion_dir, self.label_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"ディレクトリが見つかりません: {dir_path}")
    
    def _load_metadata(self) -> List[Dict]:
        """データセットのメタデータをロード"""
        metadata = []
        
        # ラベルファイルを基準にメタデータを構築
        for label_file in self.label_dir.glob('*.csv'):
            base_name = label_file.stem.replace('_labels', '')
            audio_file = self.audio_dir / f"{base_name}.wav"
            camera_file = self.camera_dir / f"{base_name}.csv"
            motion_file = self.motion_dir / f"{base_name}.csv"
            
            if audio_file.exists() and camera_file.exists() and motion_file.exists():
                metadata.append({
                    'base_name': base_name,
                    'audio_path': audio_file,
                    'camera_path': camera_file,
                    'motion_path': motion_file,
                    'label_path': label_file
                })
        
        if not metadata:
            raise FileNotFoundError("有効なデータが見つかりませんでした")
            
        return metadata
    
    def load_audio_features(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        音声ファイルからMFCC特徴量を抽出
        
        Args:
            audio_path: 音声ファイルのパス
            
        Returns:
            np.ndarray: MFCC特徴量 (n_mfcc, time_steps)
        """
        # 音声をロード
        y, _ = librosa.load(audio_path, sr=self.sr)
        
        # MFCCを抽出
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=40,  # MFCCの次元数
            n_fft=2048,
            hop_length=self.hop_length
        )
        
        return mfcc.T  # (time_steps, n_mfcc) に転置
    
    def load_camera_data(self, camera_path: Union[str, Path]) -> pd.DataFrame:
        """
        カメラデータをロード
        
        Args:
            camera_path: カメラCSVファイルのパス
            
        Returns:
            pd.DataFrame: カメラデータ
        """
        return pd.read_csv(camera_path)
    
    def load_motion_data(self, motion_path: Union[str, Path]) -> pd.DataFrame:
        """
        モーションデータをロード
        
        Args:
            motion_path: モーションCSVファイルのパス
            
        Returns:
            pd.DataFrame: モーションデータ
        """
        return pd.read_csv(motion_path)
    
    def load_label_data(self, label_path: Union[str, Path]) -> pd.DataFrame:
        """
        ラベルデータをロード
        
        Args:
            label_path: ラベルCSVファイルのパス
            
        Returns:
            pd.DataFrame: ラベルデータ
        """
        return pd.read_csv(label_path)
    
    def get_sample(self, idx: int) -> Dict:
        """
        指定したインデックスのサンプルを取得
        
        Args:
            idx: サンプルのインデックス
            
        Returns:
            Dict: サンプルデータ
        """
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError(f"インデックスが範囲外です: {idx}")
        
        meta = self.metadata[idx]
        
        # 各データをロード
        mfcc = self.load_audio_features(meta['audio_path'])
        camera_data = self.load_camera_data(meta['camera_path'])
        motion_data = self.load_motion_data(meta['motion_path'])
        labels = self.load_label_data(meta['label_path'])
        
        # フレーム数を最小のものに合わせる
        min_frames = min(len(mfcc), len(camera_data), len(motion_data), len(labels))
        mfcc = mfcc[:min_frames]
        camera_data = camera_data.iloc[:min_frames].reset_index(drop=True)
        motion_data = motion_data.iloc[:min_frames].reset_index(drop=True)
        labels = labels.iloc[:min_frames].reset_index(drop=True)
        
        return {
            'base_name': meta['base_name'],
            'audio': mfcc,
            'camera': camera_data,
            'motion': motion_data,
            'labels': labels
        }
    
    def get_all_samples(self) -> List[Dict]:
        """
        すべてのサンプルを取得
        
        Returns:
            List[Dict]: すべてのサンプルデータ
        """
        return [self.get_sample(i) for i in range(len(self.metadata))]
    
    def get_data_stats(self) -> Dict:
        """
        データセットの統計情報を取得
        
        Returns:
            Dict: 統計情報
        """
        stats = {
            'num_samples': len(self.metadata),
            'audio_features': None,
            'camera_features': None,
            'motion_features': None,
            'label_distribution': None
        }
        
        # 最初のサンプルをロードして特徴量の次元を取得
        if self.metadata:
            sample = self.get_sample(0)
            stats.update({
                'audio_features': {
                    'shape': sample['audio'].shape,
                    'dtype': str(sample['audio'].dtype)
                },
                'camera_features': {
                    'columns': sample['camera'].columns.tolist(),
                    'shape': sample['camera'].shape
                },
                'motion_features': {
                    'columns': sample['motion'].columns.tolist(),
                    'shape': sample['motion'].shape
                },
                'label_distribution': sample['labels'].mean().to_dict()
            })
        
        return stats