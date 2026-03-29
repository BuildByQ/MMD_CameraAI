import json
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
import os
import torch
import numpy as np
import json
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from pathlib import Path

# プロジェクトのルートディレクトリを設定
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / 'data'
ML_ROOT = PROJECT_ROOT.parent / 'ml'
LABEL_ROOT = PROJECT_ROOT.parent / 'predict_01'

# 1. 設定読み込み
def load_config(config_path="config.json"):
    """
    設定ファイル（config.json）を読み込んで辞書として返す関数。
    """
    config_path = ML_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"設定を読み込みました: {config_path}")
    return config

def get_dynamic_columns(config: dict, df_columns: list):
    """JSON設定から実際のCSVヘッダー名を組み立てる"""
    
    # 特徴量: ボーン名 -> _w_x, _w_y, _w_z
    motion_bones = config.get('motion_features', [])
    motion_feats = [f"{b}{s}" for b in motion_bones for s in ['_w_x', '_w_y', '_w_z']]
    audio_feats = config.get('audio_features', [])
    feature_cols = [f for f in (motion_feats + audio_feats) if f in df_columns]

    # ラベル: base + anchor_definitionsからの自動生成
    base_labels = config.get('labels_features', [])
    anchor_defs = config.get('anchor_definitions', {})
    dynamic_labels = []
    for entry in anchor_defs.values():
        for b in entry.get("bones", []):
            dynamic_labels.extend([f"bin_target_{b}_front", f"bin_target_{b}_back", 
                                   f"bin_target_{b}_focused_strict", f"bin_target_{b}_focused", 
                                   f"bin_target_{b}_focused_loose"])
    
    # 順序を維持して重複排除
    all_label_list = base_labels + dynamic_labels
    seen = set()
    label_cols = [l for l in all_label_list if l in df_columns and not (l in seen or seen.add(l))]

    return feature_cols, label_cols

class CameraLabelGRUModel(nn.Module):
    def __init__(
        self,
        input_dim,          # 統計量込みの特徴量次元
        output_dim,         # ラベル数
        num_songs,          # 曲数（embedding 用）
        embed_dim=4,        # 小さくして過学習を防ぐ
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        bidirectional=False
    ):
        super().__init__()

        # song_id embedding
        self.song_embedding = nn.Embedding(num_songs, embed_dim)
        nn.init.xavier_uniform_(self.song_embedding.weight)  # 安定化のための初期化

        self.embed_dim = embed_dim

        # embedding dropout（ID丸暗記を防ぐ）
        self.embed_dropout = nn.Dropout(0.3)

        # GRU の入力次元は「特徴量 + embedding」
        self.gru = nn.GRU(
            input_size=input_dim + embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_size * (2 if bidirectional else 1)

        # 最終 hidden → ラベル
        self.fc = nn.Linear(gru_out_dim, output_dim)

    def forward(self, x, song_idx):
        """
        x: (batch, seq_len, input_dim)
        song_idx: (batch,)
        """

        # song embedding を取得
        song_emb = self.song_embedding(song_idx)  # (batch, embed_dim)

        # embedding を L2 正規化（暴走防止）
        song_emb = F.normalize(song_emb, dim=1)

        # dropout（ID丸暗記を防ぐ）
        song_emb = self.embed_dropout(song_emb)

        # seq_len にブロードキャスト
        song_emb = song_emb.unsqueeze(1).repeat(1, x.size(1), 1)

        # 特徴量と embedding を concat
        x = torch.cat([x, song_emb], dim=2)

        # GRU
        out, h = self.gru(x)

        # 最後の hidden を使用
        last_hidden = h[-1]  # (batch, hidden)

        logits = self.fc(last_hidden)
        return logits

def build_model(config: Dict[str, Any], input_dim: int, output_dim: int, num_songs: int) -> nn.Module:
    """
    config の設定に基づいてモデルのインスタンスを生成する。
    """
    model_cfg = config.get("model", {})

    model = CameraLabelGRUModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_songs=num_songs,
        embed_dim=model_cfg.get("embed_dim", 4),
        hidden_size=model_cfg.get("hidden_size", 256),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.2),
        bidirectional=model_cfg.get("bidirectional", False),
    )

    print(f"Model Context: Input={input_dim}, Output={output_dim}, Songs={num_songs}")
    return model