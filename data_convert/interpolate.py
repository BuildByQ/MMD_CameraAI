import numpy as np
from scipy.special import comb
def bernstein_poly(i, n, t):
    """ベジェ曲線のバーンスタイン多項式を計算"""
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
def bezier_interpolate(p0, p1, p2, p3, t):
    """3次ベジェ曲線で補間"""
    t = np.clip(t, 0, 1)  # tを0-1の範囲に制限
    return (p0 * (1-t)**3 + 
            3 * p1 * t * (1-t)**2 + 
            3 * p2 * t**2 * (1-t) + 
            p3 * t**3)
def interpolate_angle(start, end, t):
    """
    角度を補間（360度の正規化なし）
    MMDの回転補間に使用
    """
    return start + (end - start) * t
def get_bezier_t(x1, y1, x2, y2, target_x, max_iter=20, tolerance=1e-6):
    """
    ベジェ曲線のX座標からtの値を求める（二分法）
    """
    # ベジェ曲線のX座標を計算する関数
    def bezier_x(t):
        return 3 * x1 * t * (1-t)**2 + 3 * x2 * t**2 * (1-t) + t**3
    
    # 二分法でtを求める
    low = 0.0
    high = 1.0
    
    for _ in range(max_iter):
        t = (low + high) / 2
        x = bezier_x(t)
        
        if abs(x - target_x) < tolerance:
            return t
        
        if x < target_x:
            low = t
        else:
            high = t
    
    return t
def interpolate_bezier_frame(frame_start, frame_end, t, param_groups):
    """
    1フレーム分のベジェ曲線補間を実行
    """
    new_frame = {'frame': frame_start['frame'] + (frame_end['frame'] - frame_start['frame']) * t}
    
    for param, interp_params in param_groups.items():
        if not interp_params or param not in frame_start or param not in frame_end:
            continue
        
        # 開始値と終了値
        start_val = frame_start[param]
        end_val = frame_end[param]
        
        # 補間パラメータを取得（後方のキーフレームから取得）
        x1 = frame_end[interp_params[0]] / 127.0
        y1 = frame_end[interp_params[1]] / 127.0
        x2 = frame_end[interp_params[2]] / 127.0
        y2 = frame_end[interp_params[3]] / 127.0
        
        # ベジェ曲線のt値を計算
        t_bezier = get_bezier_t(x1, y1, x2, y2, t)
        
        # ベジェ曲線のY座標を計算
        y = 3 * y1 * t_bezier * (1-t_bezier)**2 + 3 * y2 * t_bezier**2 * (1-t_bezier) + t_bezier**3
        
        # 補間値を計算
        new_frame[param] = start_val + (end_val - start_val) * y
    
    # 補間されたフレームのinterpパラメータを0に設定
    for col in frame_start.keys():
        if any(col.startswith(prefix) for prefix in ['X_', 'Y_', 'Z_', 'R_', 'L_', 'V_']):
            new_frame[col] = 0
    
    return new_frame