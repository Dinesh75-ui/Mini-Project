import torch
import numpy as np
import cv2

def lab_to_rgb_tensor(L, ab):
    """
    Differentiable LAB to RGB conversion for Perceptual Loss.
    L is in [-1, 1], ab is in [-1, 1].
    """
    # 1. Denormalize to standard LAB ranges
    L_val = (L + 1.0) * 50.0
    a_val = ab[:, [0], ...] * 110.0
    b_val = ab[:, [1], ...] * 110.0
    
    # 2. LAB to XYZ (linearized approximation)
    y = (L_val + 16.0) / 116.0
    x = a_val / 500.0 + y
    z = y - b_val / 200.0
    
    def f_inv(t):
        return torch.where(t > 0.20689, t**3, (t - 16.0/116.0) / 7.787)
    
    x = f_inv(x) * 0.95047
    y = f_inv(y) * 1.0
    z = f_inv(z) * 1.08883
    
    # 3. XYZ to RGB (Linear)
    r =  3.2406 * x - 1.5372 * y - 0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b =  0.0557 * x - 0.2040 * y + 1.0570 * z
    
    rgb = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb, 0, 1)

def lab_to_bgr_cv2(L_highres, ab_preds, sat_factor=1.0, tint_shift=0):
    """
    High-quality LAB to BGR conversion using OpenCV for final inference.
    """
    # 1. Apply color controls to predicted ab (range [-1, 1])
    ab_preds = ab_preds * sat_factor
    ab_preds[:, :, 0] -= (tint_shift / 110.0) # Shift 'a' channel
    
    # 2. Map back to OpenCV LAB ranges (L: 0-255, ab: 0-255)
    # OpenCV ab = (true_ab) + 128
    ab_cv2 = np.clip(ab_preds * 110. + 128., 0, 255).astype(np.uint8)
    
    # 3. Combine with high-res L
    h, w = L_highres.shape[:2]
    lab_cv2 = np.empty((h, w, 3), dtype=np.uint8)
    lab_cv2[:, :, 0] = L_highres
    lab_cv2[:, :, 1:] = ab_cv2
    
    # 4. Convert back to BGR
    return cv2.cvtColor(lab_cv2, cv2.COLOR_LAB2BGR)
