import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from utils.color import lab_to_bgr_cv2
from skimage.color import rgb2lab

class VideoColorizer:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(n_channels=1, n_classes=2).to(self.device)
        self.model.eval()
        
        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            # Support both raw state_dicts and checkpoint dictionaries
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {model_path}")
        else:
            print("Warning: No weights loaded.")

    def colorize_frame(self, frame_bgr, sat_factor=1.0, tint_shift=0):
        h, w = frame_bgr.shape[:2]
        L_highres = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        
        # Prep for network (256x256)
        small_bgr = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        L_small = rgb2lab(small_rgb).astype("float32")[:, :, 0] / 50. - 1.
        L_tensor = torch.from_numpy(L_small).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds_ab = self.model(L_tensor).squeeze(0).cpu().numpy().transpose((1, 2, 0))
        
        # Resize ab to high-res and convert to BGR
        preds_ab_highres = cv2.resize(preds_ab, (w, h), interpolation=cv2.INTER_CUBIC)
        return lab_to_bgr_cv2(L_highres, preds_ab_highres, sat_factor, tint_shift)

    def process_video(self, input_path, output_path, max_height=480, progress_callback=None, sat_factor=1.0, tint_shift=0):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scale = max_height / orig_h if max_height and orig_h > max_height else 1.0
        w, h = int(orig_w * scale), int(orig_h * scale)
            
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            if scale != 1.0: frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            
            colorized_frame = self.colorize_frame(frame, sat_factor, tint_shift)
            out.write(colorized_frame)
            
            if progress_callback: 
                progress_callback(i + 1, total_frames, frame=colorized_frame)
            
        cap.release(); out.release()
        
        # ── Final Processing: Audio Merge & Web Optimization ────────────────
        try:
            # Support both MoviePy v1.x and v2.x
            try:
                from moviepy.video.io.VideoFileClip import VideoFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip
                
            orig = VideoFileClip(input_path)
            colorized = VideoFileClip(output_path)
            
            # Use the original audio if it exists, otherwise just re-encode for web
            if orig.audio:
                if hasattr(colorized, "with_audio"):
                    final_video = colorized.with_audio(orig.audio)
                else:
                    final_video = colorized.set_audio(orig.audio)
            else:
                final_video = colorized

            temp_final = output_path.replace(".mp4", "_final.mp4")
            # Write with libx264 codec (standard for web)
            final_video.write_videofile(temp_final, codec="libx264", audio_codec="aac" if orig.audio else None)
            
            orig.close(); colorized.close(); final_video.close()
            os.replace(temp_final, output_path)
            print("Video finalized for web playback.")
            
        except Exception as e:
            print(f"Final optimization failed: {e}. Output might not play in browser.")
        
        return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="")
    args = parser.parse_args()
    VideoColorizer(args.weights).process_video(args.input, args.output)
