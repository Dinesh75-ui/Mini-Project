# 🎨 Deep Learning-Based Video Colorization System

A GAN-powered system that automatically restores realistic color to black-and-white videos, with a focus on **Telugu cinema restoration**.  
Built with PyTorch, an Attention U-Net generator, a PatchGAN discriminator, and a Streamlit web interface.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **GAN Architecture** | Attention U-Net generator + PatchGAN discriminator |
| **LAB Color Space** | Model predicts `ab` channels from `L` (lightness) — preserves original brightness |
| **Perceptual Loss** | VGG-based perceptual loss + L1 + adversarial loss for vivid, sharp colors |
| **AMP Training** | Automatic Mixed Precision (FP16) for fast GPU training |
| **Train/Val Split** | Automatic 90/10 split with augmentations disabled on validation |
| **Metric Logging** | Per-epoch JSON log → beautiful matplotlib dashboard |
| **Streamlit UI** | Real-time colorization with live frame preview and download |
| **Video Pipeline** | Frame-by-frame inference with automatic audio re-merging |

---

## 🗂️ Project Structure

```
Mini-Project/
├── app/
│   └── streamlit_app.py        ← Interactive web UI for colorizing videos
├── data/
│   ├── download_coco_subset.py ← Download COCO training images
│   └── optimize_dataset.py     ← Pre-resize images to 128px for fast loading
├── inference/
│   └── video_pipeline.py       ← Frame extraction, colorization, audio merge
├── models/
│   ├── unet.py                 ← Attention U-Net generator
│   └── discriminator.py        ← PatchGAN discriminator
├── training/
│   ├── train.py                ← Main GAN training loop (logs metrics to JSON)
│   ├── loss.py                 ← ColorizationLoss (L1 + VGG) + GANLoss
│   └── plot_metrics.py         ← Developer tool: visualise training graphs
├── utils/
│   ├── color.py                ← LAB ↔ RGB tensor conversions
│   ├── preprocessing.py        ← ColorizationDataset + augmentations
│   └── optimize_dataset.py     ← Dataset optimization utilities
├── Telugu_Branch/
│   └── data/
│       └── download_telugu_dataset.py ← Telugu cinema frame extractor
├── outputs/
│   ├── weights/                ← Saved model checkpoints (.pth)
│   └── metrics.json            ← Auto-generated training metrics log
├── requirements.txt
└── commands.txt                ← Quick-reference commands
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Dinesh75-ui/Mini-Project.git
cd Mini-Project
```

### 2. Install PyTorch (GPU recommended)
```bash
# For CUDA 12.4 (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For CPU only
pip install torch torchvision
```

### 3. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

### Option A — COCO Subset (Recommended, ~100k images)
```bash
python data/download_coco_subset.py
```
Then optimize (pre-resize to 128px for 3× faster training):
```bash
python utils/optimize_dataset.py
```

### Option B — Telugu Cinema Frames
Extract frames from your own Telugu movie files:
```bash
python Telugu_Branch/data/download_telugu_dataset.py
```

---

## 🚀 Training the Model

### Train from Scratch
```bash
python training/train.py --epochs 20 --batch 16
```

### Resume from a Checkpoint
```bash
python training/train.py --resume outputs/weights/color_model_epoch_45.pth --epochs 60
```

### Train on a Custom Dataset
```bash
python training/train.py --data path/to/your/images --epochs 30 --batch 8
```

**Training automatically:**
- Splits data 90% train / 10% validation
- Runs a validation loop every epoch and logs Val L1 Loss
- Saves `best_model.pth` whenever validation loss improves
- Saves a full checkpoint every 5 epochs
- Writes all metrics to `outputs/metrics.json`

---

## 📊 Visualising Training Metrics

After (or during) training, generate a full metrics dashboard:

```bash
python training/plot_metrics.py
```

This opens an interactive window with **6 graphs**:

| Graph | What it shows |
|---|---|
| Generator & Discriminator Loss | Per-epoch average train loss for G and D |
| Validation L1 Loss | How well the model generalises; marks the best epoch |
| Generator Loss Components | L1 (pixel) vs Perceptual (VGG) breakdown |
| Overfitting Monitor | Train G vs Val L1 with gap curve |
| Learning Rate Schedule | Cosine annealing curve for G and D |
| Epoch Duration | Time per epoch in seconds |

### Save as PNG instead of opening a window
```bash
python training/plot_metrics.py --save
# → outputs/training_metrics.png
```

### Use a custom metrics file
```bash
python training/plot_metrics.py --metrics path/to/metrics.json
```

---

## 🖥️ Running the Web Application

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

**What you can do:**
1. Upload any B&W video (MP4 / AVI / MOV)
2. Pick output resolution (480p recommended for speed, 720p, or original)
3. Adjust **Saturation** and **Tint** in the sidebar for fine-tuning
4. Watch frame-by-frame live preview during colorization
5. Download the finished colorized video

---

## 🎬 Command-Line Inference

Colorize a single video without the UI:
```bash
python inference/video_pipeline.py \
  --input  "path/to/bw_video.mp4" \
  --output "outputs/colorized.mp4" \
  --weights "outputs/weights/best_model.pth"
```

---

## 🔧 Performance Tips

- **GPU check**: The console should print `Using device: cuda | AMP: ON` at the start of training. If it shows `cpu`, verify your PyTorch CUDA installation.
- **Batch size**: Use `--batch 16` for 6 GB VRAM, `--batch 32` for 8–12 GB VRAM.
- **Gradient accumulation**: Already built-in via AMP; reduces memory pressure automatically.
- **Resume training**: Always use `--resume` to continue from the last checkpoint and avoid restarting from scratch.
- **Best model**: Use `outputs/weights/best_model.pth` for inference — this is the checkpoint with the lowest validation L1 loss.

---

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-image>=0.21.0
streamlit>=1.26.0
tqdm>=4.65.0
matplotlib>=3.7.0
moviepy>=1.0.3
```

---

## 📄 License

This project is for academic and research purposes.
