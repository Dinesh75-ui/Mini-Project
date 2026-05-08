# Deep Learning-Based Video Colorization System

A PyTorch project that automatically adds realistic color to grayscale videos using an Attention-based U-Net architecture trained on the Tiny ImageNet dataset.

## Features
- **Attention U-Net Architecture**: Advanced encoder-decoder with attention gates and skip connections for superior color restoration.
- **LAB Color Space**: The model receives the L (lightness) channel as input and predicts the a/b (color) channels, preserving the original video's brightness and detail.
- **Video Pipeline**: High-performance frame-by-frame processing with automatic audio merging.
- **Interactive UI**: Streamlit web app for easy video upload, real-time colorization, and download.

## Project Structure
```
Mini-Project/
├── models/        ← Model architecture (unet.py)
├── training/      ← Training loops and loss functions
├── inference/     ← Video processing pipeline
├── utils/         ← LAB conversion and preprocessing utilities
├── app/           ← Streamlit web application
├── outputs/       ← Saved model weights (checkpoints)
└── data/          ← Dataset storage
```

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dinesh75-ui/Mini-Project.git
   cd Mini-Project
   ```

2. **Install Dependencies**:
   It is recommended to use a GPU-enabled version of PyTorch for training.
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

## Dataset Preparation

This project uses the **Tiny ImageNet** dataset (200 classes, 100,000 images). To download and prepare the dataset automatically, run:

```bash
python data/download_tiny_imagenet.py
```
*The script will download (~250MB) and extract the dataset into the `data/` folder.*

## Training the Model

To train the model from scratch on Tiny ImageNet:

```bash
python training/train.py --epochs 20 --batch_size 32
```

To resume training from a specific checkpoint:
```bash
python training/train.py --resume outputs/weights/unet_colorization_best.pth --epochs 20
```

## Running the Application

### 1. Web Interface (Recommended)
Launch the interactive Streamlit dashboard:
```bash
streamlit run app/streamlit_app.py
```

### 2. Command Line Inference
Process a video file directly via terminal:
```bash
python inference/video_pipeline.py --input "path/to/video.mp4" --output "outputs/colorized.mp4" --weights "outputs/weights/unet_colorization_best.pth"
```

## Performance Tips
- Ensure your GPU is being used (the console should log `Using device: cuda`).
- Use a larger batch size (e.g., 32 or 64) if your GPU VRAM allows, to speed up training.
