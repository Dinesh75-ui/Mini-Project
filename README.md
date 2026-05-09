# Deep Learning-Based Video Colorization System

A PyTorch project for restoring color to black-and-white videos. It uses an Attention U-Net generator, a PatchGAN discriminator for training, and a Streamlit interface for video inference.

## Features

| Feature | Details |
|---|---|
| GAN training | Attention U-Net generator with PatchGAN discriminator |
| LAB color space | Predicts `ab` channels from the grayscale `L` channel |
| Perceptual loss | Combines Smooth L1, VGG perceptual loss, and adversarial loss |
| AMP training | Uses mixed precision automatically when CUDA is available |
| Streamlit UI | Upload a video, preview progress, and download the result |
| Video pipeline | Frame-by-frame inference with optional audio re-merge |

## Project Structure

```text
Mini-Project/
├── app/
│   └── streamlit_app.py
├── data/
│   └── download_coco_subset.py
├── inference/
│   └── video_pipeline.py
├── models/
│   ├── discriminator.py
│   └── unet.py
├── training/
│   ├── loss.py
│   ├── plot_metrics.py
│   └── train.py
├── utils/
│   ├── color.py
│   ├── optimize_dataset.py
│   └── preprocessing.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated datasets, checkpoints, videos, metrics, and plots are ignored by Git so the repository stays lightweight.

## Setup

```bash
git clone https://github.com/Dinesh75-ui/Mini-Project.git
cd Mini-Project
pip install -r requirements.txt
```

For NVIDIA GPUs, install the PyTorch build that matches your CUDA version before installing the remaining requirements. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Dataset Preparation

Download a COCO image subset:

```bash
python data/download_coco_subset.py
```

Pre-resize images for faster training:

```bash
python utils/optimize_dataset.py --src data/coco_subset --dst data/coco_128_subset
```

## Training

Train from scratch:

```bash
python training/train.py --epochs 20 --batch 16
```

Resume from a checkpoint:

```bash
python training/train.py --resume outputs/weights/color_model_epoch_45.pth --epochs 60
```

Train on a custom dataset:

```bash
python training/train.py --data path/to/images --epochs 30 --batch 8
```

Checkpoints are written to `outputs/weights/`.

## Web App

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`, upload a grayscale video, choose the output resolution, adjust color controls if needed, and download the colorized result.

## Command-Line Inference

```bash
python inference/video_pipeline.py \
  --input path/to/bw_video.mp4 \
  --output outputs/colorized.mp4 \
  --weights outputs/weights/best_model.pth
```

## Training Metrics

If `outputs/metrics.json` exists, create a metrics dashboard with:

```bash
python training/plot_metrics.py --save
```

The saved plot is written to `outputs/training_metrics.png`.

## Requirements

```text
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

## License

This project is for academic and research purposes.
