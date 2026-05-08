import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
from models.discriminator import Discriminator
from utils.preprocessing import ColorizationDataset, get_image_paths
from utils.color import lab_to_rgb_tensor
from training.loss import ColorizationLoss, GANLoss

# ── AMP (Automatic Mixed Precision) ────────────────────────────────────────
try:
    from torch.amp import autocast, GradScaler
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_DEVICE = "cuda"

def add_noise(tensor, std=0.01):
    return tensor + torch.randn_like(tensor) * std if std > 0 else tensor

def train(dataset_path=None, epochs=20, batch_size=16, lr=2e-4, save_dir="outputs/weights", image_size=128, resume_path=None):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device : {device} | AMP: {'ON' if use_amp else 'OFF'}")

    # ── Models & Optimisers ────────────────────────────────────────────────
    net_G = UNet(n_channels=1, n_classes=2).to(device)
    net_D = Discriminator(in_channels=3).to(device)

    criterion_G = ColorizationLoss().to(device)
    criterion_GAN = GANLoss().to(device)

    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(0.5, 0.999))
    # We use a lower learning rate for the Discriminator to prevent it from overpowering G.
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))

    # ── Resume Logic ───────────────────────────────────────────────────────
    start_epoch = 1
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        net_G.load_state_dict(ckpt['model_state_dict'])
        optimizer_G.load_state_dict(ckpt['optimizer_G_state_dict'])
        if 'discriminator_state_dict' in ckpt:
            net_D.load_state_dict(ckpt['discriminator_state_dict'])
            optimizer_D.load_state_dict(ckpt['optimizer_D_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # ── Dataset Auto-Detection & Split ────────────────────────────────────
    print("\nLoading dataset...")
    coco_opt = os.path.join("data", "coco_128_subset")
    tiny_opt = os.path.join("data", "tiny_imagenet_128_subset")
    
    if dataset_path:
        full_dataset = ColorizationDataset(root_dir=dataset_path, is_optimized="subset" in dataset_path, size=image_size)
    elif os.path.isdir(coco_opt):
        full_dataset = ColorizationDataset(root_dir=coco_opt, is_optimized=True, size=image_size)
    elif os.path.isdir(tiny_opt):
        full_dataset = ColorizationDataset(root_dir=tiny_opt, is_optimized=True, size=image_size)
    else:
        full_dataset = ColorizationDataset(root_dir=os.path.join("data", "tiny-imagenet-200"), size=image_size)

    # 90/10 Train/Val split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.split = 'val' # Disable augmentations for validation

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_amp, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs)
    scaler_G, scaler_D = GradScaler(enabled=use_amp), GradScaler(enabled=use_amp)

    # ── Training Loop ─────────────────────────────────────────────────────
    best_val_loss = float('inf')
    print(f"\nStarting Optimized Training: Epoch {start_epoch} to {epochs}")
    print(f"Train samples: {train_size} | Val samples: {val_size}\n")
    
    try:
        for epoch in range(start_epoch, epochs + 1):
            net_G.train(); net_D.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs} [Train]")
            for batch in pbar:
                L, real_ab = batch['L'].to(device), batch['ab'].to(device)
                real_img = torch.cat([L, real_ab], dim=1)

                # 1. Forward G
                with autocast(device_type=_AMP_DEVICE, enabled=use_amp):
                    fake_ab = net_G(L)
                    fake_img = torch.cat([L, fake_ab], dim=1)

                # 2. Update D
                optimizer_D.zero_grad(set_to_none=True)
                with autocast(device_type=_AMP_DEVICE, enabled=use_amp):
                    loss_D = (criterion_GAN(net_D(add_noise(real_img)), True) + 
                             criterion_GAN(net_D(add_noise(fake_img.detach())), False)) * 0.5
                scaler_D.scale(loss_D).backward(); scaler_D.step(optimizer_D); scaler_D.update()

                # 3. Update G
                optimizer_G.zero_grad(set_to_none=True)
                with autocast(device_type=_AMP_DEVICE, enabled=use_amp):
                    loss_G_GAN = criterion_GAN(net_D(fake_img), True)
                    loss_G, l1, perc = criterion_G(fake_ab, real_ab, lab_to_rgb_tensor(L, fake_ab), lab_to_rgb_tensor(L, real_ab), loss_G_GAN)
                scaler_G.scale(loss_G).backward(); scaler_G.step(optimizer_G); scaler_G.update()

                pbar.set_postfix({'G': f"{loss_G.item():.2f}", 'D': f"{loss_D.item():.2f}"})

            # ── Validation Loop ───────────────────────────────────────────
            net_G.eval()
            val_l1 = 0
            with torch.no_grad():
                for v_batch in tqdm(val_loader, desc="Validating", leave=False):
                    v_L, v_real_ab = v_batch['L'].to(device), v_batch['ab'].to(device)
                    v_fake_ab = net_G(v_L)
                    val_l1 += torch.nn.functional.l1_loss(v_fake_ab, v_real_ab).item()
            
            avg_val_l1 = val_l1 / len(val_loader)
            print(f"  Epoch {epoch} Val L1 Loss: {avg_val_l1:.4f}")

            # Save best model
            if avg_val_l1 < best_val_loss:
                best_val_loss = avg_val_l1
                torch.save({'model_state_dict': net_G.state_dict()}, os.path.join(save_dir, "best_model.pth"))
                print(f"  ✨ New Best Model Saved (Val Loss: {avg_val_l1:.4f})")

            scheduler_G.step(); scheduler_D.step()
            if epoch % 5 == 0 or epoch == epochs:
                torch.save({'epoch': epoch, 'model_state_dict': net_G.state_dict(), 'discriminator_state_dict': net_D.state_dict(),
                            'optimizer_G_state_dict': optimizer_G.state_dict(), 'optimizer_D_state_dict': optimizer_D.state_dict()}, 
                            os.path.join(save_dir, f"color_model_epoch_{epoch}.pth"))

    except KeyboardInterrupt:
        torch.save({'epoch': epoch, 'model_state_dict': net_G.state_dict(), 'discriminator_state_dict': net_D.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(), 'optimizer_D_state_dict': optimizer_D.state_dict()}, 
                    os.path.join(save_dir, "interrupted.pth"))
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", "--batch_size", type=int, default=16, dest="batch")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()
    train(dataset_path=args.data or None, epochs=args.epochs, batch_size=args.batch, resume_path=args.resume or None)
