import torch
import torch.nn as nn
from torchvision import models

class GANLoss(nn.Module):
    """
    Standard GAN Loss (BCEWithLogits)
    """
    def __init__(self, target_real_label=0.9, target_fake_label=0.0):
        super().__init__()
        # Label Smoothing: 0.9 instead of 1.0 prevents D from becoming too confident,
        # which helps the Generator learn more stable colors.
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

class PerceptualLoss(nn.Module):
    """
    VGG16-based Perceptual Loss to preserve structure and content.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg = nn.Sequential(*list(vgg[:16])).eval() # Use up to relu3_3
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Normalization constants expected by the pretrained VGG model.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x_rgb, y_rgb):
        """
        x_rgb: (B, 3, H, W) RGB generated (approximate)
        y_rgb: (B, 3, H, W) RGB target (approximate)
        """
        # Normalize for VGG
        x = (x_rgb - self.mean) / self.std
        y = (y_rgb - self.mean) / self.std
        
        features_x = self.vgg(x)
        features_y = self.vgg(y)
        return nn.functional.l1_loss(features_x, features_y)

class ColorizationLoss(nn.Module):
    def __init__(self, lambda_l1=100.0, lambda_perceptual=20.0, lambda_gan=1.0):
        """
        Combined loss for colorization.
        SmoothL1 loss for pixel accuracy, Perceptual for structure, GAN for realism.
        """
        super(ColorizationLoss, self).__init__()
        # SmoothL1 is less sensitive to outliers, which reduces "blotchy" artifacts.
        self.l1_loss = nn.SmoothL1Loss()
        self.perceptual_loss = PerceptualLoss()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_gan = lambda_gan

    def forward(self, preds_ab, target_ab, preds_rgb=None, target_rgb=None, gan_loss_val=0):
        """
        preds_ab: (B, 2, H, W)
        target_ab: (B, 2, H, W)
        preds_rgb: (B, 3, H, W) - required for perceptual loss
        target_rgb: (B, 3, H, W) - required for perceptual loss
        """
        l1 = self.l1_loss(preds_ab, target_ab)
        
        perc = 0
        if preds_rgb is not None and target_rgb is not None:
            perc = self.perceptual_loss(preds_rgb, target_rgb)
        
        total = (self.lambda_l1 * l1) + (self.lambda_perceptual * perc) + (self.lambda_gan * gan_loss_val)
        return total, l1, perc
