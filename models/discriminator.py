import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator (Pix2Pix style).
    Instead of outputting a single scalar for the whole image,
    it outputs a grid of values (e.g. 30x30), each judging a 70x70 patch.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        feature,
                        kernel_size=4,
                        stride=1 if feature == features[-1] else 2,
                        padding=1,
                        bias=False,
                        padding_mode="reflect",
                    ),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: Concatenated image or just the colorized image.
           For colorization, we usually pass the full 3-channel image (L + predicted ab).
        """
        x = self.initial(x)
        return self.model(x)

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}") # Expected grid output
