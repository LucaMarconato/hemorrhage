import torch
import torch.nn
from torch import nn


# import torch.nn.modules.loss


class Vae(nn.Module):
    class Loss(nn.Module):
        def __init__(self):
            super().__init__()
            self.rec_loss = nn.MSELoss(reduction='mean')

        def forward(self, outputs, y):
            # logger.add_embedding(mu, global_step=self.i)
            x, x_reconstructed, mu, logvar = outputs
            # x_pooled = x_pooled.view(x_reconstructed.shape[0], x_reconstructed.shape[1])
            # print(f'x.shape = {x.shape}, x_reconstructed.shape = {x_reconstructed.shape}')
            reconstruction_loss = torch.sqrt(torch.sum((x - x_reconstructed) ** 2))
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = reconstruction_loss + kld
            return total_loss

    def __init__(self):
        super(Vae, self).__init__()
        self.bottleneck_size = 100

        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_convolutional = nn.Sequential(
            nn.Conv2d(1, 2, 4, 2, 1),  # in (512, 512) out (256, 256)
            nn.ReLU(),
            nn.Conv2d(2, 4, 4, 2, 1),  # in (256, 256) out (128, 128)
            nn.ReLU(),
            nn.Conv2d(4, 8, 4, 2, 1),  # in (128, 128) out (64, 64)
        )

        self.decoder_convolutional = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 4, 2, 1),
        )

        self.encoder_linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64 * 64 * 8, 64 * 8),
            # nn.ReLU(),
            # nn.Linear(64 * 8, self.bottleneck_size),
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(self.bottleneck_size, 64 * 8),
            nn.ReLU(),
            nn.Linear(64 * 8, 64 * 64 * 8)
        )

        self.fc_mean = nn.Linear(64 * 8, self.bottleneck_size)
        self.fc_logvar = nn.Linear(64 * 8, self.bottleneck_size)

    def encode(self, x):
        x = self.encoder_convolutional(x)
        x = x.view(x.shape[0], x.numel() // x.shape[0])
        x = self.encoder_linear(x)
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_linear(z)
        z = z.view(z.shape[0], 8, 64, 64)
        z = self.decoder_convolutional(z)
        return z

    def forward(self, x):
        with torch.autograd.detect_anomaly():
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            # print(z.shape)
            z = self.decode(z)
            # print(z.shape)
            return x, z, mu, logvar

    def loss_function(self):
        return self.__class__.Loss()


if __name__ == '__main__':
    vae = Vae()
    from hemo.utils.pytorch_summary import Summary

    Summary(vae.encoder_convolutional, input_size=(10, 1, 512, 512))
    Summary(vae.decoder_convolutional, input_size=(10, 8, 64, 64))
