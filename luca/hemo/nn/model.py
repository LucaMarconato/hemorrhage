import math
import torch
import torch.nn
from inferno.extensions.layers.convolutional import *
from torch import nn

# from .cell_pooling import CellPooling
# from .gnn import ResGnn
from .logger import Logger
# from .noise import GaussianNoiseRegularizer
# from .unet import UNet, SimpleUNet, GroupNormUNet, DeadSimpleUNet
from .vae import Vae

logger = Logger.instance()


class Model(torch.nn.Module):
    class Loss(nn.Module):
        def __init__(self):
            super().__init__()
            self.rec_loss = nn.MSELoss(reduction='mean')
            self.i = 1

        def forward(self, outputs, y):
            # print("y",y.shape)
            y = y.view(y.size(1), 1, y.size(2)).permute(0, 1, 2)
            # print("y",y.shape)
            # y has shape (#cells, 3, channels)
            num_cells = y.size(0)
            n_target_channels = y.size(2)

            unetres, rec, mu, logvar, neighbors = outputs
            # rec = rec.view()
            assert neighbors.size(0) == num_cells
            k = neighbors.size(1)

            # this will approach one
            fade_in = 1.0  # - math.exp(-1.0 * float(self.i) * 0.01)

            kld = (-0.5 * 0.25 / float(logvar.numel())) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            skld = (fade_in * kld)
            rec = rec.view((num_cells, 1, n_target_channels))
            # rec_me  = rec[:,0,:]
            # rec_nh  = rec[:,1:,:]
            if rec.dtype == torch.half:
                y = y.half()
            rec_loss_me = self.rec_loss(rec, y)
            total = (rec_loss_me + skld) * (num_cells / 500.0)

            # print("i",self.i, fade_in, "rec",rec_loss_me.item(), 'kld', kld.item(), 'skdl', skld.item(), 'total', total.item())
            if self.i % 10 == 0:
                print("total", total.item(), "rec_loss", rec_loss_me.item(), "kld", kld.item(), "raticdo",
                      (rec_loss_me / kld).item())
            self.i += 1

            logger.add_embedding(mu, global_step=self.i)
            return total

            # if self.i % 10 == 0:

    def __init__(self, in_channels, k):

        super().__init__()

        # for gnn
        self.k = k
        self.divideable_by = 4
        self.unet_out_channels = 20
        # self.unet = SimpleUNet(in_channels=in_channels+1, out_channels=self.unet_out_channels,
        #     initial_features=20, gain=2, depth=2, dim=2)
        self.unet = DeadSimpleUNet(in_channels=in_channels + 1, out_channels=self.unet_out_channels,
                                   initial_features=20, gain=2, depth=1, dim=2)

        self.cell_pool = CellPooling()

        ################################################
        # graph nn
        self.res_gnn = ResGnn(in_channels=self.unet_out_channels + in_channels)

        ################################################
        # vae
        self.vae_in_channels = (self.unet_out_channels + in_channels) * 2  # *(self.k + 1)

        vae_embedding_channels = 15

        vae_out_channels = in_channels
        self.vae = VAE(in_channels=self.vae_in_channels, embedding_channels=vae_embedding_channels,
                       out_channels=vae_out_channels)

        self.selu = nn.SELU()

    def loss_function(self):
        return self.__class__.Loss()

    def forward(self, x, mask, neighbors):

        x_raw = x

        # remove batch from neighbors
        neighbors = neighbors[0, ...]

        # add mask to input
        if x.dtype == torch.half:
            bmask = (mask >= 1).half()
        else:
            bmask = (mask >= 1).float()

        x = torch.cat([x, bmask], dim=1)

        unet_res = self.unet(x)

        to_pool = torch.cat([unet_res, x_raw], dim=1)

        # pooling part-
        xp = self.cell_pool(to_pool, mask)

        # noise
        # xp = self.gnoise(xp)

        # graph neural network
        xpp = self.res_gnn(xp, neighbors)
        xp = torch.cat([xp, xpp], dim=1)
        decoded, mu, logvar = self.vae(xp)

        # we also return neighbors
        # since we might need them in loss
        # function
        return (unet_res, decoded, mu, logvar) + (neighbors,)
