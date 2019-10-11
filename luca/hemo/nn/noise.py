#
# import torch
# import torch.nn as nn
#
#
#
# # credit `Ben` https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887/2
# class GaussianNoiseRegularizer(nn.Module):
#     def __init__(self, stddev):
#         super().__init__()
#         self.stddev = stddev
#
#     def forward(self, din):
#         if self.training:
#             return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
#         return din
#
#

