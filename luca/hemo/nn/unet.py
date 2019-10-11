# import torch
# import torch.nn
# from torch import nn
#
# # from inferno.extensions.models.unet.unet_base import UNetBase
# from inferno.extensions.layers.reshape import Concatenate
# # from inferno.extensions.layers.convolutional import ConvELUND, DeconvELUND, ConvActivationND, DeconvND
# from inferno.extensions.layers.convolutional_blocks import ResidualBlock
# from inferno.extensions.layers.identity import Identity
#
#
#
# class AddELU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.elu = nn.ELU()
#
#     def forward(self, a, b):
#         return self.elu(a+b)
#
# class AddReLU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, a, b):
#         return self.relu(a+b)
#
#
# class UNet(UNetBase):
#     def __init__(self, dim, in_channels, out_channels, depth, initial_features, gain):
#
#         self.dim = dim
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.depth = depth
#         self.initial_features = initial_features
#         self.gain  = gain
#
#         super().__init__(dim=dim, in_channels=in_channels, depth=depth)
#
#
#     def get_num_channels(self, part, index):
#         if part == 'start':
#             return self.initial_features
#         elif part == 'end':
#             return self.out_channels
#         elif part in ('conv_down', 'conv_up', 'bridge', 'ds'):
#             return self.initial_features * self.gain**(index + 1)
#         elif part == 'us':
#             return self.initial_features * self.gain**(index + 1) # now we can add
#         elif part == 'bottom':
#             return self.initial_features * self.gain**(self.depth + 1)
#         elif part == 'combine':
#             # concat us and brige
#             #us = self.initial_features * self.gain**(index + 2)
#             return self.initial_features * self.gain**(index + 1)
#             #return us + bridge
#         else:
#             raise RuntimeError()
#
#
#     def get_downsample_factor(self, index):
#         return 2
#
#     def start_op_factory(self, in_channels, out_channels):
#         return ConvELUND(in_channels, out_channels, 3, dim=self.dim), False
#
#     def end_op_factory(self, in_channels, out_channels):
#         return ConvELUND(in_channels, out_channels, 1, dim=self.dim), False
#
#
#     def conv_op_factory(self, in_channels, out_channels):
#         res = ResidualBlock([ConvELUND(in_channels, out_channels, 3, dim=self.dim),
#                             ConvActivationND(out_channels, out_channels, 3, dim=self.dim, activation=None)],
#                             resample=ConvELUND(in_channels, out_channels, 1, dim=self.dim))
#         rconv = nn.Sequential(nn.Dropout2d(0.3), res, nn.ELU())
#         return rconv
#
#     def conv_down_op_factory(self, in_channels, out_channels, index):
#         return self.conv_op_factory(in_channels, out_channels), False
#
#     def conv_up_op_factory(self, in_channels, out_channels, index):
#         return self.conv_op_factory(in_channels, out_channels), False
#
#     def conv_bottom_op_factory(self, in_channels, out_channels):
#         return self.conv_op_factory(in_channels, out_channels), False
#
#     def downsample_op_factory(self, factor, in_channels, out_channels, index):
#         return ConvELUND(in_channels, out_channels, 3, stride=factor, dim=self.dim), False
#
#     def upsample_op_factory(self, factor, in_channels, out_channels, index):
#         return DeconvND(kernel_size=2, in_channels=in_channels, out_channels=out_channels,
#                 stride=factor, dim=self.dim, activation=None), False
#
#
#
#     def combine_op_factory(self, in_channels_horizonatal, in_channels_down, out_channels, index):
#         assert in_channels_horizonatal == in_channels_down
#         assert in_channels_down == out_channels
#         #nn.Sequential(Add)
#         return AddELU(), False
#
#     def bridge_op_factory(self, in_channels, out_channels, index):
#         return Identity(),False
#
#
#
# class SimpleUNet(UNetBase):
#     def __init__(self, dim, in_channels, out_channels, depth, initial_features, gain):
#
#         self.dim = dim
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.depth = depth
#         self.initial_features = initial_features
#         self.gain  = gain
#
#         super().__init__(dim=dim, in_channels=in_channels, depth=depth)
#
#
#     def get_num_channels(self, part, index):
#         if part == 'start':
#             return self.initial_features
#         elif part == 'end':
#             return self.out_channels
#         elif part in ('conv_down', 'bridge', 'ds'):
#             return self.initial_features * self.gain**(index + 1)
#         elif part in ('conv_up'):
#             return self.initial_features * self.gain**(index )
#         elif part == 'us':
#             return self.initial_features * self.gain**(index + 1)
#         elif part == 'bottom':
#             return self.initial_features * self.gain**(self.depth )
#         elif part == 'combine':
#             return self.initial_features * self.gain**(index + 1)
#         else:
#             raise RuntimeError()
#
#
#
#     def get_downsample_factor(self, index):
#         return 2
#
#     def start_op_factory(self, in_channels, out_channels):
#         return ConvELUND(in_channels, out_channels, 3, dim=self.dim), False
#
#     def end_op_factory(self, in_channels, out_channels):
#         return ConvELUND(in_channels, out_channels, 1, dim=self.dim), False
#
#
#     def conv_op_factory(self, in_channels, out_channels, activate):
#         res = ResidualBlock([ConvELUND(in_channels, out_channels, 3, dim=self.dim),
#                                 ConvActivationND(out_channels, out_channels, 3, dim=self.dim, activation=None)],
#                                 resample=ConvELUND(in_channels, out_channels, 1, dim=self.dim))
#         if activate:
#            return nn.Sequential(nn.Dropout2d(0.3), res, nn.ELU())
#         else:
#             return nn.Sequential(nn.Dropout2d(0.3), res)
#
#     def conv_down_op_factory(self, in_channels, out_channels, index):
#         return self.conv_op_factory(in_channels, out_channels, True), False
#
#     def conv_up_op_factory(self, in_channels, out_channels, index):
#         return self.conv_op_factory(in_channels, out_channels, index==0 ), False
#
#     def conv_bottom_op_factory(self, in_channels, out_channels):
#         return self.conv_op_factory(in_channels, out_channels, False), False
#
#     def downsample_op_factory(self, factor, in_channels, out_channels, index):
#         assert in_channels == out_channels
#         return nn.MaxPool2d(kernel_size=2, stride=2),False
#         #return ConvELUND(in_channels, out_channels, 3, stride=factor, dim=self.dim), False
#
#     def upsample_op_factory(self, factor, in_channels, out_channels, index):
#         assert in_channels == out_channels
#         return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),False
#         #return DeconvND(kernel_size=2, in_channels=in_channels, out_channels=out_channels,
#         #        stride=factor, dim=self.dim, activation=None), False
#
#     def combine_op_factory(self, in_channels_horizonatal, in_channels_down, out_channels, index):
#         assert in_channels_horizonatal == in_channels_down
#         assert in_channels_down == out_channels
#         #nn.Sequential(Add)
#         return AddELU(), False
#
#     def bridge_op_factory(self, in_channels, out_channels, index):
#         return Identity(),False
#
#
#
#
# class GroupNormUNet(UNetBase):
#     def __init__(self, dim, in_channels, out_channels, depth, initial_features, gain):
#
#         self.dim = dim
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.depth = depth
#         self.initial_features = initial_features
#         self.gain  = gain
#
#         super().__init__(dim=dim, in_channels=in_channels, depth=depth)
#
#
#     def get_num_channels(self, part, index):
#         if part == 'start':
#             return self.initial_features
#         elif part == 'end':
#             return self.out_channels
#         elif part in ('conv_down', 'conv_up', 'bridge', 'ds'):
#             return self.initial_features * self.gain**(index + 1)
#         elif part == 'us':
#             return self.initial_features * self.gain**(index + 1) # now we can add
#         elif part == 'bottom':
#             return self.initial_features * self.gain**(self.depth + 1)
#         elif part == 'combine':
#             # concat us and brige
#             #us = self.initial_features * self.gain**(index + 2)
#             return self.initial_features * self.gain**(index + 1)
#             #return us + bridge
#         else:
#             raise RuntimeError()
#
#
#     def get_downsample_factor(self, index):
#         return 2
#
#     def start_op_factory(self, in_channels, out_channels):
#         return  nn.Sequential(ConvActivationND(in_channels, out_channels, 3, activation=None, dim=self.dim),
#                              nn.GroupNorm(num_channels=out_channels, num_groups=4),
#                              nn.ReLU(inplace=True)), False
#
#
#     def end_op_factory(self, in_channels, out_channels):
#         return  nn.Sequential(ConvActivationND(in_channels, out_channels, 3, activation=None, dim=self.dim),
#                              nn.GroupNorm(num_channels=out_channels, num_groups=4),
#                              nn.ReLU(inplace=True)), False
#
#     def conv_op_factory(self, in_channels, out_channels):
#
#         # conv a
#         conv_0_gn_relu = nn.Sequential(ConvActivationND(in_channels, out_channels, 3, activation=None, dim=self.dim),
#                              nn.GroupNorm(num_channels=out_channels, num_groups=4),
#                              nn.ReLU(inplace=True))
#
#         conv_1_gn= nn.Sequential(ConvActivationND(out_channels, out_channels, 3, activation=None, dim=self.dim),
#                              nn.GroupNorm(num_channels=out_channels, num_groups=4))
#
#         res_block = ResidualBlock([conv_1_gn])
#         rconv = nn.Sequential(conv_0_gn_relu, res_block, nn.ReLU(inplace=True))
#         return rconv
#
#     def conv_down_op_factory(self, in_channels, out_channels, index):
#         return self.conv_op_factory(in_channels, out_channels), False
#
#     def conv_up_op_factory(self, in_channels, out_channels, index):
#         return self.conv_op_factory(in_channels, out_channels), False
#
#     def conv_bottom_op_factory(self, in_channels, out_channels):
#         return self.conv_op_factory(in_channels, out_channels), False
#
#     def downsample_op_factory(self, factor, in_channels, out_channels, index):
#         conv = ConvELUND(in_channels, out_channels, 3, stride=factor, dim=self.dim)
#         gn = nn.GroupNorm(num_channels=out_channels, num_groups=4)
#         return nn.Sequential(conv, gn, nn.ReLU(inplace=True)), False
#
#
#     def upsample_op_factory(self, factor, in_channels, out_channels, index):
#         deconv =  DeconvND(kernel_size=2, in_channels=in_channels, out_channels=out_channels,
#                 stride=factor, dim=self.dim, activation=None)
#         gn = nn.GroupNorm(num_channels=out_channels, num_groups=4)
#
#         return nn.Sequential(deconv, gn), False
#
#
#     def combine_op_factory(self, in_channels_horizonatal, in_channels_down, out_channels, index):
#         assert in_channels_horizonatal == in_channels_down
#         assert in_channels_down == out_channels
#         #nn.Sequential(Add)
#         return AddReLU(), False
#
#     def bridge_op_factory(self, in_channels, out_channels, index):
#         return Identity(),False
#
#
#
#
#
# class DeadSimpleUNet(UNetBase):
#     def __init__(self, dim, in_channels, out_channels, depth, initial_features, gain):
#
#         self.dim = dim
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.depth = depth
#         self.initial_features = initial_features
#         self.gain  = gain
#
#         super().__init__(dim=dim, in_channels=in_channels, depth=depth)
#
#
#     def get_num_channels(self, part, index):
#         if part == 'start':
#             return self.initial_features
#         elif part == 'end':
#             return self.out_channels
#         elif part in ('conv_down', 'bridge', 'ds'):
#             return self.initial_features * self.gain**(index + 1)
#         elif part in ('conv_up'):
#             return self.initial_features * self.gain**(index )
#         elif part == 'us':
#             return self.initial_features * self.gain**(index + 1)
#         elif part == 'bottom':
#             return self.initial_features * self.gain**(self.depth )
#         elif part == 'combine':
#             return self.initial_features * self.gain**(index + 1)
#         else:
#             raise RuntimeError()
#
#
#
#     def get_downsample_factor(self, index):
#         return 2
#
#     def start_op_factory(self, in_channels, out_channels):
#
#         conv0 = ConvActivationND(in_channels, out_channels, 1, activation=None, dim=self.dim)
#         gn0 = nn.GroupNorm(num_channels=out_channels, num_groups=4)
#         relu = nn.ReLU(inplace=True)
#
#         return nn.Sequential(conv0, gn0, relu), False
#
#     def end_op_factory(self, in_channels, out_channels):
#
#         conv0 = ConvActivationND(in_channels, out_channels, 1, activation=None, dim=self.dim)
#         gn0 = nn.GroupNorm(num_channels=out_channels, num_groups=4)
#         relu = nn.ReLU(inplace=True)
#
#         return nn.Sequential(conv0, gn0, relu), False
#
#
#     def conv_op_factory(self, in_channels, out_channels, activate):
#
#
#         conv0 = ConvActivationND(in_channels, out_channels, 1, activation=None, dim=self.dim)
#         gn0 = nn.GroupNorm(num_channels=out_channels, num_groups=4)
#         relu = nn.ReLU(inplace=True)
#
#
#         conv1 = ConvActivationND(out_channels, None, 3, depthwise=True, activation=None, dim=self.dim)
#         gn1 = nn.GroupNorm(num_channels=out_channels, num_groups=4)
#
#
#         residual = ResidualBlock([conv1, gn1])
#
#
#         if activate:
#            return nn.Sequential(conv0, gn0, residual, relu)
#         else:
#             return nn.Sequential(conv0, gn0, residual)
#
#     def conv_down_op_factory(self, in_channels, out_channels, index):
#         return self.conv_op_factory(in_channels, out_channels, True), False
#
#     def conv_up_op_factory(self, in_channels, out_channels, index):
#         return self.conv_op_factory(in_channels, out_channels, index==0 ), False
#
#     def conv_bottom_op_factory(self, in_channels, out_channels):
#         return self.conv_op_factory(in_channels, out_channels, False), False
#
#     def downsample_op_factory(self, factor, in_channels, out_channels, index):
#         assert in_channels == out_channels
#         return nn.MaxPool2d(kernel_size=2, stride=2),False
#         #return ConvELUND(in_channels, out_channels, 3, stride=factor, dim=self.dim), False
#
#     def upsample_op_factory(self, factor, in_channels, out_channels, index):
#         assert in_channels == out_channels
#         return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),False
#         #return DeconvND(kernel_size=2, in_channels=in_channels, out_channels=out_channels,
#         #        stride=factor, dim=self.dim, activation=None), False
#
#     def combine_op_factory(self, in_channels_horizonatal, in_channels_down, out_channels, index):
#         assert in_channels_horizonatal == in_channels_down
#         assert in_channels_down == out_channels
#         #nn.Sequential(Add)
#         return AddReLU(), False
#
#     def bridge_op_factory(self, in_channels, out_channels, index):
#         return Identity(),False
