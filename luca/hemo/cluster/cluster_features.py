# import numpy
# from scipy.ndimage.filters import gaussian_filter1d
#
# from ..utils import clip_normalize
#
# class DistanceToBin(object):
#
#     def __len__(self):
#         raise NotImplementedError()
#     def __call__(self):
#         raise NotImplementedError()
#
# class SqrtDistBining(DistanceToBin):
#
#     def __init__(self, clip_range = (5, 100), n_bins=10, power=0.01):
#         super().__init__()
#         self.clip_range = numpy.array(clip_range)
#         self.n_bins = n_bins
#         self.power = power
#     def __len__(self):
#         return self.n_bins
#
#
#     def __call__(self, dists):
#         dists = numpy.require(dists, dtype='float')
#         dists = numpy.power(dists, self.power)
#         drange  =  numpy.power(self.clip_range, self.power)
#         dists = clip_normalize(dists, *drange)
#         dists *= self.n_bins - 1
#         dists = numpy.round(dists).astype('int')
#
#         return dists
#
#
#
# class FeatureAccumulator(object):
#
#     def __init__(self, n_clusters, binning):
#         self.binning = binning
#         self.hist = numpy.zeros([n_clusters, n_clusters, len(binning)])
#         self.n_clusters = n_clusters
#
#     def acc(self, assignments, neighbours, distances):
#         n_cells = neighbours.shape[0]
#         n_neighbours = neighbours.shape[1]
#
#         bloated_assignments = numpy.repeat(assignments, repeats=n_neighbours)
#
#
#         nh_assignments = numpy.take(assignments, neighbours)
#         dist_bin = self.binning(distances)
#
#         flat_nh_assignments = numpy.ravel(nh_assignments)
#         flat_dist_bin = numpy.ravel(dist_bin)
#         numpy.add.at(self.hist, (bloated_assignments, flat_nh_assignments, flat_dist_bin), 1)
#
#     def get_features(self):
#         # normalize hist
#         s = numpy.sum(self.hist, axis=(1,2))
#
#         h = self.hist / s[:,None]
#         h = numpy.nan_to_num(h)
#         # smoothing along distance axis
#         h = gaussian_filter1d(input=h, sigma=0.5,
#                               axis=1, mode='constant',
#                               truncate=4)
#         return h.reshape([self.n_clusters, -1])
#
#
#
# def cell_nh_feat(
#     features01,
#     neighbours,
#     distances,
#     dist_binning,
#     n_feature_bins
# ):
#     n_cells = neighbours.shape[0]
#     n_neighbours = neighbours.shape[1]
#     n_features = features01.shape[1]
#     n_dist_bins = len(dist_binning)
#
#     features01 = numpy.clip(features01, 0, 1)
#     # assert features01.min() >= 0
#     # if features01.max() > 1:
#     #     print(f'features01.max() = {features01.max()}')
#     # assert features01.max() <= 1
#     assert features01.shape[0] == n_cells
#     assert distances.shape[0] == n_cells
#     assert distances.shape[1] == n_neighbours
#
#     result = numpy.zeros([n_cells, n_dist_bins, n_features, n_feature_bins])
#
#
#     cell_i = numpy.arange(n_cells)
#     cell_i = numpy.repeat(cell_i, repeats=n_neighbours)
#
#
#
#
#
#
#     dist_bin = dist_binning(distances)
#
#     for fi in range(n_features):
#
#         nh_features01 = numpy.take(features01[:,fi], neighbours)
#         nh_feature_bins = numpy.round(nh_features01 * (n_dist_bins - 1)).astype('int')
#
#
#
#         numpy.add.at(result, (cell_i.ravel(), dist_bin.ravel(), fi, nh_feature_bins.ravel()), 1)
#
#     return result.reshape([n_cells, -1])
