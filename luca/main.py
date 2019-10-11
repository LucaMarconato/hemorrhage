import click
import pickle
import itertools
import matplotlib
import pushover_notification
import pylab
import torch.utils.data
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from hemo.ds.dataset import *
from hemo.nn.model import *
from hemo.nn.vae import Vae


root_folder = '/data/hemo'

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# @click.command()
# def sandbox():


# model = Model(in_channels=38, k=3)
# ds_train  = SpatialTranscriptomicsDs(root_folder=root_folder,k=model.k, divideable_by=model.divideable_by)
# train_loader = data.DataLoader(ds_train, batch_size=1,
#                                             num_workers=4)
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
# for i in range(len(ds_train)):
#     print(i)
#     image, mask, nn, targets = ds_train[i]
#     scaler.partial_fit(targets)
#     # break
#
# print('self.tmax_mean = numpy.{}'.format(repr(scaler.mean_)))
# print('self.tmax_var = numpy.{}'.format(repr(scaler.var_)))

import gc
import inferno.trainers.callbacks


class ShowMinimalConsoleInfo(inferno.trainers.callbacks.Callback):
    """
    Callback to show only minimum training info on console
    viz. current epoch number, current learning rate,
    training loss and training error if exists.
    """

    def __init__(self, *args, **kwargs):
        super(ShowMinimalConsoleInfo, self).__init__(*args, **kwargs)

    def begin_of_fit(self, **_):
        self.trainer.quiet()

    def end_of_epoch(self, **_):
        training_loss = self.trainer.get_state('training_loss')
        training_error = self.trainer.get_state('training_error')
        learning_rate = self.trainer.get_state('learning_rate')

        self.trainer.console.info("--------------------------------")
        self.trainer.console.info("Epoch " + str(self.trainer.epoch_count))
        if training_loss is not None:
            self.trainer.console.info("Train Loss " + str(training_loss.item()))
        if training_error is not None:
            self.trainer.console.info("Train Error " + str(training_error.item()))
        self.trainer.console.info("Current LR " + str(learning_rate))


class GarbageCollection(inferno.trainers.callbacks.Callback):
    """
    Callback that triggers garbage collection at the end of every
    training iteration in order to reduce the memory footprint of training
    """

    def end_of_training_iteration(self, **_):
        # print('calling the garbage collector')
        gc.collect()


@click.command()
@click.option('--load', default=False, type=bool, help='load existing state')
@click.option('--folder', default='derived_data', type=str, help='derived data directory')
def train(load, folder):
    my_train(load, folder)


## workaround to make the debugger work with click in PyCharm
def my_train(load, folder):
    click.echo('starting training')
    os.makedirs(folder, exist_ok=True)

    # joint_transform = Compose(
    #     RandomRotate(),
    #     RandomTranspose(),
    #     RandomFlip()
    # )

    # setup logger
    os.makedirs('derived_data/log', exist_ok=True)
    Logger.instance().setup('derived_data/log')

    vae = Vae()

    ds = HemoDataset(root_folder=root_folder, image_transform=None, training=True)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1536, num_workers=8)

    # Build trainer
    trainer = Trainer(vae)
    trainer.save_to_directory(folder)

    if load:
        trainer.load()
    # trainer.cuda(devices=[0, 1])
    trainer.cuda()

    trainer.build_criterion(vae.loss_function())
    trainer.build_optimizer('Adam', lr=0.001)
    # trainer.validate_every((2, 'epochs'))
    trainer.save_every((1, 'epochs'))
    trainer.set_max_num_epochs(100)
    trainer.build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                           log_images_every='never',
                                           # log_images_every=(1, 'iteration'),
                                           log_directory=folder))

    # Bind loaders
    trainer.bind_loader('train', train_loader, num_inputs=1, num_targets=1)

    # bind callbacks
    trainer.register_callback(GarbageCollection(), trigger='end_of_training_iteration')
    trainer.register_callback(ShowMinimalConsoleInfo(), trigger='end_of_training_iteration')

    # trainer.bind_loader('train', train_loader, num_inputs=3, num_targets=1)
    trainer.fit()
    pushover_notification.send('embeddings generated')


@click.command()
@click.option('--folder', default='derived_data', type=str, help='derived data directory')
@click.option('--n', default='0', type=int, help='embed the first n mini batches, n = 0 for all')
def embed(folder, n):
    my_embed(folder, n)


def my_embed(folder, n):
    click.echo('computing embeddings')

    vae = Vae()
    ds_train = HemoDataset(root_folder=root_folder, training=True)
    ds_test = HemoDataset(root_folder=root_folder, training=False)
    batch_size = 512
    train_loader = data.DataLoader(ds_train, batch_size=batch_size, num_workers=8)
    test_loader = data.DataLoader(ds_test, batch_size=batch_size, num_workers=8)

    trainer = Trainer(vae)
    trainer.save_to_directory(folder)
    trainer.load()
    trainer.cuda()
    trainer.eval_mode()
    i = 0

    def compute_embedding_for_loader(loader, training_set: bool):
        with torch.no_grad():
            all_mu = []
            all_filenames = []
            with tqdm(total=len(loader)) as bar:
                for k, x in enumerate(loader):
                    nonlocal i
                    i += 1
                    if n != 0 and i > n:
                        break
                    if training_set:
                        image, targets = x
                    else:
                        image = x
                    image = trainer.to_device(image)
                    _, x_reconstructed_batch, mu_batch, _ = trainer.apply_model(image)
                    image = image.detach().cpu().numpy()
                    x_reconstructed_batch = x_reconstructed_batch.detach().cpu().numpy()
                    mu_batch = mu_batch.detach().cpu().numpy()
                    all_mu.append(mu_batch)
                    start = k * loader.batch_size
                    end = start + loader.batch_size
                    if training_set:
                        paths = loader.dataset.training_file_paths
                    else:
                        paths = loader.dataset.testing_file_paths
                    filenames = [os.path.basename(paths[j]) for j in range(start, end)]
                    all_filenames.append(filenames)
                    bar.update(1)

            all_mu = np.concatenate(all_mu, axis=0)
            all_filenames = list(itertools.chain.from_iterable(all_filenames))
            torch.cuda.empty_cache()
            gc.collect()
            return all_mu, all_filenames

    h5_path = os.path.join(folder, 'embeddings.h5')
    with h5py.File(h5_path, 'w') as f5:
        embeddings, filenames = compute_embedding_for_loader(train_loader, training_set=True)
        # breakpoint()
        f5['training_data/embeddings'] = embeddings
        pickle_path = os.path.join(folder, 'embeddings_training_data_row_names.pickle')
        pickle.dump(filenames, open(pickle_path, 'wb'))
        pushover_notification.send('training embeddings generated')

        embeddings, filenames = compute_embedding_for_loader(test_loader, training_set=False)
        f5['testing_data/embeddings'] = embeddings
        pickle_path = os.path.join(folder, 'embeddings_testing_data_row_names.pickle')
        pickle.dump(filenames, open(pickle_path, 'wb'))
        pushover_notification.send('testing embeddings generated')


# if False:

# if show_umap:
#     app = pg.mkQApp()
#     viewer = LayerViewerWidget()
#     viewer.setWindowTitle('LayerViewer')
#     viewer.show()
#     image = numpy.moveaxis(image, 0, 2)
#     image = numpy.swapaxes(image, 0, 1)
#
#     unetres = numpy.moveaxis(unetres[0, ...], 0, 2)
#     unetres = numpy.swapaxes(unetres, 0, 1)
#     mask = numpy.swapaxes(mask[0, ...], 0, 1)
#
#     layer = MultiChannelImageLayer(name='img', data=image)
#     viewer.addLayer(layer=layer)
#
#     layer = MultiChannelImageLayer(name='umap', data=unetres)
#     viewer.addLayer(layer=layer)
#
#     # labels = numpy.zeros(image.shape[0:2], dtype='uint8')
#     label_layer = ObjectLayer(name='labels', data=mask)
#     viewer.addLayer(layer=label_layer)
#     QtGui.QApplication.instance().exec_()
#
# # import seaborn
# # import pandas as pd
# # seaborn.pairplot(pd.DataFrame(res[:,0:5]))
# # pylab.show()
#
# embedding, embedding_list = list_umap(x_list=all_res, n_neighbors=30, min_dist=0.0)
#
# for i in range(n):
#     img_name = ds_train.image_filenames[i]
#     img_name = os.path.basename(img_name)
#     print(i, img_name)
#
#     fname = os.path.join(out_dir, f'{img_name}.h5')
#     f5file = h5py.File(fname, 'r+')
#
#     # f5file['labels'] = numpy.swapaxes(mask[0,...],0,1)
#     f5file['vae_embedding'] = all_res[i]
#     f5file['umap_embedding'] = embedding_list[i]
#     f5file.close()
#
# # A random colormap for matplotlib
# cmap = matplotlib.colors.ListedColormap(numpy.random.rand(1000000, 3))
#
# n = len(all_ids)
# perm = numpy.random.permutation(numpy.arange(n))
#
# pylab.scatter(x=embedding[perm, 0], y=embedding[perm, 1], c=all_ids[perm], s=12, edgecolors='face',
#               cmap=cmap)
# pylab.show()


@click.command()
@click.option('--folder', default='log', type=str, help='load dir')
@click.option('--n', default=10, type=int, help='compute the umap on n samples')
@click.option('--show_umap', default=False, type=bool, help='shown the umap')
@click.option('--out_dir', default='out', type=str, help='output folder')
def predict(folder, n, show_umap, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    LOG_DIRECTORY = folder
    SAVE_DIRECTORY = folder
    DATASET_DIRECTORY = folder
    click.echo('Predict')

    model = Model(in_channels=38, k=3)
    ds_train = SpatialTranscriptomicsDs(root_folder=root_folder, k=model.k, training=False,
                                        divideable_by=model.divideable_by)
    train_loader = data.DataLoader(ds_train, batch_size=1,
                                   num_workers=0)
    if n == 0:
        n = len(ds_train)
    print('N', n)

    trainer = Trainer(model)
    trainer.save_to_directory(SAVE_DIRECTORY)
    trainer.load()
    trainer.eval_mode()
    with torch.no_grad():
        all_res = []
        all_muh = []
        all_targets = []
        all_ids = []
        num_cells = []
        for i in range(n):

            torch.cuda.empty_cache()
            gc.collect()
            print(i)
            img, mask, neighbours, targets = ds_train[i]
            n_cells = neighbours.shape[0]
            img = trainer.to_device(torch.from_numpy(img))
            mask = trainer.to_device(torch.from_numpy(mask))
            neighbours = trainer.to_device(torch.from_numpy(neighbours))

            unetres, rec, muh, logvar, nn = trainer.apply_model(img[None, ...], mask[None, ...], neighbours[None, ...])

            # img.detach()
            mask = mask.cpu().numpy()
            nn.detach()
            logvar.detach()
            rec = rec.detach().cpu().numpy()
            image = img.detach().cpu().numpy()  # [0,:,:].swapaxes(0,1)
            unetres = unetres.detach().cpu().numpy()  # [0,:,:].swapaxes(0,1)

            if show_umap:
                app = pg.mkQApp()
                viewer = LayerViewerWidget()
                viewer.setWindowTitle('LayerViewer')
                viewer.show()
                image = numpy.moveaxis(image, 0, 2)
                image = numpy.swapaxes(image, 0, 1)

                unetres = numpy.moveaxis(unetres[0, ...], 0, 2)
                unetres = numpy.swapaxes(unetres, 0, 1)
                mask = numpy.swapaxes(mask[0, ...], 0, 1)

                layer = MultiChannelImageLayer(name='img', data=image)
                viewer.addLayer(layer=layer)

                layer = MultiChannelImageLayer(name='umap', data=unetres)
                viewer.addLayer(layer=layer)

                # labels = numpy.zeros(image.shape[0:2], dtype='uint8')
                label_layer = ObjectLayer(name='labels', data=mask)
                viewer.addLayer(layer=label_layer)
                QtGui.QApplication.instance().exec_()

            muh = muh.detach().cpu().numpy()  # [0,:,:].swapaxes(0,1)
            res = muh
            all_res.append(res)
            all_ids.append(numpy.ones(n_cells) * i)

            # num_cells.append(n_cells)
            img_name = ds_train.image_filenames[i]
            img_name = os.path.basename(img_name)
            print(i, img_name)

            fname = os.path.join(out_dir, f'{img_name}.h5')
            print(fname)
            with h5py.File(fname, 'w') as f5:
                try:
                    del f5['labels']
                except:
                    pass
                f5['labels'] = numpy.swapaxes(mask[0, ...], 0, 1)

                try:
                    del f5['vae_embedding']
                except:
                    pass
                f5['vae_embedding'] = res

                try:
                    del f5['targets']
                except:
                    pass
                f5['targets'] = targets

                try:
                    del f5['masks']
                except:
                    pass
                f5['masks'] = mask

                try:
                    del f5['rec']
                except:
                    pass
                f5['rec'] = rec

        # res = numpy.concatenate(all_res, axis=0)
        all_ids = numpy.concatenate(all_ids, axis=0)

        if False:

            # import seaborn
            # import pandas as pd
            # seaborn.pairplot(pd.DataFrame(res[:,0:5]))
            # pylab.show()

            embedding, embedding_list = list_umap(x_list=all_res, n_neighbors=30, min_dist=0.0)

            for i in range(n):
                img_name = ds_train.image_filenames[i]
                img_name = os.path.basename(img_name)
                print(i, img_name)

                fname = os.path.join(out_dir, f'{img_name}.h5')
                f5file = h5py.File(fname, 'r+')

                # f5file['labels'] = numpy.swapaxes(mask[0,...],0,1)
                f5file['vae_embedding'] = all_res[i]
                f5file['umap_embedding'] = embedding_list[i]
                f5file.close()

            # A random colormap for matplotlib
            cmap = matplotlib.colors.ListedColormap(numpy.random.rand(1000000, 3))

            n = len(all_ids)
            perm = numpy.random.permutation(numpy.arange(n))

            pylab.scatter(x=embedding[perm, 0], y=embedding[perm, 1], c=all_ids[perm], s=12, edgecolors='face',
                          cmap=cmap)
            pylab.show()
    import pushover_notification
    pushover_notification.send('predictions generated')


@click.command()
@click.option('--h5folder', default='log', type=str, help='load dir')
@click.option('--k', default=50, type=int, help='k')
@click.option('--centers_file', default='kmean_centers.h5', type=str, help='where to store kmean_centers')
def cluster(h5folder, k, centers_file):
    def yield_f5s():
        for filename in os.listdir(h5folder):
            if filename.endswith('.h5'):
                filename = os.path.join(h5folder, filename)
                # Qprint(filename)
                with h5py.File(filename) as f5:
                    # vae_embedding = f5['vae_embedding'][...]
                    yield f5

    print('find embedding min max')
    mima = sklearn.preprocessing.MinMaxScaler()
    for f5 in yield_f5s():
        vae_embedding = f5['vae_embedding'][...]
        mima.partial_fit(vae_embedding)

    print('compute nh feat')
    to_bin = SqrtDistBining()
    scaler_vae_embedding = sklearn.preprocessing.StandardScaler()
    scaler_hist = sklearn.preprocessing.StandardScaler()
    x_hist = []
    for f5 in yield_f5s():
        vae_embedding = f5['vae_embedding'][...]
        labels = f5['labels'][...]

        center_of_mass = get_region_centers(labels)
        neighbours, distances = get_knn(center_of_mass, 7)

        features01 = mima.transform(vae_embedding)

        hfeat = cell_nh_feat(
            features01=features01,
            neighbours=neighbours,
            distances=distances,
            dist_binning=to_bin,
            n_feature_bins=10
        )
        scaler_hist.partial_fit(hfeat)
        scaler_vae_embedding.partial_fit(vae_embedding)
        x_hist.append(hfeat)

    print('scale feat')
    all_feat = []
    for f5, hist in zip(yield_f5s(), x_hist):
        vae_embedding = f5['vae_embedding'][...]

        scaled_hist = scaler_hist.transform(hist)
        scaled_hist *= (vae_embedding.shape[1] / scaled_hist.shape[1])
        scaled_embedding = scaler_vae_embedding.transform(vae_embedding) * 1.0
        feat = numpy.concatenate([scaled_hist, scaled_embedding], axis=1)
        all_feat.append(feat)

    print('kmeans')
    x = numpy.concatenate(all_feat, axis=0)
    assert numpy.isfinite(x).all()

    cluster_alg = MiniBatchKMeans(n_clusters=k, n_init=10)
    cluster_alg.fit(x)

    print('predict')

    with h5py.File(centers_file) as f5:
        try:
            del f5['centers']
        except:
            pass
        assert numpy.isfinite(cluster_alg.cluster_centers_).all()
        f5['centers'] = cluster_alg.cluster_centers_

    for f5, hist in zip(yield_f5s(), x_hist):
        vae_embedding = f5['vae_embedding'][...]

        scaled_hist = scaler_hist.transform(hist)
        scaled_hist *= (vae_embedding.shape[1] / scaled_hist.shape[1])
        scaled_embedding = scaler_vae_embedding.transform(vae_embedding) * 1.0
        feat = numpy.concatenate([scaled_hist, scaled_embedding], axis=1)

        cluster_assignment = cluster_alg.predict(feat)
        try:
            del f5['cluster_assignment']
        except:
            pass

        f5['cluster_assignment'] = cluster_assignment


@click.command()
@click.option('--h5folder', default='log', type=str, help='load dir')
@click.option('--centers_file', default='kmean_centers.h5', type=str, help='where to load kmean_centers')
def visualize_clusters(h5folder, centers_file):
    with h5py.File(centers_file, 'r') as f5:
        centers = f5['centers'][:]

    n = centers.shape[0] + 1
    lut = numpy.random.randint(low=0, high=255, size=n * 4)
    lut = lut.reshape([n, 4])
    lut[:, 3] = 255
    lut[0, 3] = 0

    for filename in os.listdir(h5folder):
        if filename.endswith('.h5'):
            filename = os.path.join(h5folder, filename)
            with h5py.File(filename) as f5:
                labels = f5['labels'][...]
                cluster_assignment = f5['cluster_assignment'][...] + 1
                cluster_assignment = numpy.concatenate([numpy.array([0]), cluster_assignment])

            print(cluster_assignment)

            img_name = os.path.basename(filename)[:-3]
            img_name = os.path.join(root_folder, 'ome', img_name)

            image = skimage.io.imread(img_name)

            labels = numpy.swapaxes(labels, 0, 1)
            assigned_labels = numpy.take(cluster_assignment, labels)
            # print(image.shape, labels.shape)

            image, _ = pad_test(image, labels, 4)

            app = App.instance()

            viewer = LayerViewerWidget()
            viewer.setWindowTitle('LayerViewer')
            viewer.show()
            image = numpy.moveaxis(image, 0, 2)
            image = numpy.swapaxes(image, 0, 1)
            labels = numpy.swapaxes(labels, 0, 1)
            assigned_labels = numpy.swapaxes(assigned_labels, 0, 1)

            layer = MultiChannelImageLayer(name='img', data=image)
            viewer.addLayer(layer=layer)

            label_layer = ObjectLayer(name='labels', data=assigned_labels, lut=lut)
            viewer.addLayer(layer=label_layer)
            QtGui.QApplication.instance().exec_()


@click.command()
@click.option('--h5folder', default='log', type=str, help='load dir')
@click.option('--centers_file', default='kmean_centers.h5', type=str, help='where to load kmean_centers')
@click.option('--cluster_assignments_file', default='cluster_assignment.h5', type=str, help='where to store aissgments')
def visualize_refined_clusters(h5folder, centers_file, cluster_assignments_file):
    with h5py.File(cluster_assignments_file, 'r') as f5:
        cluster_assignments2 = f5['cluster_assignment'][:] + 1
        cluster_assignments2 = numpy.concatenate([numpy.array([0]), cluster_assignments2])
    n_clusters = cluster_assignments2.shape[0]

    n = cluster_assignments2.max() + 1
    lut = numpy.random.randint(low=0, high=255, size=n * 4)
    lut = lut.reshape([n, 4])
    lut[:, 3] = 255
    lut[0, 3] = 0

    for filename in os.listdir(h5folder):
        if filename.endswith('.h5'):

            filename = os.path.join(h5folder, filename)
            with h5py.File(filename) as f5:
                labels = f5['labels'][...]
                cluster_assignment = f5['cluster_assignment'][...] + 1
                cluster_assignment = numpy.concatenate([numpy.array([0]), cluster_assignment])

            if len(cluster_assignment) > 100:
                img_name = os.path.basename(filename)[:-3]
                img_name = os.path.join(root_folder, 'ome', img_name)

                image = skimage.io.imread(img_name)

                labels = numpy.swapaxes(labels, 0, 1)
                assigned_labels = numpy.take(cluster_assignment, labels)
                assigned_labels = numpy.take(cluster_assignments2, assigned_labels)
                # print(image.shape, labels.shape)

                image, _ = pad_test(image, labels, 4)

                app = App.instance()

                viewer = LayerViewerWidget()
                viewer.setWindowTitle('LayerViewer')
                viewer.show()
                image = numpy.moveaxis(image, 0, 2)
                image = numpy.swapaxes(image, 0, 1)
                labels = numpy.swapaxes(labels, 0, 1)
                assigned_labels = numpy.swapaxes(assigned_labels, 0, 1)

                layer = MultiChannelImageLayer(name='img', data=image)
                viewer.addLayer(layer=layer)

                label_layer = ObjectLayer(name='labels', data=assigned_labels, lut=lut)
                viewer.addLayer(layer=label_layer)
                QtGui.QApplication.instance().exec_()


@click.command()
@click.option('--h5folder', default='log', type=str, help='load dir')
@click.option('--k', default=10, type=int, help='new number of clusters')
@click.option('--k_knn', default=6, type=int, help='k')
@click.option('--centers_file', default='kmean_centers.h5', type=str, help='where to load kmean_centers')
@click.option('--cluster_assignments_file', default='cluster_assignment.h5', type=str, help='where to store aissgments')
def refine_clusters(h5folder, k, centers_file, k_knn, cluster_assignments_file):
    with h5py.File(centers_file, 'r') as f5:
        centers = f5['centers'][:]

    n_clusters = centers.shape[0]

    to_bin = SqrtDistBining()
    acc = FeatureAccumulator(n_clusters=n_clusters, binning=to_bin)

    for filename in os.listdir(h5folder):
        if filename.endswith('.h5'):
            filename = os.path.join(h5folder, filename)
            with h5py.File(filename) as f5:
                labels = f5['labels'][...]
                cluster_assignment = f5['cluster_assignment'][...]
                n_cells = cluster_assignment.shape[0]
                if n_cells > 100:
                    center_of_mass = get_region_centers(labels)
                    neighbours, distances = get_knn(center_of_mass, k_knn)
                    acc.acc(assignments=cluster_assignment, neighbours=neighbours, distances=distances)

    x = acc.get_features()
    assert numpy.isfinite(x).all()
    x = numpy.concatenate([centers, x], axis=1)
    scaler = sklearn.preprocessing.StandardScaler()
    assert numpy.isfinite(x).all()
    x = scaler.fit_transform(x)
    assert numpy.isfinite(x).all()
    x[:, 0:n_clusters] *= 100.0
    cluster_alg = MiniBatchKMeans(n_clusters=k, n_init=10)
    assignments = cluster_alg.fit_predict(x)
    print(assignments)
    with h5py.File(cluster_assignments_file) as f5:
        try:
            del f5['cluster_assignment']
        except:
            pass
        f5['cluster_assignment'] = assignments


@click.command()
@click.option('--i', default=0, type=int, help='which index')
def plot_ds(i):
    model = Model(in_channels=38, k=3)
    root_folder = '/data/spatial_zurich/data'
    ds = SpatialTranscriptomicsDs(root_folder=root_folder, k=model.k, training=False, divideable_by=model.divideable_by)

    image, mask, nn, targets = ds[i]

    app = pg.mkQApp()
    viewer = LayerViewerWidget()
    viewer.setWindowTitle('LayerViewer')
    viewer.show()
    image = numpy.moveaxis(image, 0, 2)
    image = numpy.swapaxes(image, 0, 1)
    mask = numpy.swapaxes(mask[0, ...], 0, 1)
    print('image', image.shape)
    layer = MultiChannelImageLayer(name='img', data=image)
    viewer.addLayer(layer=layer)

    labels = numpy.zeros(image.shape[0:2], dtype='uint8')
    label_layer = ObjectLayer(name='labels', data=None)
    viewer.addLayer(layer=label_layer)
    print('image', mask.shape)
    viewer.setData('labels', image=mask)
    QtGui.QApplication.instance().exec_()


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(embed)
cli.add_command(predict)
# cli.add_command(sandbox)
cli.add_command(plot_ds)
cli.add_command(cluster)
cli.add_command(visualize_clusters)
cli.add_command(refine_clusters)
cli.add_command(visualize_refined_clusters)

if __name__ == '__main__':
    cli()
    # predict.callback(root_folder, 1, False, 'out')
    # import click.testing
    # click.testing.CliRunner().invoke(train, [], catch_exceptions=False)
    # click.testing.CliRunner().invoke(predict, ['--n', '10'], catch_exceptions=False)
    # my_train(load=False, folder='derived_data')
# @click.option('--folder', default='log', type=str, help='load dir')
#
# @click.option('--n', default=10,type=int, help='compute the umap on n samples')
# @click.option('--show_umap', default=False,type=bool, help='shown the umap')
# @click.option('--out_dir', default='out',type=str, help='output folder')
