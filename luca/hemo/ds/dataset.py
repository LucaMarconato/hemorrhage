import os

import h5py
import numpy as np
import pandas as pd
import skimage.io
import torch
import torch.utils.data as data
from tqdm import tqdm
import torch.nn.functional as F


# current_folder = os.path.dirname(os.path.abspath(__file__))
# mem = joblib.Memory(current_folder, verbose=1)
# os.makedirs('joblib', exist_ok=True)


class HemoDataset(data.Dataset):
    def __init__(self, root_folder, training, mean=None, var=None, image_transform=None):
        super().__init__()
        self.training = training

        self.mean = mean
        self.var = var

        training_folder = os.path.join(root_folder, 'hemo18gb/train')
        self.training_file_paths = os.listdir(training_folder)
        self.training_file_paths = [os.path.join(training_folder, f) for f in self.training_file_paths]
        testing_folder = os.path.join(root_folder, 'hemo18gb/test')
        self.testing_file_paths = os.listdir(training_folder)
        self.testing_file_paths = [os.path.join(testing_folder, f) for f in self.testing_file_paths]

        self.f5_path = os.path.join(root_folder, 'training_targets.h5')

        def create_targets_h5():
            df = pd.read_csv(os.path.join(root_folder, 'stage_1_train.csv'))
            targets = dict()
            print('preprocessing the .csv files')
            with tqdm(total=len(df)) as bar:
                for i, row in enumerate(df.iterrows()):
                    filename_and_hemo_class = row[1][0]
                    hemo_class_target = row[1][1]
                    pos = filename_and_hemo_class.rfind('_')
                    filename = filename_and_hemo_class[:pos]
                    if i % 6 == 0:
                        current_filename = filename
                        current_target = []
                    if current_filename == filename:
                        current_target.append(hemo_class_target)
                    if i % 6 == 5:
                        targets[filename] = np.array(current_target)
                    bar.update(1)
                    # if i == 6000:
                    #     break
            print('saving the preprocessed data into th .h5 format')
            with h5py.File(self.f5_path, 'w') as f5:
                with tqdm(total=len(targets.items())) as bar:
                    for k, v in targets.items():
                        f5[k] = v
                        bar.update(1)

        if not os.path.isfile(self.f5_path):
            create_targets_h5()

        self.image_transform = image_transform

    def __len__(self):
        if self.training:
            return len(self.training_file_paths)
        else:
            return len(self.testing_file_paths)

    def __getitem__(self, i):
        if self.training:
            image_file_paths = self.training_file_paths
        else:
            image_file_paths = self.testing_file_paths
        image = skimage.io.imread(image_file_paths[i])
        image = np.require(image, dtype='float32')

        if self.mean is not None:
            image -= self.mean
            image /= np.sqrt(self.var)

        if self.image_transform is not None:
            image = self.image_transform(image)
        image = np.asarray(image)
        # removes the negative stride (to test check image.strides before and after)
        image = np.sort(image)[::-1].copy()
        if image.shape != (512, 512):
            # print(f'resizing {image_file_paths[i]}')
            import cv2
            image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            # image = F.upsample(torch.from_numpy(image).unsqueeze(dim=0), size=(512, 512), mode='bilinear').numpy()
        image = image[np.newaxis, :, :]
        if self.training:
            image_filename = os.path.basename(image_file_paths[i]).replace('.jpg', '')
            with h5py.File(self.f5_path, 'r') as f5:
                targets = f5[image_filename][...].copy()
            return image, targets
        else:
            return image


if __name__ == '__main__':
    root_folder = '/data/hemo'
    ds = HemoDataset(root_folder=root_folder, training=True)
    print(ds[1])

    # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    #
    # for x in ds:
    #     image, targets = x
    #     v = image.reshape([image.shape[0], -1])
    #     scaler.partial_fit(v)
    #
    # print(scaler.mean_)
    # print(scaler.var_)
