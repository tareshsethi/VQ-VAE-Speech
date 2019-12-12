from torch.utils.data import Dataset
import pickle
import os
import numpy as np


class IBMFeaturesDataset(Dataset):

    def __init__(self, ibm_path, subdirectory, normalizer=None, features_path='features'):
        self._ibm_path = ibm_path
        self._subdirectory = subdirectory
        features_path = self._ibm_path + os.sep + features_path
        self._sub_features_path = features_path + os.sep + self._subdirectory
        self._files_number = len(os.listdir(self._sub_features_path))
        self._normalizer = normalizer

    def __getitem__(self, index):
        dic = None
        path = self._sub_features_path + os.sep + str(index) + '.pickle'

        if index == 9:
            path = self._sub_features_path + os.sep + str(8) + '.pickle'


        if not os.path.isfile(path):
            raise OSError("No such file '{}'".format(path))

        if os.path.getsize(path) == 0:
            raise OSError("Empty file '{}'".format(path))

        with open(path, 'rb') as file:
            dic = pickle.load(file)

        if self._normalizer:
            dic['input_features'] = (dic['input_features'] - self._normalizer['train_mean']) / self._normalizer['train_std']
            dic['output_features'] = (dic['output_features'] - self._normalizer['train_mean']) / self._normalizer['train_std']

        dic['quantized'] = np.array([]) if dic['quantized'] is None else dic['quantized']
        dic['one_hot'] = np.array([]) if dic['one_hot'] is None else dic['one_hot']
        dic['index'] = index

        return dic

    def __len__(self):
        return self._files_number
