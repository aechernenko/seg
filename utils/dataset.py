from typing import Dict

import torch
import os
import h5py
import numpy as np
from torch.utils.data import Dataset

from dsp.filter.standard import filters
from dr_library.record.reader.hdf_work import HDFWorkRecordReader


class SegDataset(Dataset):

    def __init__(self,
                 bundle_path: str,
                 group_name: str,
                 markup_mapping: Dict,
                 filter_id: int = 0):

        if not os.path.isfile(bundle_path):
            raise FileNotFoundError(bundle_path)

        self._filter = filters[filter_id]

        # load data to memory
        with h5py.File(bundle_path, 'r') as h5:

            markup_classes = h5.attrs['class_map'].split(',')
            self._sample_height = h5.attrs['sample_height']
            self._sample_width = h5.attrs['sample_width']
            self._t_thinning = h5.attrs['t_thinning']
            self._l_thinning = h5.attrs['l_thinning']

            if sorted(markup_classes) != sorted(markup_mapping.keys()):
                raise Exception(f"Bundle classes and meta classes must not differs")

            if group_name in h5:
                group = h5[group_name]
                self._coords = group['coords'][:]
                self._source_labels = group['labels'][:]
                self._records = group['records'][:]
            else:
                raise AttributeError(f"{group_name} not found in {bundle_path}")

            self._records_map = h5['records_map'][:].astype(str)

        self._markup_maping = markup_mapping
        self._target_classes = sorted(set(self._markup_maping.values()))
        self._trainable_target_classes = [tc for tc in self._target_classes if not tc.startswith('z_')]

        # weighting samples
        self._weights = np.empty(shape=(len(self),), dtype=float)
        target_prob = 1 / len(self._target_classes)
        for tc in self._target_classes:
            markup_group = [m for m, t in self._markup_maping.items() if t == tc]
            markup_group_prob = 1 / len(markup_group)
            for sl, mc in enumerate(markup_classes):
                if mc in markup_group:
                    indices = np.argwhere(self._source_labels == sl).flatten()
                    if indices.size > 0:
                        markup_item_prob = 1 / indices.size
                        self._weights[indices] = markup_item_prob * markup_group_prob * target_prob

    @property
    def sample_size(self):
        """Size of sample after decimation"""
        return self._sample_height // self._t_thinning, self._sample_width // self._l_thinning

    @property
    def weights(self):
        return self._weights

    @property
    def sample_shape(self):
        return 3, *self.sample_size

    @property
    def mask_shape(self):
        return self.n_classes, *self.sample_size

    @property
    def target_classes(self):
        return self._trainable_target_classes

    @property
    def n_classes(self):
        return len(self.target_classes)

    def __getitem__(self, index):
        record_path_idx = self._records[index]
        record_path = self._records_map[record_path_idx, 0]
        reader = HDFWorkRecordReader(record_path)
        # assert markup_class in reader.markup_hand.class_ids_with_objects

        row, column = self._coords[index]

        half_sample_height = self._sample_height // 2
        slice_row_from = row - half_sample_height
        slice_row_to = row + half_sample_height

        half_sample_width = self._sample_width // 2
        slice_column_from = column - half_sample_width
        slice_column_to = column + half_sample_width

        row_slice = np.s_[slice_row_from: slice_row_to: self._t_thinning]
        column_slice = np.s_[slice_column_from: slice_column_to: self._l_thinning]
        sample_slice = np.s_[row_slice, column_slice]

        # нарезаем сэмпл
        sample_data = reader.filter.get_data(sample_slice)

        if self._filter is not None:
            sample_data = self._filter(sample_data)

        sample_data = sample_data.transpose((2, 0, 1)).astype(float)
        assert sample_data.shape == self.sample_shape

        # нарезаем маски по целевым обучаемым классам
        mask_data = np.full(shape=self.mask_shape, fill_value=False)

        for class_id in reader.markup_hand.class_ids_with_objects:
            if class_id in self._markup_maping:
                target_class = self._markup_maping[class_id]
                if target_class in self._trainable_target_classes:
                    label = self._trainable_target_classes.index(target_class)
                    mask = reader.markup_hand.get_combine_mask(class_id, np_s=sample_slice)
                    mask_data[label] = np.logical_or(mask_data[label], mask)

        return {
            'image': torch.tensor(sample_data, dtype=torch.float32),
            'mask': torch.tensor(mask_data, dtype=torch.float32),
        }

    def __len__(self):
        return self._coords.shape[0]
