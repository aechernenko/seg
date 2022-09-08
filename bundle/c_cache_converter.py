import os
from glob import glob
from typing import Tuple, List, Union

import h5py
import numpy as np


def convert_c_cache(c_cache: str,
                    bundle_path: str,
                    train_records: Union[List, Tuple],
                    validation_records: Union[List, Tuple],
                    test_records: Union[List, Tuple],
                    sample_height: int,
                    sample_width: int,
                    t_thinning: int,
                    l_thinning: int,
                    classes: List[str] = None,
                    chunk_size: int = 128):

    c_cache = os.path.abspath(c_cache)
    if not os.path.isdir(c_cache):
        raise NotADirectoryError(c_cache)

    if os.path.exists(bundle_path):
        raise FileExistsError(bundle_path)

    if classes is None:
        class_dirs = glob(os.path.join(c_cache, '*'))
    else:
        class_dirs = [os.path.join(c_cache, cl) for cl in classes]
        for class_dir in class_dirs:
            if not os.path.isdir(class_dir):
                raise NotADirectoryError(c_cache)

    train, val, test = dict(), dict(), dict()
    for group in (train, val, test):
        for key in ('coords', 'records', 'labels'):
            group[key] = list()

    record_id = 0
    class_ids = list()
    records_map = list()
    _rec2id = dict()

    for label, class_dir in enumerate(class_dirs):

        class_ids.append(os.path.basename(class_dir))

        hash_ = glob(os.path.join(class_dir, '*'))
        if len(hash_) != 1:
            raise Exception(f"Unique hash not found in {class_dir}")
        class_hash_dir = os.path.join(class_dir, hash_[0])

        npz_paths = glob(os.path.join(class_hash_dir, '**', '*.npz'), recursive=True)

        for npz in npz_paths:

            record = '/' + os.path.relpath(npz, class_hash_dir)[:-3] + 'h5'
            assert os.path.isfile(record)
            if record in train_records:
                group = train
            elif record in validation_records:
                group = val
            elif record in test_records:
                group = test
            else:
                continue

            coord_array = np.load(npz, allow_pickle=True)['coords']
            if coord_array.shape:
                for coord in coord_array:
                    group['coords'].append(coord)
                    group['labels'].append(label)
                    if record in _rec2id:
                        group['records'].append(_rec2id[record])
                    else:
                        _rec2id[record] = record_id
                        records_map.append(record)
                        group['records'].append(record_id)
                        record_id += 1

    with h5py.File(bundle_path, 'w') as h5:

        h5.attrs['class_map'] = ','.join(class_ids)
        h5.attrs['sample_height'] = sample_height
        h5.attrs['sample_width'] = sample_width
        h5.attrs['t_thinning'] = t_thinning
        h5.attrs['l_thinning'] = l_thinning


        chunk_rm_shape = (chunk_size, 1)
        rm_shape = (len(records_map), 1)
        dtype = h5py.special_dtype(vlen=str)
        h5.create_dataset('records_map', shape=rm_shape, dtype=dtype, chunks=chunk_rm_shape)
        h5['records_map'][0:] = np.vstack(records_map)

        for group, group_name in zip((train, val, test), ('train', 'val', 'test')):

            assert len(group['labels']) == len(group['records']) == len(group['coords'])
            group_size = len(group['labels'])
            indices = np.arange(group_size)
            np.random.shuffle(indices)

            gr = h5.create_group(group_name)

            gr.create_dataset('coords', data=np.vstack(group['coords']),
                              dtype='uint32', chunks=(chunk_size, 2))

            gr.create_dataset('records', shape=(group_size, ),
                              dtype='uint16', chunks=(chunk_size, ))
            gr['records'][0:] = np.hstack(group['records'])

            gr.create_dataset('labels', shape=(group_size, ),
                              dtype='uint8', chunks=(chunk_size, ))
            gr['labels'][0:] = np.hstack(group['labels'])


if __name__ == '__main__':
    from records import train_records, validation_records, test_records
    from lim_seg_bundle import rules

    convert_c_cache(c_cache='../bundle/c_cache',
                    bundle_path='../data/c_bundle_hs_128_256.h5',
                    train_records=train_records,
                    validation_records=validation_records,
                    test_records=test_records,
                    sample_height=rules['size_t'],
                    sample_width=rules['size_l'],
                    t_thinning=rules['t_thinning'],
                    l_thinning=rules['l_thinning'],
                    classes=None)
