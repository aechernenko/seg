bundle_path = './data/c_bundle_hs_128_256.h5'
dir_checkpoint = './checkpoints/'

markup_mapping = {
    'human_step': 'human__step',
    'human__step': 'human__step',
}

if len(set(markup_mapping.values())) > 1:
    raise NotImplementedError('implemented support for single target class only')

train_dataset_size = 1024  # size of training set that is resampled for each epoch
val_dataset_size = 1024  # size of frozen val set for all epochs
val_random_state = 0
epochs = 50
batch_size = 16
learning_rate = 1e-4

save_checkpoint = True
amp = True
