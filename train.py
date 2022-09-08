import logging
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from utils.dataset import SegDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet.model import UNet


DIR_CHECKPOINT = './checkpoints'


def train_net(bundle_path,
              markup_mapping,
              train_dataset_size,
              val_dataset_size,
              val_random_state: int = 0,
              epochs: int = 5,
              batch_size: int = 8,
              learning_rate: float = 0.0001,
              save_checkpoint: bool = True,
              amp: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Datasets
    dataset_args = dict(bundle_path=bundle_path, markup_mapping=markup_mapping, filter_id=13)
    train_dataset = SegDataset(group_name='train', **dataset_args)
    val_dataset = SegDataset(group_name='val', **dataset_args)

    # Samplers
    train_sampler = WeightedRandomSampler(weights=train_dataset.weights,
                                          num_samples=train_dataset_size)

    val_sampler = WeightedRandomSampler(weights=val_dataset.weights,
                                        num_samples=val_dataset_size,
                                        replacement=False,
                                        generator=torch.Generator().manual_seed(val_random_state))
    frozen_val_set = list(val_sampler)

    # Data Loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_dataset, sampler=frozen_val_set, **loader_args)

    # Net init
    if train_dataset.n_classes > 1:
        raise NotImplementedError
    net = UNet(n_channels=3, n_classes=train_dataset.n_classes)
    net.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {train_dataset_size}
        Validation size: {val_dataset_size}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=train_dataset_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device)
                true_masks = true_masks.to(device=device)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(torch.sigmoid(masks_pred),
                                       true_masks, multiclass=False)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            val_score = evaluate(net, val_loader, device)
            scheduler.step(val_score)

            logging.info('Validation Dice score: {}'.format(val_score))

        if save_checkpoint:
            os.makedirs(DIR_CHECKPOINT, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(DIR_CHECKPOINT, f'checkpoint_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved!')


if __name__ == '__main__':
    import conf
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    train_net(bundle_path=conf.bundle_path,
              markup_mapping=conf.markup_mapping,
              train_dataset_size=conf.train_dataset_size,
              val_dataset_size=conf.val_dataset_size,
              val_random_state=conf.val_random_state,
              epochs=conf.epochs,
              batch_size=conf.batch_size,
              learning_rate=conf.learning_rate,
              save_checkpoint=conf.save_checkpoint,
              amp=conf.amp)
