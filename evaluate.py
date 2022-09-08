import torch
from tqdm import tqdm

from utils.dice_score import dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with tqdm(total=num_val_batches, desc=f'Validation round', unit='batch', position=0, leave=False) as pbar:
        for batch in dataloader:
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = mask_true.float()

            with torch.no_grad():
                mask_pred = net(image)
                if net.n_classes == 1:
                    mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    raise NotImplementedError
            pbar.update(1)

    net.train()
    return dice_score / num_val_batches
