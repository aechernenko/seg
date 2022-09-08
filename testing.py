import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from utils.dataset import SegDataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from unet.model import UNet
from evaluate import evaluate


class TestModel:

    def __init__(self, model_path, dataset: SegDataset):

        if dataset.n_classes > 1:
            raise NotImplementedError

        self.net = UNet(n_channels=3, n_classes=dataset.n_classes)
        self.dataset = dataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f'Loading model {model_path}')
        logging.info(f'Using device {self.device}')

        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))

        logging.info('Model loaded!')

        self.net.eval()

    def predict_by_idx(self, idx, threshold=None):
        sample = self.dataset[idx]
        img = sample['image']
        mask = sample['mask']
        np_img = img.permute(1, 2, 0).numpy()
        np_mask = mask.permute(1, 2, 0).numpy()

        with torch.no_grad():
            output = self.net(torch.unsqueeze(img, 0).to(self.device))

            if self.net.n_classes == 1:
                probs = torch.sigmoid(output)[0]
            else:
                raise NotImplementedError

            pred_mask = probs.cpu().permute(1, 2, 0).numpy()
            if threshold is not None:
                pred_mask = pred_mask > threshold

        return np_img, np_mask, pred_mask

    def show_predict_by_idx(self, idx, threshold=None):
        img, markup, predict = self.predict_by_idx(idx, threshold)

        plt.title('Image')
        plt.imshow(img/img.max())
        plt.show()

        plt.title('Markup')
        plt.imshow(markup)
        plt.show()

        plt.title(f'Predict. Threshold: {threshold}')
        plt.imshow(predict)
        plt.show()

    def evaluate(self, n_samples, batch_size=8, random_state=0):
        sampler = WeightedRandomSampler(weights=self.dataset.weights,
                                        num_samples=n_samples, replacement=False,
                                        generator=torch.Generator().manual_seed(random_state))
        frozen_set = list(sampler)
        loader = DataLoader(self.dataset, sampler=frozen_set,
                            batch_size=batch_size, num_workers=4,
                            pin_memory=True)
        score = evaluate(self.net, loader, self.device)
        return score.cpu().item()


if __name__ == '__main__':
    model_path = '/net/ml4/home/ac/git/seg/checkpoints/checkpoint_epoch48.pth'

    bundle_path = './data/c_bundle_hs_128_256.h5'

    markup_mapping = {
        'human_step': 'human__step',
        'human__step': 'human__step',
    }

    test_dataset = SegDataset(group_name='test', bundle_path=bundle_path,
                              markup_mapping=markup_mapping, filter_id=13)

    test_model = TestModel(model_path, test_dataset)
