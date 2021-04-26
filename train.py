import argparse
import os

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from utils import get_params

from data.transforms import ToTensor, Resize
from data.dataset import SegmentationDataset, data_loaders

from models.segmentation import UNet

from training.utils import make_reproducible, print_model_info
from training.model_training import train
from training.losses import CombinedLoss, CrossEntropyLoss, MultilabelDiceLoss
from training.metrics import DiceMetric

# for printing
torch.set_printoptions(precision=2)

# for reproducibility
make_reproducible(seed=0)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def main(args):
    params = get_params()

    transform = torchvision.transforms.Compose([Resize(size=params['image_size']), ToTensor()])
    train_dataloader, val_dataloader = data_loaders(dataset=SegmentationDataset,
                                                    transform=transform,
                                                    params=params)

    model = UNet(n_channels=1, n_classes=2,
                 init_features=params['init_features'], depth=params['depth'],
                 image_size=params['image_size'], adaptive_layer_type=params['adaptive_layer']).to(device)
    print_model_info(model, params)

    criterion = CombinedLoss([CrossEntropyLoss(), MultilabelDiceLoss()], [0.4, 0.6])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    metric = DiceMetric()

    writer = None
    if not args.nolog:
        writer = SummaryWriter(log_dir=os.path.join(params['log_dir'], model.name))
        print("To see the learning process, use command in the new terminal:\n" +
              "tensorboard --logdir <path to log directory>")
        print()

    train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metric,
          n_epochs=params['n_epochs'],
          device=device,
          writer=writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nolog', action='store_true', help='turn off logging')
    args = parser.parse_args()

    main(args)
