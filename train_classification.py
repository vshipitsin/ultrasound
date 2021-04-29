import os

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from utils import get_params, parse_args

from data.dataset import ClassificationDataset, data_loaders

from models.classification import ResNet

from training.utils import make_reproducible, print_model_info
from training.model_training import train
from training.metrics import F1Score

# for printing
torch.set_printoptions(precision=2)

# for reproducibility
make_reproducible(seed=0)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def main(args):
    params = get_params()

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=params['image_size']),
                                                torchvision.transforms.ToTensor()])
    train_dataloader, val_dataloader = data_loaders(dataset=ClassificationDataset,
                                                    transform=transform,
                                                    params=params)

    model = ResNet(n_channels=1, n_classes=params['n_classes'],
                   blocks=params['blocks'], filters=params['filters'],
                   image_size=params['image_size'], adaptive_layer_type=params['adaptive_layer']).to(device)
    print_model_info(model, params)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    metric = F1Score()

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
    main(parse_args())
