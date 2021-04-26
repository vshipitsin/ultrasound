from tqdm import tqdm
import torch


def run_epoch(model, iterator,
              criterion, optimizer,
              metric,
              phase='train', epoch=0,
              device='cpu', writer=None):
    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_metric = 0.0

    with torch.set_grad_enabled(is_train):
        for (images, targets) in tqdm(iterator, desc=f"{phase}", ascii=True):
            images, targets = images.to(device), targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_metric += metric(predictions.detach(), targets)

        if writer is not None:
            writer.add_scalar(f"loss_epoch/{phase}", epoch_loss / len(iterator), epoch)
            writer.add_scalar(f"metric_epoch/{phase}", epoch_metric / len(iterator), epoch)

        return epoch_loss / len(iterator), epoch_metric / len(iterator)


def train(model,
          train_dataloader, val_dataloader,
          criterion,
          optimizer, scheduler,
          metric,
          n_epochs,
          device,
          writer):
    best_val_loss = float('+inf')
    for epoch in range(n_epochs):
        train_loss, train_metric = run_epoch(model, train_dataloader,
                                             criterion, optimizer,
                                             metric,
                                             phase='train', epoch=epoch,
                                             device=device, writer=writer)
        val_loss, val_metric = run_epoch(model, val_dataloader,
                                         criterion, None,
                                         metric,
                                         phase='val', epoch=epoch,
                                         device=device, writer=writer)
        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{model.name}.best.pth")

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.2f} | Train Metric: {train_metric:.2f}')
        print(f'\t  Val Loss: {val_loss:.2f} |   Val Metric: {val_metric:.2f}')

    if writer is not None:
        writer.close()
