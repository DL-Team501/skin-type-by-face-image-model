import torch
from torch import nn, utils, optim, softmax, argmax
from torchvision.models import ViT_B_16_Weights, vit_b_16
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from src.data.dataset import create_dataloaders
from src.training_config import training_device
from src.utils import save_checkpoint, loss_metric_curve_plot


def create_model():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    for param in model.conv_proj.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.heads = nn.Sequential(
        OrderedDict([('head', nn.Linear(in_features=768, out_features=8))])
    )

    return model


def train_step(model: nn.Module,
               dataloader: utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: optim.Optimizer):
    model.train()

    train_loss = 0.
    train_accuracy = 0.

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred_logit = model(X)
        loss = loss_fn(y_pred_logit, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        y_pred_prob = softmax(y_pred_logit, dim=1)
        y_pred_class = argmax(y_pred_prob, dim=1)
        train_accuracy += accuracy_score(y.to(training_device).numpy(),
                                         y_pred_class.detach().to(training_device).numpy())

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)

    return train_loss, train_accuracy


def valid_step(model: nn.Module,
               dataloader: utils.data.DataLoader,
               loss_fn: nn.Module):
    model.eval()

    valid_loss = 0.
    valid_accuracy = 0.

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred_logit = model(X)
            loss = loss_fn(y_pred_logit, y)
            valid_loss += loss.item()

            y_pred_prob = softmax(y_pred_logit, dim=1)
            y_pred_class = argmax(y_pred_prob, dim=1)

            valid_accuracy += accuracy_score(y.to(training_device).numpy(),
                                             y_pred_class.detach().to(training_device).numpy())

    valid_loss = valid_loss / len(dataloader)
    valid_accuracy = valid_accuracy / len(dataloader)

    return valid_loss, valid_accuracy


def train_model(model: nn.Module,
                train_dataloader: utils.data.DataLoader,
                valid_dataloader: utils.data.DataLoader,
                loss_fn: nn.Module,
                optimizer: optim.Optimizer,
                epochs: int = 10):
    results = {"train_loss": [],
               "train_accuracy": [],
               "valid_loss": [],
               "valid_accuracy": []}

    best_valid_loss = float("inf")

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer)

        valid_loss, valid_accuracy = valid_step(model=model,
                                                dataloader=valid_dataloader,
                                                loss_fn=loss_fn)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            file_name = "best_model.pth"
            save_checkpoint(file_name, model, best_valid_loss, epoch, optimizer, valid_accuracy)

        print(f"Epoch: {epoch + 1} | ",
              f"Train Loss: {train_loss:.4f} | ",
              f"Train Accuracy: {train_accuracy:.4f} | ",
              f"Valid Loss: {valid_loss:.4f} | ",
              f"Valid Accuracy: {valid_accuracy:.4f}")

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["valid_loss"].append(valid_loss)
        results["valid_accuracy"].append(valid_accuracy)

    return results


# **Main Execution**
if __name__ == "__main__":
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders()
    model = create_model()

    SEED = 123
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)

    MODEL_RESULTS = train_model(model.to(device),
                                train_dataloader,
                                valid_dataloader,
                                nn.CrossEntropyLoss(),
                                optim.Adam(model.parameters(), lr=0.01),
                                15)

    loss_metric_curve_plot(MODEL_RESULTS)
    checkpoint = torch.load("best_model.pth")
    print(f'Best Loss: {checkpoint["loss"]}')
    print(f'Epoch: {checkpoint["epoch"] + 1}')
    print(f'Best Metric: {checkpoint["metric"]}')