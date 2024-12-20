"""Training loop for MRI classification CNN models."""
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from torch import nn
from tqdm import tqdm


class SaveBestModelLoss:
    """
    Save the best model while training.

    If the current epoch's validation loss is less than the previous least loss, then save the model state.
    """
    def __init__(self, save_path: str | Path):
        self.best_valid_loss = float('inf')
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, current_valid_loss: float, epoch: int, model: nn.Module) -> None:
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), str(Path(self.save_path, f'best_model.pth')))


class SaveBestModelAccuracy:
    """
    Save the best model while training.

    If the current epoch's validation accuracy is greater than the previous greatest accuracy,
    then save the model state.
    """
    def __init__(self, save_path: str | Path):
        self.best_valid_acc = 0
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, current_valid_acc: float, epoch: int, model: nn.Module) -> None:
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            print(f"\nBest validation accuracy: {self.best_valid_acc}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), str(Path(self.save_path, f'best_model.pth')))


def run_training_loop(model: nn.Module, train_dataset: torch.utils.data.Dataset,
                      valid_dataset: torch.utils.data.Dataset, num_epochs: int, optimizer: torch.optim.Optimizer,
                      scheduler: torch.optim.lr_scheduler, save_directory: str | Path,
                      batch_size: int, from_pretrained: bool = False) -> dict[str, list[float]]:
    """
    Train model.

    :param model: Model to train.
    :param train_dataset: Dataloader for train subset.
    :param valid_dataset: Dataloader for validation subset.
    :param num_epochs: Maximum number of epochs to train for.
    :param optimizer: Optimizer.
    :param scheduler: Learning rate scheduler.
    :param batch_size: Batch size.
    :param save_directory: Directory to save best model to.
    :param from_pretrained: If True, freeze all layers except the last one at the beginning of training. After one
    epoch, unfreeze all the layers.
    :return: History of training.
    """
    # Set device dynamically to either GPU (if available) or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_balanced_accuracy": [],
        "valid_balanced_accuracy": []
    }

    save_best = SaveBestModelAccuracy(save_directory)
    if from_pretrained:
        model.freeze()

    model.to(device)  # Ensure model is on the correct device

    for epoch in range(num_epochs):
        if from_pretrained and epoch == 1:
            model.unfreeze()

        train_dataset.shuffle()
        valid_dataset.shuffle()

        model.train()
        train_acc, val_acc, balanced_accuracy, running_loss = 0.0, 0.0, 0.0, 0.0

        prbar = tqdm(train_dataset, leave=True)
        step = 1
        results, ground_truths = [], []

        for data in prbar:
            criterion = torch.nn.CrossEntropyLoss()
            inputs, labels = data

            # Move inputs and labels to the correct device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output_classifier = model(inputs)
            labels = torch.argmax(labels, dim=-1)

            loss = criterion(output_classifier, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probabilities = torch.argmax(torch.softmax(output_classifier, dim=-1), dim=-1)
            step_score = accuracy_score(labels.cpu().flatten().numpy(), probabilities.cpu().flatten().numpy(),
                                        normalize=False)
            train_acc += step_score
            results.extend(probabilities.cpu().flatten().numpy())
            ground_truths.extend(labels.cpu().flatten().numpy())
            balanced_accuracy = balanced_accuracy_score(ground_truths, results, adjusted=False)
            prbar.set_description(
                f"Epoch: {epoch}, LR: {scheduler.get_last_lr()} Step loss: {round(loss.item(), 3)}, "
                f"Step acc: {round(step_score, 3)}, "
                f"Train Accuracy: {round(train_acc / (step * batch_size), 3)}, "
                f"Balanced Accuracy: {round(balanced_accuracy, 3)}, "
                f"Running Loss: {round(running_loss / step, 3)}"
            )
            step += 1

        history["train_loss"].append(running_loss / len(train_dataset))
        history["train_balanced_accuracy"].append(balanced_accuracy)

        # Validation loop
        model.eval()
        results, ground_truths = [], []
        len_valid_dataset = len(valid_dataset)
        with torch.no_grad():
            prbar = tqdm(valid_dataset, leave=True)
            loss_running_valid = 0
            step = 0

            for data in prbar:
                criterion = torch.nn.CrossEntropyLoss()
                inputs, labels = data

                # Move inputs and labels to the correct device
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                labels = torch.argmax(labels, dim=-1)
                loss = criterion(outputs, labels)

                probabilities = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                step_score = accuracy_score(labels.cpu().flatten().numpy(), probabilities.cpu().flatten().numpy(),
                                            normalize=False)
                val_acc += step_score
                loss_running_valid += loss.item()
                step += 1

                results.extend(probabilities.cpu().flatten().numpy())
                ground_truths.extend(labels.cpu().flatten().numpy())

                balanced_accuracy = balanced_accuracy_score(ground_truths, results, adjusted=False)
                prbar.set_description(
                    f"Validation Loss - item: {round(loss.item(), 3)}, "
                    f"Validation Accuracy: {round(val_acc / (step * batch_size), 3)}, "
                    f"Balanced accuracy: {round(balanced_accuracy, 3)}, "
                    f"Running Loss: {round(loss_running_valid / step, 3)}"
                )

        scheduler.step()
        save_best(balanced_accuracy, epoch, model)
        history["valid_loss"].append(loss_running_valid / len_valid_dataset)
        history["valid_balanced_accuracy"].append(balanced_accuracy)

    return history


def evaluate_model(model: nn.Module, dataloader: torch.utils.data.Dataset) -> dict[str, float]:
    """
    Evaluate model on given dataset. Return ROC AUC, accuracy and balanced accuracy.

    :param model: Trained model.
    :param dataloader: Dataloader object
    :return: Dictionary with metric name as key and metric value as value.
    """
    # Set device dynamically to either GPU (if available) or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = {}
    predictions, ground_truths = [], []
    model.to(device)
    model.eval()

    with torch.no_grad():
        prbar = tqdm(dataloader, leave=True)

        for data in prbar:
            inputs, labels = data

            # Move inputs and labels to the correct device
            inputs, labels = inputs.to(device), labels.to(device)

            labels = torch.argmax(labels, dim=-1)
            outputs = model(inputs)
            probabilities = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)

            predictions.extend(probabilities.cpu().flatten().numpy())
            ground_truths.extend(labels.cpu().flatten().numpy())

    metrics["accuracy"] = accuracy_score(ground_truths, predictions)
    metrics["balanced_accuracy"] = balanced_accuracy_score(ground_truths, predictions, adjusted=False)

    if len(np.unique(ground_truths)) == 2:
        metrics["roc_auc"] = roc_auc_score(ground_truths, predictions)
        metrics["roc_curve"] = roc_curve(ground_truths, predictions)

    metrics["confusion_matrix"] = confusion_matrix(ground_truths, predictions)

    return metrics