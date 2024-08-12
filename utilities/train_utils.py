import torch
nn = torch.nn
from typing import Iterable, List
import time
import numpy as np
from sklearn.metrics import classification_report, f1_score
import copy
import logging
import torch.optim


def train_cls(
        train_data: Iterable,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        epochs: int,
        device: str,
        classes: List[str],
        val_data: Iterable = None,
        patience: int = None) -> torch.nn.Module:
    """Train a given network no an arbitrary classification task

    Args:
        train_data (Iterable): Data to train on
        model (torch.nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer to use
        criterion (): Criterion to calculate loss for
        epochs (int): Epochs to train for
        device (str): Device to load data to
        classes (List[str]): List of classes in correct order
        val_data (Iterable, optional): Validation data. Defaults to None.

    Returns:
        torch.nn.Module: Trained model
    """

    best_model = model
    best_score = 0
    current_patience = patience
    
    if val_data is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience//2)

    for e in range(epochs):

        running_loss = 0
        epoch_time = 0
        start_time_prev = time.time()
        for idx, (i, y_true) in enumerate(train_data):
            start_time = time.time()

            i = i.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)

            optimizer.zero_grad()

            y_pred = model(i)

            loss = criterion(y_pred, y_true)
            loss.backward()

            optimizer.step()

            end_time_0 = time.time()

            running_loss += loss.item()

            iter_time = start_time - start_time_prev

            epoch_time += iter_time

            print("Epoch: {:>4} / {:>4} Batch: {:>4} / {:>4} Loss: {:.4f}  ({:.4f}s) ({:.4f}s)     ".format(
                e + 1,
                epochs,
                idx + 1,
                len(train_data),
                running_loss / (idx + 1),
                (end_time_0 - start_time) / y_pred.shape[0],
                iter_time / y_pred.shape[0]
            ), end="\r")

            start_time_prev = start_time

        print("Epoch: {:>4} / {:>4} Batch: {:>4} / {:>4} Loss: {:.4f} ({:.4f}s)                                  ".format(
            e + 1, epochs, idx + 1, len(train_data), running_loss / (idx + 1), epoch_time))

        if val_data is None:
            continue

        model.eval()

        y_pred_list = []
        y_true_list = []

        for i, y_true in val_data:
            i = i.to(device)
            y_true = y_true.to(device)

            y_pred = model(i)

            y_pred_list.extend(
                np.argmax(y_pred.cpu().detach().numpy(), axis=-1))
            y_true_list.extend(y_true.cpu().detach().numpy())

        logging.debug("\n{}".format(classification_report(y_true_list,
                                                          y_pred_list, target_names=classes)))

        cur_score = f1_score(y_true_list, y_pred_list, average="micro", zero_division=0)

        if cur_score > best_score:
            best_model = copy.deepcopy(model)
            best_score = cur_score
            current_patience = patience
        else:
            current_patience -= 1
            if current_patience < 1:
                logging.info("No more patience left")
                model.train()
                break

        logging.info("Best score: {:.3f}   Current score: {:.3f}   Patience: {}".format(
            best_score,
            cur_score,
            current_patience
        ))
        
        scheduler.step(cur_score)

        model.train()

    return best_model


def train_keypoint(
        train_data: Iterable,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        epochs: int,
        device: str,
        classes: List[str] = None,
        val_data: Iterable = None,
        patience: int = None) -> torch.nn.Module:
    """Train a given keypoint model

    Args:
        train_data (Iterable): Data to train on
        model (torch.nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer to use
        criterion (): Criterion to calculate loss for
        epochs (int): Epochs to train for
        device (str): Device to load data to
        classes (List[str]): List of classes in correct order
        val_data (Iterable, optional): Validation data. Defaults to None.

    Returns:
        torch.nn.Module: Trained model
    """

    best_model = model
    best_score = np.finfo(float).max
    current_patience = patience

    for e in range(epochs):

        running_loss = 0
        epoch_time = 0
        start_time_prev = time.time()
        for idx, (i, y_true_prob, y_true_coord) in enumerate(train_data):
            start_time = time.time()

            i = i.to(device, non_blocking=True)
            y_true_prob = y_true_prob.to(device, non_blocking=True)
            y_true_coord = y_true_coord.to(device, non_blocking=True)

            optimizer.zero_grad()

            y_pred_prob, y_pred_coord = model(i)

            loss = criterion(y_pred_prob, y_pred_coord, y_true_prob, y_true_coord)
            loss.backward()

            optimizer.step()

            end_time_0 = time.time()

            running_loss += loss.item()

            iter_time = start_time - start_time_prev

            epoch_time += iter_time

            print("Epoch: {:>4} / {:>4} Batch: {:>4} / {:>4} Loss: {:.4f}  ({:.4f}s) ({:.4f}s)     ".format(
                e + 1,
                epochs,
                idx + 1,
                len(train_data),
                running_loss / (idx + 1),
                (end_time_0 - start_time) / y_pred_prob.shape[0],
                iter_time / y_pred_prob.shape[0]
            ), end="\r")

            start_time_prev = start_time

        print("Epoch: {:>4} / {:>4} Batch: {:>4} / {:>4} Loss: {:.4f} ({:.4f}s)                                  ".format(
            e + 1, epochs, idx + 1, len(train_data), running_loss / (idx + 1), epoch_time))

        if val_data is None:
            continue

        model.eval()

        y_pred_prob_list = []
        y_true_prob_list = []
        y_pred_coord_list = []
        y_true_coord_list = []
        val_loss = 0.0
        num_samples = 0

        for i, y_true_prob, y_true_coord in val_data:
            num_samples += i.shape[0]
            i = i.to(device)
            y_true_prob = y_true_prob.to(device)
            y_true_coord = y_true_coord.to(device)

            y_pred_prob, y_pred_coord = model(i)

            val_loss += criterion(y_pred_prob, y_pred_coord, y_true_prob, y_true_coord).item()

            y_pred_prob_n = y_pred_prob.cpu().detach().numpy()
            y_pred_coord_cpu = y_pred_coord.cpu().detach().numpy()
            y_true_coord_cpu = y_true_coord.cpu().detach().numpy()

            for idx in range(i.shape[0]):
                for c in range(y_true_prob.shape[1]):
                    if y_true_prob[idx, c] == 1:
                        y_true_prob_list.append(c + 1)
                    else:
                        y_true_prob_list.append(0)
                    y_pred_prob_list.append(np.argmax(y_pred_prob_n[idx, c], axis=-1) * (c + 1))
                    y_pred_coord_list.append(y_pred_coord_cpu[idx, c])
                    y_true_coord_list.append(y_true_coord_cpu[idx, c])

        val_loss = val_loss / num_samples

        logging.debug("\n{}".format(classification_report(y_true_prob_list,
                                                          y_pred_prob_list, target_names=["No Bear", *classes])))
        logging.debug("Predicted coordinates:")
        for i, c in enumerate(classes):
            relevant_coords = [p for p, x in zip(y_pred_coord_list, y_true_prob_list) if x == (i + 1)]
            logging.debug("{}:   Mean: {}  STD: {}".format(
                c,
                np.mean(relevant_coords, axis=0),
                np.std(relevant_coords, axis=0)))
        logging.debug("Error:")
        for i, c in enumerate(classes):
            relevant_coords = np.array([np.linalg.norm(p - t) for p, t, x in
                                                zip(y_pred_coord_list, y_true_coord_list, y_true_prob_list) if x == (i + 1)])
            relevant_coords_px = np.array([np.linalg.norm((p - t) * np.array([3980, 1080])) for p, t, x in
                                                zip(y_pred_coord_list, y_true_coord_list, y_true_prob_list) if x == (i + 1)])
            logging.debug("{}:   Mean: {}  STD: {}".format(
                c,
                np.mean(relevant_coords, axis=0),
                np.std(relevant_coords, axis=0)))
            logging.debug("{} (px):   Mean: {}  STD: {}".format(
                c,
                np.mean(relevant_coords_px, axis=0) ,
                np.std(relevant_coords_px, axis=0)))
        logging.debug("MAE: {}".format(np.mean([np.linalg.norm(p - t) for p, t, x in
                                                zip(y_pred_coord_list, y_true_coord_list, y_true_prob_list) if x > 0])))
        logging.debug("Val loss: {}".format(val_loss))

        cur_score = val_loss

        if cur_score < best_score:
            best_model = copy.deepcopy(model)
            best_score = cur_score
            current_patience = patience
        else:
            current_patience -= 1
            if current_patience < 1:
                logging.info("No more patience left")
                model.train()
                break

        logging.info("Best score: {:.3f}   Current score: {:.3f}   Patience: {}".format(
            best_score,
            cur_score,
            current_patience
        ))

        model.train()

    return best_model
