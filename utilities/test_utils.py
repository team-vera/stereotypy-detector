import torch
from typing import Iterable, Dict, List
import numpy as np
from sklearn.metrics import classification_report
import time


def test_cls(test_data: Iterable,
             model: torch.nn.Module,
             device: str,
             classes: List[str],
             print_cls_report: bool = False) -> Dict[str, Dict[str, float]]:
    """Return the classification report for the given model and test data

    Args:
        test_data (Iterable): Data to test on
        model (torch.nn.Module): Model to test
        device (str): Device to do calculation on
        classes (List[str]): Class names
        print_cls_report (bool, optional): Print the classification report. Defaults to False

    Returns:
        Dict[str, Dict[str, float]]: Classification report
    """

    model.eval()

    y_pred_list = []
    y_true_list = []

    runtimes = []

    for i, y_true in test_data:
        i = i.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        start_time = time.time()
        y_pred = model(i)
        end_time = time.time()

        y_pred_list.extend(
            np.argmax(y_pred.cpu().detach().numpy(), axis=-1))

        y_true_list.extend(y_true.cpu().detach().numpy())

        runtimes.append(end_time - start_time)

    return_data = {}

    return_data["score"] = classification_report(y_true_list,
                                                 y_pred_list,
                                                 target_names=classes,
                                                 output_dict=True,
                                                 zero_division=0)

    if print_cls_report:
        print(classification_report(y_true_list,
                                    y_pred_list,
                                    target_names=classes,
                                    zero_division=0,
                                    digits=4))

    return_data["time"] = runtimes

    return return_data
