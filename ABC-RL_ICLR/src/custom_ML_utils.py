import os
import torch
from tqdm import tqdm
import torch.nn.functional as torch_func
from custom_exceptions import PathNotFoundException


def compute_accuracy(predictions: torch.tensor, targets: torch.tensor) -> float:
    """
    Compute the model's accuracy
    :param predictions: tensor containing the predicted labels
    :param targets: tensor containing the target labels
    :return: the accuracy in [0, 1]
    """
    accuracy = torch.div(torch.sum(predictions == targets), len(targets))
    return accuracy


def temperature_scaled_softmax(logits: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    """
    Temperature scaled softmax function
    :param logits: logits matrix of shape (batch_size, num_classes)
    :param temperature: float value for the temperature scaling
    :return: temperature scaled softmax matrix of shape (batch_size, num_classes)
    """
    logits = logits / temperature
    return torch_func.softmax(logits, dim=1)


def get_logits_preds_targets_data(model: torch.nn.Module,
                                  dataloader: torch.utils.data.DataLoader,
                                  device: torch.device,
                                  *args, **kwargs):
    """
    Compute the logits given input data loader and model
    :param model: model utilized for the logits computation
    :param dataloader: loader for the training data
    :param device: device used for computation
    :return: logits, predictions, targets, and data
    """
    logits_lst = []
    targets_lst = []
    predictions_lst = []
    data_lst = []

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(dataloader),
                                              desc='Computation ongoing...',
                                              ascii=True,
                                              total=len(dataloader)):
            logits = model(data.to(device))
            logits_lst.append(logits.detach().cpu())

            predictions = torch.argmax(torch_func.softmax(logits, dim=1), dim=1)
            predictions_lst.append(predictions.detach().cpu().reshape(-1, 1))

            targets_lst.append(target.detach().cpu().reshape(-1, 1))

            data_lst.append(data.detach().cpu())

    logits = torch.vstack(logits_lst)
    data = torch.vstack(data_lst)
    predictions = torch.vstack(predictions_lst).reshape(-1)
    targets = torch.vstack(targets_lst).reshape(-1)
    if 'save_to_folder' in kwargs:
        try:
            os.makedirs(kwargs['save_to_folder'], exist_ok=True)
            torch.save(obj=logits, f=kwargs['save_to_folder'] + 'logits.pt')
            torch.save(obj=predictions, f=kwargs['save_to_folder'] + 'predictions.pt')
            torch.save(obj=targets, f=kwargs['save_to_folder'] + 'targets.pt')
            torch.save(obj=data, f=kwargs['save_to_folder'] + 'data.pt')
        except PathNotFoundException as e:
            print(e)
    return logits, predictions, targets, data

#####################################
#####################################
#####################################
# import os
# import numpy as np
# from UTLS import utls
# from collections import OrderedDict
# from sklearn.linear_model import LogisticRegressionCV as lrcv
#
#


#
#


#
#
# def get_trained_regressor(training_data: np.ndarray, target_labels: np.ndarray, **kwargs):
#     return lrcv(n_jobs=-1).fit(training_data, target_labels)