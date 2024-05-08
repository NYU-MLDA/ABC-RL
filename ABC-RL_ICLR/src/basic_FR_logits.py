import os
import sys
import torch
import logging
from tqdm import tqdm
from torch import optim
from typing import Callable
from torch.nn import functional as tf


def dFR_Logits(logits_output: torch.Tensor, logits_reference: torch.Tensor, epsilon: float = 1e-12):
    """
    compute the dFR_Logits distance, i.e. the FR distance between two probability distributions resulting from the softmax
    probability evaluated at two data points, it works by broadcasting since they logits have the same class dimension and
    one of the two has size 1 in the other dimension
    :param logits_output: the logits output by the model
    :param logits_reference: the logits from the reference
    :param epsilon: approx term, useful to make sure that the acos argument does not trigger any exception
    :return: tensor with the computed distance
    """
    inner = torch.sum(torch.sqrt(tf.softmax(logits_output, dim=1) * tf.softmax(logits_reference, dim=1)), dim=1)
    return 2 * torch.acos(torch.clamp(inner, -1 + epsilon, 1 - epsilon))


def logits_centroid_estimator(logits: torch.Tensor, targets: torch.Tensor, init_tensor: torch.Tensor,
                              distance: Callable,
                              epochs: int = 100, lr: float = .01):
    """
    estimate the centroids for the logits class by class using gradient descent
    :param logits: logits tensor NxC, where N is the number of samples and C is the number of classes
    :param targets: target labels corresponding to each sample so to estimate the centroids per clas
    :param init_tensor: initial placeholder tensor for the centroids
    :param distance: function to compute the desired distance
    :param epochs: iterations to optimize the centroids estimation
    :param lr: learning rate for the centroid estimation
    :return: the final estimated centroid as a tensor, and the list with the loss per epoch
    """
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger('logits_centroid_estimator')
    logger.setLevel(logging.INFO)
    n_classes = init_tensor.shape[1]
    centroid = [
        torch.autograd.Variable(init_tensor[i].reshape(1, -1), requires_grad=True)
        for i in range(n_classes)
    ]

    logger.info('Running centroid estimation...')
    epoch_loss = [[] for _ in range(n_classes)]

    for c in range(n_classes):
        optimizer = optim.Adam([centroid[c]], lr=lr)
        for _ in range(epochs):
            filtered_id = targets == c
            # logger.info('Centroid class {klass}: {centroid}'.format(klass=c, centroid=centroid[c]))
            optimizer.zero_grad()
            if filtered_id.sum() > 0:
                d = distance(logits[filtered_id].detach(), centroid[c])
                loss = torch.mean(d)
                logger.info('Loss class {klass}: {loss}'.format(klass=c, loss=loss.item()))
                epoch_loss[c].append(loss.item())
                loss.backward()
                optimizer.step()

    return torch.vstack(centroid), epoch_loss


def estimate_logits_centroids(reference_logits: torch.Tensor, reference_targets: torch.Tensor, device: torch.device,
                              num_classes: int, distance: Callable = dFR_Logits, epochs: int = 100, lr: float = .01,
                              *args, **kwargs):
    """
    estimate the logits centroids using logits_centroid_estimator
    :param reference_logits: logits used as reference for each class
    :param reference_targets: target labels corresponding to the logits
    :param device: device to perform the computation
    :param num_classes: number of classes in the dataset
    :param distance: distance function to be used in the centroid estimation
    :param epochs: number of epochs for the centroids estimation
    :param lr: learning rate for the centroids estimation
    :param args: []
    :param kwargs: optional path to the file where the centroids will be stored
    :return: centroid, epoch_loss, logits, targets
    """
    init_tensor = torch.eye(num_classes)

    init_tensor = init_tensor.to(device)
    logits = reference_logits.to(device)
    targets = reference_targets.to(device)

    centroid, epoch_loss = logits_centroid_estimator(
        logits, targets, init_tensor, distance, epochs, lr)

    # save tensor
    if 'save_dir' in kwargs:
        os.makedirs(os.path.dirname(kwargs['save_dir']), exist_ok=True)
        torch.save(centroid, '/'.join((kwargs['save_dir'], 'centroids.pt')))

    return centroid, epoch_loss, logits, targets


def FR_0(logits_output: torch.Tensor, logits_reference: dict, device=torch.device, temperature: float = 1.,
         epsilon=1e-12):
    """
    compute the FR_0 distance of the model's logits from the reference distribution, i.e. the centroids computed class by
    class
    :param logits_output: model's logits
    :param logits_reference: centroids, reference logits, dictionary with one centroid per class
    :param device: device the computations are performed on
    :param temperature: temperature term for the scaling of the logits
    :param epsilon: epsilon term for the clamp in the acos computation
    :return: tensor with the computed FR_0
    """
    FR_0_lst = []
    logits_output = logits_output.to(device)
    for log_ref_id in tqdm(range(len(logits_reference)), ascii=True):
        mu = logits_reference[log_ref_id].to(device)
        FR_0_lst.append(
            dFR_Logits(logits_output / temperature, mu.reshape(1, -1) / temperature, epsilon).reshape(-1, 1))
    return torch.sum(torch.hstack(FR_0_lst), 1)