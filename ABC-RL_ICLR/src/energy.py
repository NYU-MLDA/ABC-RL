import torch
from custom_ML_utils import temperature_scaled_softmax as tss


def energy_score(logits: torch.Tensor, t: float = 1.):
    """
    Compute the energy scores given the logits
    :param logits: logits matrix of shape (batch_size, num_classes)
    :param t: temperature parameter for the temperature scaling
    :return: scores matrix of shape (batch_size, 1) containing the softmax response scores
    """
    energy_score = t * torch.logsumexp(logits / t, dim=1)
    return energy_score