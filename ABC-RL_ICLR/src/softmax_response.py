import torch
from custom_ML_utils import temperature_scaled_softmax as tss


def softmax_response_score(logits: torch.Tensor):
    """
    Compute the softmax response attack scores given the logits
    :param logits: logits matrix of shape (batch_size, num_classes)
    :return: scores matrix of shape (batch_size, 1) containing the softmax response scores
    """
    scores_softmax_response, _ = torch.max(tss(logits=logits, temperature=1.), dim=1)
    return scores_softmax_response