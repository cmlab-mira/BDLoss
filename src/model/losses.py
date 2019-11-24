import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """The Dice loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """
        # Get the one-hot encoding of the ground truth label.
        target = torch.clamp(target, min=0, max=2)
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice loss.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (output * target).sum(reduced_dims)
        union = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return 1 - score.mean()

class BDLoss(nn.Module):
    """The batch-wise dice loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """
        # Define the class of tumor
        class_t = output.size(1) - 1
        smooth = 1e-10
        alpha = 0.5
        gamma = 0.4

        # Get the one-hot encoding of the ground truth label.
        target = torch.clamp(target, min=0, max=2)
        target = torch.zeros_like(output).scatter_(1, target, 1) # (N, 1, *) -> (N, C, *)

        reduced_dims = list(range(2, output.dim()))
        ground_truth_sum = target.sum(reduced_dims) # (N, C)

        w_b = torch.ones_like(ground_truth_sum) # (N, C)
        partial_volume = torch.pow( target.sum(reduced_dims)[:, class_t] + smooth, 1/3) # (N, 1)
        total_volume = (1 / partial_volume).sum()
        w_b[:, class_t] = (1 / partial_volume) / (total_volume) * output.size(0)

        intersection = ((output*target).sum(reduced_dims))
        union = (((output**2).sum(reduced_dims) + (target**2).sum(reduced_dims)))

        _score = (2.0 * w_b * intersection / (union + 1e-10)).mean(0) + smooth
        _loss = alpha * torch.pow(-1 * torch.log(_score), gamma)

        return _loss.mean()
