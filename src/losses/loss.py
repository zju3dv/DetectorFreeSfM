from loguru import logger

import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fine_type = config["fine_type"]

    def compute_fine_loss(self, data):
        loss = 0
        num_steps = len(data['reference_points_refined'])
        for i in range(num_steps):
            if self.fine_type == "l2_with_std" and data['std'] is not None:
                loss += self._compute_fine_loss_l2_std(data['reference_points_refined'][i], data['reference_points_gt'], data['std'][i], data['track_valid_mask'])
            else:
                loss += self._compute_fine_loss_l2(data['reference_points_refined'][i], data['reference_points_gt'], data['track_valid_mask'])
        return loss / num_steps

    def _compute_fine_loss_l2(self, points_pred, points_gt, mask):
        """
        Args:
            points_pred (torch.Tensor): [b, n_view, n_track, 2] <x, y>
            points_gt (torch.Tensor): [..., 2] <x, y>
            mask (torch.Tensor): [...]
        """
        # l2 loss with std
        if mask is not None:
            if mask.sum() == 0:
                # No available GT
                logger.error("No available GT")
                mask[0, 0, 0] = 1
        offset_l2 = ((points_pred - points_gt)[mask.bool()] ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, points_pred, points_gt, std, mask):
        """
        Args:
            points_pred (torch.Tensor): [b, n_view, n_track, 2] <x, y>
            points_gt (torch.Tensor): [..., 2] <x, y>
            std (torch.Tensor): [...]
            mask (torch.Tensor): [...]
        """
        # use std as weight that measures uncertainty
        inverse_std = 1.0 / torch.clamp(std, min=1e-10)
        weight = (
            inverse_std / torch.mean(inverse_std)
        ).detach()  # avoid minizing loss through increase std

        # l2 loss with std
        if mask is not None:
            if mask.sum() == 0:
                logger.error("No available GT")
                mask[0, 0, 0] = 1
        offset_l2 = ((points_pred - points_gt)[mask.bool()] ** 2).sum(-1)
        loss = (offset_l2 * weight[mask.bool()])

        return loss.mean()

    def forward(self, data):
            """
            Update:
                data (dict): update{
                    'loss': [1] the reduced loss across a batch,
                    'loss_scalars' (dict): loss scalars for tensorboard_record
                }
            """
            loss_scalars = {}
            loss = 0

            # Distance loss
            loss_f = self.compute_fine_loss(data,
            )
            if loss_f is not None:
                loss += loss_f
                loss_scalars.update(
                    {"loss_fine": loss_f.clone().detach().cpu()}
                )
            else:
                assert self.training is False
                loss_scalars.update({"loss_f": torch.tensor(1.0)})  # 1 is the upper bound

            loss_scalars.update({"loss": loss.clone().detach().cpu()})
            data.update({"loss": loss, "loss_scalars": loss_scalars})