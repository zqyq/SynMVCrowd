import torch
import torch.nn.functional as F
from torch import nn


class GaussianMSE_scale(nn.Module):

    def __init__(self, scale):
        self.scale = scale
        super().__init__()

    def forward(self, x, target, kernel):
        target = self._traget_transform(x, self.scale * target, kernel)
        # plt.imshow(target.detach().cpu().numpy()[4, 0])
        # plt.show()
        # plt.imshow(x.detach().cpu().numpy()[4, 0])
        # plt.show()
        return F.mse_loss(x, target), target

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target


class GaussianMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, target, kernel):
        target = self._traget_transform(x, target, kernel)
        # plt.imshow(target.detach().cpu().numpy()[4, 0])
        # plt.show()
        # plt.imshow(x.detach().cpu().numpy()[4, 0])
        # plt.show()
        return F.mse_loss(x, target), target

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target


class GaussianMSE_norm(nn.Module):
    def __init__(self, scale=100):
        self.scale = scale
        super().__init__()

    def forward(self, x, target, kernel):
        before_sum = target.sum()
        target = self._traget_transform(x, target, kernel)
        after_sum = target.sum()
        target = target * before_sum * self.scale / (after_sum + 1e-6)
        return F.mse_loss(x, target), target

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target
