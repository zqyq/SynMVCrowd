# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
# from docs.tutorials.deform_source_mesh_to_target_mesh import edge_losses
from torch.nn.modules.loss import _Loss

from scripts.multiview_detector.loss.ot_loss.Cost import EDCost, MDCost
from scripts.multiview_detector.loss.ot_loss.geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# device = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

eps = 1e-20


def grid(H, W, stride):
    coodx = (torch.arange(0, W, step=stride) + stride) / W
    coody = (torch.arange(0, H, step=stride) + stride) / H
    y, x = torch.meshgrid([coody.type(dtype) / 1, coodx.type(dtype) / 1])
    a = torch.stack((x, y), dim=2).view(-1, 2)
    return a


class GeneralizedLoss(_Loss):
    def __init__(self, type='md', reduction='mean', scale=10, device='cuda:0') -> None:
        super().__init__()
        self.reduction = reduction
        self.tau1 = 1
        self.tau2 = 1
        if type == 'md':
            self.cost = MDCost()
        else:
            self.cost = EDCost()
        self.blur = 0.0001
        self.scaling = 0.75
        self.reach = 0.5
        self.p = 1
        self.uot = SamplesLoss(blur=self.blur, scaling=self.scaling, debias=False, backend='tensorized', cost=self.cost,
                               reach=self.reach, p=self.p)
        self.pointLoss = nn.L1Loss(reduction=reduction)
        self.pixelLoss = nn.MSELoss(reduction=reduction)
        self.scale = scale
        self.device = device

    def forward(self, dens, dots, cam_loc):
        # dens: [1,1,200,200], dots: [1,1,312,2], cal_loc: [view,3]
        bs = dens.size(0)
        point_loss, pixel_loss, emd_loss = 0, 0, 0
        entropy = 0
        for i in range(bs):
            den = dens[i, 0]
            if dots.shape[2] < 1 or den.sum() < 1e-8:

                point_loss += torch.abs(den).mean()
                pixel_loss += torch.abs(den).mean()
                emd_loss += torch.abs(den).mean()
                print('value too small')
                return point_loss + pixel_loss + emd_loss

            else:
                B = self.scale * torch.ones(1, dots.shape[2], 1).to(self.device) # .to(device[2])
                B_coord = torch.squeeze(dots, dim=0).to(self.device) # .to(device[2])

                A, A_coord = self.den2coord(den)

                oploss, F, G = self.uot(A, A_coord, B, B_coord, cam_loc)
                C = self.cost(A_coord, B_coord, calc=cam_loc)

                PI = torch.exp((F.view(1, -1, 1) + G.view(1, 1, -1) - C).detach() / (self.blur ** self.p)) * A * B.view(
                    1, 1, -1)

                entropy += torch.mean((eps + PI) * torch.log(eps + PI))
                emd_loss += torch.mean(oploss)
                point_loss += self.pointLoss(PI.sum(dim=1).view(1, -1, 1), B)
                pixel_loss += self.pixelLoss(PI.sum(dim=2).detach().view(1, -1, 1), A)

                # print("emd_loss: {}, point_loss: {}, pixel_loss: {}, entropy: {}".format(emd_loss, point_loss, pixel_loss, entropy))

                mp = torch.sum(PI)
                mb = torch.sum(B)
                ma = torch.sum(A)

                actual_tau1 = self.actual_tau1(ma, mb, mp)
                actual_tau2 = self.actual_tau2(ma, mb, mp)

                loss = (emd_loss + actual_tau2 * pixel_loss + actual_tau1 * point_loss + self.blur * entropy) / bs
                # print(f'emd:{emd_loss}, pixel:{actual_tau2 * pixel_loss}, point:{actual_tau1 * point_loss}, entropy:{self.blur * entropy}')
                return loss

    def actual_tau1(self, ma, mb, mp):
        return 0 if ma < mb < mp or mp < mb < ma else self.tau1

    def actual_tau2(self, ma, mb, mp):
        return 0 if mb < ma < mp or mp < ma < mb else self.tau2

    def den2coord(self, den):
        den1 = den
        # view1
        coord1 = torch.nonzero(den1)
        denval1 = den1[coord1[:, 0], coord1[:, 1]]
        coord1 = coord1.float()
        coord1[:, 0] = coord1[:, 0] / float(den1.shape[-2])
        coord1[:, 1] = coord1[:, 1] / float(den1.shape[-2])
        coord1[:, [0, 1]] = coord1[:, [1, 0]]
        # print("coord1: {}".format(coord1.shape))
        return torch.reshape(denval1, (1, -1, 1)), coord1.unsqueeze(0)


def test():
    import matplotlib.pyplot as plt
    # cam_pos = [-7, 384 - 190]
    # cam_pos = [189, 384 - 6]
    cam_pos = [102, 384 - 391]

    x = np.linspace(0, 400, 1000)
    y = np.linspace(0, 400, 1000)
    x, y = np.meshgrid(x, y)

    gtx = np.linspace(20, 320, 5)
    gty = np.linspace(20, 384, 5)
    xx, yy = np.meshgrid(gtx, gty)
    gt_coord = np.column_stack((xx.reshape(-1), yy.reshape(-1)))

    direcY = gt_coord - cam_pos
    rot_col1 = direcY.copy()
    abs_dy = np.sqrt(np.sum(direcY ** 2, -1))
    sigma1 = np.ones_like(abs_dy) * 0.5
    sigma2 = np.ones_like(abs_dy) * 1.5
    inv_var = np.zeros((abs_dy.shape[0], 2, 2))
    inv_var[:, 0, 0] = 1 / (sigma1)
    inv_var[:, 1, 1] = 1 / (sigma2)
    rot_col1 = rot_col1 / np.sqrt(rot_col1[:, 0] ** 2 + rot_col1[:, 1] ** 2)[..., None]

    '''
    (x, y) -> (-y, x)
    rotation = [x,  -y
                y   x]
    '''
    rot_col2 = rot_col1.copy()
    rot_col2[:, [0, 1]] = rot_col2[:, [1, 0]]
    rot_col2[:, 0] = -1 * rot_col2[:, 0]

    rot_col1 = rot_col1[..., None]
    rot_col2 = rot_col2[..., None]

    rot_mat = np.concatenate((rot_col1, rot_col2), axis=-1)
    rot_mat_T = np.transpose(rot_mat, (0, 2, 1))

    # t = inv_var @ rot_mat_T
    inv_cov = rot_mat @ inv_var @ rot_mat_T

    for i in range(25):
        z = inv_cov[i, 0, 0] * (x - gt_coord[i, 0]) ** 2 + inv_cov[i, 1, 1] * (y - gt_coord[i, 1]) ** 2 \
            + (inv_cov[i, 1, 0] + inv_cov[i, 0, 1]) * (x - gt_coord[i, 0]) * (y - gt_coord[i, 1]) - 100
        print(inv_cov[i, 1, 0] + inv_cov[i, 0, 1])
        plt.contour(x, y, z, 0)

    plt.show()


if __name__ == '__main__':
    # test()
    # dens: [1,1,200,200], dots: [1,1,312,2], cal_loc: [view,3]
    from thop import profile
    from scripts.vis import show_tensor_images#noqa
    dens=torch.randn(7, 1, 224, 224)
    dots=torch.randint(0, 224, (7, 1, 312, 2)).float()
    cam_loc=torch.randn(7, 3)
    # 计算浮点数FLOPs
    md_loss_fn = GeneralizedLoss(type='md', reduction='mean', scale=10, device='cuda:0')
    ed_loss_fn = GeneralizedLoss(type='ed', reduction='mean', scale=10, device='cuda:0')
    loss_md = md_loss_fn(dens.cuda(), dots.cuda(), cam_loc.cuda())
    loss_ed = ed_loss_fn(dens.cuda(), dots.cuda(), cam_loc.cuda())
    print(f'md_loss: {loss_md}, ed_loss: {loss_ed}')

    flops_md, params_md = profile(md_loss_fn, inputs=(dens.cuda(), dots.cuda(), cam_loc.cuda(),))
    flops_ed, params_ed = profile(ed_loss_fn, inputs=(dens.cuda(), dots.cuda(), cam_loc.cuda(),))
    print(f'MD Loss - FLOPs: {flops_md}, Params: {params_md}'
          f'nED Loss - FLOPs: {flops_ed}, Params: {params_ed}')
    




