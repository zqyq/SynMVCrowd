import os

import kornia.geometry
# from scripts.multiview_detector.track_tacular_models.conv_world_feat import ConvWorldFeat, DeformConvWorldFeat
# from scripts.multiview_detector.track_tacular_models.trans_world_feat import TransformerWorldFeat, DeformTransWorldFeat, \
#     DeformTransWorldFeat_aio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg11

from scripts.multiview_detector.models.resnet import resnet18
# from scripts.multiview_detector.utils.image_utils import img_color_denormalize, array2heatmap
from scripts.multiview_detector.utils.projection import get_worldcoord_from_imgcoord_mat, project_2d_points

from scripts.vis import show_tensor_images#noqa
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def output_head(in_dim, feat_dim, out_dim):
    if feat_dim:
        fc = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(feat_dim, out_dim, 1))
    else:
        fc = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
    return fc


def output_head2(in_dim, feat_dim=64, out_dim=1):
    fc = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, padding=1), nn.ReLU(),
                       nn.Conv2d(feat_dim, feat_dim * 2, 3, padding=1), nn.ReLU(),
                       nn.Conv2d(feat_dim * 2, out_dim, 1))

    return fc


def create_reference_map(dataset, n_points=4, downsample=2, visualize=False):
    H, W = dataset.Rworld_shape  # H,W; N_row,N_col
    H, W = H // downsample, W // downsample

    ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32),
                                  torch.linspace(0.5, W - 0.5, W, dtype=torch.float32))
    ref = torch.stack((ref_x, ref_y), -1).reshape([-1, 2])
    if n_points == 4:
        zs = [0, 0, 0, 0]
    elif n_points == 8:
        zs = [-0.4, -0.2, 0, 0, 0.2, 0.4, 1, 1.8]
    else:
        raise Exception
    ref_maps = torch.zeros([H * W, dataset.num_cam, n_points, 2])
    world_zoom_mat = np.diag([dataset.world_reduce * downsample, dataset.world_reduce * downsample, 1])
    Rworldgrid_from_worldcoord_mat = np.linalg.inv(
        dataset.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ dataset.base.world_indexing_from_xy_mat)
    for cam in range(dataset.num_cam):
        mat_0 = Rworldgrid_from_worldcoord_mat @ get_worldcoord_from_imgcoord_mat(dataset.base.intrinsic_matrices[cam],
                                                                                  dataset.base.extrinsic_matrices[cam])
        for i, z in enumerate(zs):
            mat_z = Rworldgrid_from_worldcoord_mat @ get_worldcoord_from_imgcoord_mat(
                dataset.base.intrinsic_matrices[cam],
                dataset.base.extrinsic_matrices[cam],
                z / dataset.base.worldcoord_unit)
            img_pts = project_2d_points(np.linalg.inv(mat_z), ref)
            ref_maps[:, cam, i, :] = torch.from_numpy(project_2d_points(mat_0, img_pts))
        pass
        if visualize:
            fig, ax = plt.subplots()
            field_x = (ref_maps[:, cam, 3, 0] - ref_maps[:, cam, 1, 0]).reshape([H, W])
            field_y = (ref_maps[:, cam, 3, 1] - ref_maps[:, cam, 1, 1]).reshape([H, W])
            ax.streamplot(ref_x.numpy(), ref_y.numpy(), field_x.numpy(), field_y.numpy())
            ax.set_aspect('equal', 'box')
            ax.invert_yaxis()
            plt.show()

    ref_maps[:, :, :, 0] /= W
    ref_maps[:, :, :, 1] /= H
    return ref_maps


class MVDeTr(nn.Module):
    def __init__(self, dataset, arch='resnet18', z=0, world_feat_arch='conv', bottleneck_dim=512,
                 outfeat_dim=64, droupout=0.5, num_cam=7, devices=('cuda:0', 'cuda:1')):
        super().__init__()

        self.num_cam = num_cam
        self.view_size = self.num_cam
        self.devices = devices
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        if arch == 'vgg11':
            self.base = vgg11(pretrained=False).features
            self.base[-1] = nn.Identity()
            self.base[-4] = nn.Identity()
            base_dim = 512
        elif arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(pretrained=False,
                                                     replace_stride_with_dilation=[False, True, True]).children())[
                                       :-2]).to(devices[0])
            base_dim = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]
        # if bottleneck_dim:
        #     self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, bottleneck_dim, 1), nn.Dropout2d(droupout))
        #     base_dim = bottleneck_dim
        # else:
        #     self.bottleneck = nn.Identity()

        # img heads
        self.img_heatmap1 = output_head(base_dim, outfeat_dim, 1).to(devices[0])
        # self.img_offset = output_head(base_dim, outfeat_dim, 2)
        # self.img_wh = output_head(base_dim, outfeat_dim, 2)

        self.depth_scales = 1
        self.depth_classifier = nn.Sequential(nn.Conv2d(512, 64, 1), nn.ReLU(),
                                              nn.Conv2d(64, self.depth_scales, 1, bias=False)).to(devices[0])

        self.feat_before_merge = nn.ModuleDict({
            f'{i}': nn.Conv2d(512, 512, 3, padding=1)
            for i in range(self.depth_scales)
        }).to(devices[0])

        # # world heads
        # self.world_heatmap = output_head(base_dim, outfeat_dim, 1)
        # # init
        # # self.img_heatmap[-1].bias.data.fill_(-2.19)
        # # fill_fc_weights(self.img_wh)
        # self.world_heatmap[-1].bias.data.fill_(-2.19)
        # pass

        # self.map_classifier_shot = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        #                                     # # w/o large kernel
        #                                     # nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        #                                     # nn.Conv2d(512, 1, 3, padding=1, bias=False)).to(self.devices[0])
        #
        #                                     # with large kernel
        #                                     nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
        #                                     nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False),).to(self.devices[0])
        self.map_classifier_shot = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                                 # # w/o large kernel
                                                 # nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                                 # nn.Conv2d(512, 1, 3, padding=1, bias=False)).to(self.devices[0])

                                                 # with large kernel
                                                 nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                                 nn.Conv2d(512, 1, 3, padding=1, bias=False), ).to(devices[0])

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret

    def forward(self, imgs, train=True):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam

        img_feat = self.base(imgs.reshape([B * N, C, H, W]))
        # img_feat = self.bottleneck(img_feat)
        img_res = self.img_heatmap1(img_feat)
        # img_res = img_res.reshape(B, self.num_cam, img_res.shape[-3], img_res.shape[-2], img_res.shape[-1])
        img_feat = img_feat.reshape(B, self.num_cam, img_feat.shape[-3], img_feat.shape[-2], img_feat.shape[-1])

        wf = []
        for batch in range(B):
            depth_select = self.depth_classifier(img_feat[batch]).softmax(dim=0)
            wraped_feat = 0
            # normalized_h = wld_map_paras[batch][5] / 1750.0
            # muti_hei = [normalized_h * 1750]

            for i in range(self.depth_scales):
                in_feat = img_feat[batch] * depth_select[:, i][:, None]
                # h = muti_hei[i]
                cur_height_f = []
                for cam in range(self.num_cam):
                    proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to(self.devices[0])
                    world_feature = kornia.geometry.warp_perspective(in_feat[cam][None], proj_mat,
                                                                     self.reducedgrid_shape)
                    cur_height_f.append(world_feature)
                    if len(cur_height_f) > 1:
                        t = torch.stack(cur_height_f, dim=1).max(1)[0]
                        cur_height_f = []
                        cur_height_f.append(t)

                wraped_feat += self.feat_before_merge[f'{i}'](cur_height_f[0])
            wf.append(wraped_feat)
        wf = torch.stack(wf, dim=0)
        batch, patch, C, w_H, w_W = wf.shape
        wraped_feat = wf.reshape([batch * patch, C, w_H, w_W]).to(self.devices[0])
        world_heatmap_mse = self.map_classifier_shot(wraped_feat)
        # world_heatmap_mse=torch.relu(world_heatmap_mse)
        # mse_fea = self.mse_fea(world_heatmap_mse)
        # wraped_feat = self.feat_down(wraped_feat)
        #
        # world_heatmap_ot = self.map_classifier_ot1(torch.cat((mse_fea, wraped_feat), dim=1))
        # world_heatmap_mse = torch.where(world_heatmap_mse < 0, -world_heatmap_mse, world_heatmap_mse)
        if train:
            world_heatmap_mse = torch.where(world_heatmap_mse < 0, -world_heatmap_mse, world_heatmap_mse)
        else:
            world_heatmap_mse = torch.where(world_heatmap_mse < 0, torch.zeros_like(world_heatmap_mse),
                                            world_heatmap_mse)
        return world_heatmap_mse[None], img_res


def test():
    from scripts.multiview_detector.datasets.lcvcs.frameDataset import frameDataset
    from scripts.multiview_detector.datasets.lcvcs.Wildtrack import Wildtrack
    from torch.utils.data import DataLoader
    from scripts.multiview_detector.utils.decode import ctdet_decode

    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=False, augmentation=False)
    create_reference_map(dataset, 4)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    model = MVDeTr(dataset, world_feat_arch='deform_trans').cuda()
    # model.load_state_dict(torch.load(
    #     '../../logs/wildtrack/augFCS_deform_trans_lr0.001_baseR0.1_neck128_out64_alpha1.0_id0_drop0.5_dropcam0.0_worldRK4_10_imgRK12_10_2021-04-09_22-39-28/MultiviewDetector.pth'))
    imgs, world_gt, imgs_gt, affine_mats, frame = next(iter(dataloader))
    imgs = imgs.cuda()
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(imgs, affine_mats)
    xysc = ctdet_decode(world_heatmap, world_offset)
    pass


if __name__ == '__main__':
    test()
