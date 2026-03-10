import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg11
from scripts.multiview_detector.models.resnet import resnet18
import matplotlib.pyplot as plt

from scripts.multiview_detector.models.spatial_transformer_lowRes import spatial_transoformation_layer


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.batch_size = dataset.batch_size
        self.view_size = dataset.view_size
        self.patch_num = dataset.patch_num
        self.cropped_size = dataset.cropped_size

        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        # self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        self.coord_map = self.create_coord_map(self.cropped_size + [1])

        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:0')  #('cuda:1') zq
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            aa = list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2]
            aa = aa[:3] + aa[4:]
            base = nn.Sequential(*aa)

            # previous
            # base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])

            split = 7
            self.base_pt1 = base[:split].to('cuda:0')  #('cuda:1') zq
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 1, 1, bias=False)).to('cuda:0')
        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        # self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * 1, 512, 3, padding=1), nn.ReLU(),
        #                                     # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
        #                                     nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
        #                                     nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=True)).to('cuda:0')
        pass

    def forward(self,
                imgs,
                camera_paras,
                wld_map_paras,
                hw_random,
                visualize=False):

        camera_paras2_shape = (self.batch_size * self.view_size, 15)
        camera_paras2 = torch.reshape(camera_paras, shape=camera_paras2_shape)

        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features0 = []
        imgs_result = []
        for cam in range(self.num_cam):
            t = imgs[:, cam]
            img_feature = self.base_pt1(imgs[:, cam].to('cuda:0'))  #('cuda:1') zq
            img_feature = self.base_pt2(img_feature.to('cuda:0'))
            # img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to('cuda:0'))
            imgs_result.append(img_res)

            # projetion layer for CVCS dataset:
            paras = [self.batch_size, self.view_size, self.patch_num, self.cropped_size]
            world_feature = spatial_transoformation_layer(paras,
                                                          [img_feature.to('cuda:0'),   # 'cuda:0'
                                                           camera_paras2[cam:cam+1].to('cuda:0'), # 'cuda:0'
                                                           wld_map_paras.to('cuda:0'), # 'cuda:0'
                                                           hw_random.to('cuda:0')      # 'cuda:0'
                                                           ])

            if visualize:
                plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
                plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()

            world_features0.append(world_feature.to('cuda:0'))

        world_features = torch.cat(world_features0 + [self.coord_map.repeat([B*self.patch_num, 1, 1, 1]).to('cuda:0')], dim=1)
        if visualize:
            plt.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
            plt.show()
        map_result = self.map_classifier(world_features.to('cuda:0'))
        # map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
        map_result = torch.reshape(map_result, shape=(B, self.patch_num, map_result.shape[2], -1))

        if visualize:
            plt.imshow(torch.norm(map_result[0].detach(), dim=0).cpu().numpy())
            plt.show()
        return map_result, imgs_result

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


def test():
    from scripts.multiview_detector.datasets.lcvcs.frameDataset import frameDataset
    from scripts.multiview_detector.datasets.lcvcs.Wildtrack import Wildtrack
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    map_res, img_res = model(imgs, visualize=True)
    pass


if __name__ == '__main__':
    test()
