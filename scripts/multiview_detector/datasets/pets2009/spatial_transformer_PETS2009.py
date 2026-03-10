import os

import cv2
import numpy as np
import torch

import scripts.multiview_detector.datasets.pets2009.camera_proj_PETS2009 as proj

# tf.disable_v2_behavior()
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
pixelheight = 0.4
# detect the enviroment is under Windows or Linux
root_path = '/mnt/d/yunfei' if os.name == 'posix' else 'D:\Pycharm_jobs'
server_name = os.path.join(root_path, "MVFR-Semi-MVCC/PETS2009")


def get_view_mask(view):
    if view == 1:
        view = 'view1'
        view_gp_mask = np.load(f'/mnt/d/Datasets/PETS2009/mask/view1_gp_mask.npz')
    if view == 2:
        view = 'view2'
        view_gp_mask = np.load(f'/mnt/d/Datasets/PETS2009/mask/view2_gp_mask.npz')
    if view == 3:
        view = 'view3'
        view_gp_mask = np.load(f'/mnt/d/Datasets/PETS2009/mask/view3_gp_mask.npz')
    # gp view mask:
    view_gp_mask = view_gp_mask.f.arr_0
    # view_gp_mask = cv2.resize(view_gp_mask, (W, H))
    view_gp_mask = torch.from_numpy(view_gp_mask)
    return view_gp_mask, view


def get_all_view_masks():
    view_masks = {'0': None, '1': None, '2': None, '3': None, '12': None, '13': None, '23': None, '123': None}
    v0_mask = get_view_mask(1)
    v1_mask = get_view_mask(2)
    v2_mask = get_view_mask(3)
    view_masks['0'] = v0_mask[0]
    view_masks['1'] = v1_mask[0]
    view_masks['2'] = v2_mask[0]
    view_masks['01'] = (v0_mask[0] + v1_mask[0]).clamp(0, 1)
    view_masks['02'] = (v0_mask[0] + v2_mask[0]).clamp(0, 1)
    view_masks['12'] = (v1_mask[0] + v2_mask[0]).clamp(0, 1)
    view_masks['012'] = (v0_mask[0] + v1_mask[0] + v2_mask[0]).clamp(0, 1)
    return view_masks


# device = torch.device("cuda")
class SpatialTransformer_2DTo2D_real():

    def __init__(self,
                 view,
                 # output_size,
                 patch_num=1,
                 height=torch.arange(0, 14),
                 device='cuda:0',
                 **kwargs):
        # self.locnet = localization_net
        self.device = device
        self.view = view
        # self.output_size = output_size
        self.patch_num = patch_num
        self.height = height
        self.proj_view = []
        self.maskview1 = []
        self.maskview3 = []
        self.maskview4 = []
        super(SpatialTransformer_2DTo2D_real, self).__init__(**kwargs)

    def __call__(self, inputs, mask=None):
        view = self.view
        height = self.height
        # output = tf.compat.v1.placeholder(tf.float32,shape=(1,1,192,160,3))
        result = []
        for k in range(len(inputs)):
            temp = inputs[k]
            for i in range(len(view)):
                view_i = view[i]
                self.proj_2DTo2D(view_i, temp[i:i + 1, ...], height)
                # output_i, mask = self.proj_splat(inputs[:, ...])  # inputs
                for j in range(temp.shape[0]):
                    output_i, mask = self.proj_splat(temp[j:j + 1, ...])  # inputs
                    if j == 0:
                        output = output_i.unsqueeze(0)
                    else:
                        output = torch.cat([output, output_i.unsqueeze(0)], axis=0)
            # print("output", output.size())
            result.append(output)
        return result

    def Image2World(self, view, imgcoords):
        N = imgcoords.shape[0]
        wld_coords = []
        for i in range(N):
            imgcoords_i = imgcoords[i, :]

            Xi = imgcoords_i[0]
            Yi = imgcoords_i[1]
            Zw = imgcoords_i[2]

            XYw = proj.Image2World(view, Xi, Yi, Zw)
            wld_coords.append(XYw)
        wld_coords = torch.tensor(wld_coords)
        return wld_coords

    def World2Image(self, view, wldcoords):
        N = wldcoords.size()[0]
        imgcoords = []
        for i in range(N):
            wldcoords_i = wldcoords[i, :]
            Xw = wldcoords_i[0].item()
            Yw = wldcoords_i[1].item()
            Zw = wldcoords_i[2].item()
            XYi = proj.World2Image(view, Xw, Yw, Zw)
            imgcoords.append(XYi)
        imgcoords = torch.tensor(imgcoords)
        # print("imgcoords",imgcoords)
        return imgcoords

    def proj_2DTo2D(self, view, inputs, height):
        # print("len(self.proj_view)",len(self.proj_view))
        if len(self.proj_view) == 0:
            w = 768
            h = 576
            W = int(610 / 4)
            H = int(710 / 4)

            # D = hi #30/4
            bbox = [-31, 29, -45, 25]
            resolution_scaler = 10

            image_size = [int(h / 4), int(w / 4)]
            # print("height",height)
            height_len = len(height)
            ph = height * pixelheight * 1000  # average height of a person in millimeters
            # print("ph",ph)
            # nR, fh, fw, fdim = self.tf_static_shape(inputs)
            nR = inputs.size()[0]
            fh = inputs.size()[1]
            fw = inputs.size()[2]
            fdim = inputs.size()[3]

            self.batch_size, self.gp_x, self.gp_y = nR, W, H

            rsz_h = float(fh) / h
            rsz_w = float(fw) / w

            # Create voxel grid
            grid_rangeX = torch.linspace(0, W - 1, W).to(self.device)
            # print("grid_rangeX",grid_rangeX)
            grid_rangeY = torch.linspace(0, H - 1, H).to(self.device)
            # grid_rangeZ = hi # np.linspace(0, D - 1, D)
            # grid_rangeX, grid_rangeY, grid_rangeZ = np.meshgrid(grid_rangeX, grid_rangeY, grid_rangeZ)
            grid_rangeX, grid_rangeY = torch.meshgrid(grid_rangeX, grid_rangeY)
            # print("grid_rangeX",grid_rangeX)
            grid_rangeX = torch.reshape(grid_rangeX.t(), [-1])
            grid_rangeY = torch.reshape(grid_rangeY.t(), [-1])
            # print(grid_rangeX)
            # grid_rangeZ = np.reshape(grid_rangeZ, [-1])

            grid_rangeX = grid_rangeX * 4 / resolution_scaler + bbox[0]
            # print("grid_rangeX",grid_rangeX)
            grid_rangeX = grid_rangeX * 1000
            # grid_rangeX = np.expand_dims(grid_rangeX, [1,2])
            grid_rangeX = grid_rangeX.unsqueeze(1)
            grid_rangeX = grid_rangeX.unsqueeze(2)
            # grid_rangeX = np.expand_dims(grid_rangeX, 1)
            # print("grid_rangeX",grid_rangeX.size())
            # print("grid_rangeX",grid_rangeX)
            grid_rangeX = grid_rangeX.repeat(1, 1, height_len)
            # print("grid_rangeX",grid_rangeX.size())
            # print("grid_rangeX",grid_rangeX)

            grid_rangeY = grid_rangeY * 4 / resolution_scaler + bbox[2]
            grid_rangeY = grid_rangeY * 1000
            # grid_rangeY = np.expand_dims(grid_rangeY, [1,2])
            grid_rangeY = grid_rangeY.unsqueeze(1)
            grid_rangeY = grid_rangeY.unsqueeze(2)
            grid_rangeY = grid_rangeY.repeat(1, 1, height_len)

            # grid_rangeZ = grid_rangeZ * 400* np.ones(grid_rangeX.shape)
            # grid_rangeZ = np.expand_dims(grid_rangeZ, 1)
            # print("grid_rangeX",grid_rangeX.shape)
            grid_rangeZ = ph * torch.ones(grid_rangeX.shape).to(self.device)  # (30720, 14)
            # print("grid_rangeZ",grid_rangeZ.size())

            wldcoords = torch.cat(([grid_rangeX, grid_rangeY, grid_rangeZ]), axis=1)
            # print("wldcoords",wldcoords)
            # print("wldcoords",wldcoords[:,:,0].shape)
            if view == 1:
                view = 'view1'
                view_gp_mask = np.load(f'{server_name}/mask/view1_gp_mask.npz')
            if view == 2:
                view = 'view2'
                view_gp_mask = np.load(f'{server_name}/mask/view2_gp_mask.npz')
            if view == 3:
                view = 'view3'
                view_gp_mask = np.load(f'{server_name}/mask/view3_gp_mask.npz')
            #
            # gp view mask:
            view_gp_mask = view_gp_mask.f.arr_0
            view_gp_mask = cv2.resize(view_gp_mask, (W, H))
            view_gp_mask = torch.from_numpy(view_gp_mask)
            # # view_gp_mask = tf.expand_dims(view_gp_mask, axis=0)
            # # view_gp_mask = tf.expand_dims(view_gp_mask, axis=1)
            # # view_gp_mask = tf.expand_dims(view_gp_mask, axis=-1)
            #
            view_gp_mask = torch.unsqueeze(view_gp_mask, 0)
            view_gp_mask = torch.unsqueeze(view_gp_mask, 1)
            view_gp_mask = torch.unsqueeze(view_gp_mask, -1)
            batch_size = nR
            num_channels = fdim  ###### remember to add the depth dim
            #
            # # view_gp_mask = tf.tile(view_gp_mask, [batch_size, 1, 1, self.gp_z, num_channels])
            # # view_gp_mask = tf.tile(view_gp_mask, [int(self.batch_size / self.patch_num),
            # #                                       self.patch_num, 1, 1, num_channels])
            view_gp_mask = view_gp_mask.repeat(int(self.batch_size / self.patch_num),
                                               self.patch_num, 1, 1, num_channels)
            view_gp_mask = view_gp_mask.to(torch.float32)
            # print("view_gp_mask",view_gp_mask.size())
            self.view_gp_mask = view_gp_mask.to(self.device)

            # view1_ic = self.World2Image('view1', wldcoords)
            # view2_ic = self.World2Image('view2', wldcoords)
            # view3_ic = self.World2Image('view3', wldcoords)

            for i in range(height_len):
                # print("wldcoords[:,:,i]",wldcoords[:,:,i])
                view_ic = self.World2Image(view, wldcoords[:, :, i])
                # print("view_ic",view_ic)
                view_ic = torch.transpose(view_ic, 0, 1)
                # print("view_ic",view_ic)
                # print(view_ic.size())
                # # normalization:
                view_ic[0:1, :] = view_ic[0:1, :] * rsz_w
                # print("view_ic[0:1, :]",view_ic[0:1, :])
                view_ic[1:2, :] = view_ic[1:2, :] * rsz_h
                view_ic[2:3, :] = view_ic[2:3, :]  # /400

                self.proj_view.append(torch.cat(
                    [view_ic[0:1, :], view_ic[1:2, :], view_ic[2:3, :]], axis=0).to(self.device))
            # print("self.proj_view", self.proj_view[0].shape)
            # print("self.proj_view",self.proj_view[0])

    def proj_splat(self, inputs):
        # print("inputs", inputs.requires_grad)
        # print(" inputs", inputs.grad)
        self.Ibilin = []
        # print(len(self.proj_view), len(self.proj_view))
        # print("inputs",inputs.shape)
        for i in range(len(self.proj_view)):
            # with tf.compat.v1.variable_scope('ProjSplat'):
            nR = inputs.size()[0]
            fh = inputs.size()[1]
            fw = inputs.size()[2]
            fdim = inputs.size()[3]
            # nR, fh, fw, fdim = self.tf_static_shape(inputs)
            nV = self.proj_view[i].size()[1]
            # print("nV",nV)

            im_p = self.proj_view[i]
            im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]

            B_im_x = torch.clamp(im_x, 0, fw - 1)
            B_im_y = torch.clamp(im_y, 0, fh - 1)

            # im_x0 = tf.cast(tf.floor(im_x), 'int32')
            B_im_x0 = torch.floor(B_im_x).to(torch.int32).to(self.device)
            B_im_x1 = B_im_x0 + 1
            B_im_x1 = torch.clamp(B_im_x1, 0, fw - 1)

            B_im_y0 = torch.floor(B_im_y).to(torch.int32).to(self.device)
            B_im_y1 = B_im_y0 + 1
            B_im_y1 = torch.clamp(B_im_y1, 0, fh - 1)

            B_im_x0_f, B_im_x1_f = B_im_x0.to(torch.float32), B_im_x1.to(torch.float32)
            B_im_y0_f, B_im_y1_f = B_im_y0.to(torch.float32), B_im_y1.to(torch.float32)

            B_ind_grid = torch.arange(0, nR).to(self.device)
            B_ind_grid = B_ind_grid.unsqueeze_(1)
            B_im_ind = B_ind_grid.repeat(1, nV)

            def _get_gather_inds(x, y):
                temp = torch.reshape(torch.stack([B_im_ind, y, x], axis=2), [-1, 3]).to(self.device)
                # print("temp", temp.requires_grad)
                # print(" temp", temp.grad)
                return temp

            # Gather  values
            B_Ia = self.gather_nd(inputs, _get_gather_inds(B_im_x0, B_im_y0))
            B_Ib = self.gather_nd(inputs, _get_gather_inds(B_im_x0, B_im_y1))
            B_Ic = self.gather_nd(inputs, _get_gather_inds(B_im_x1, B_im_y0))
            B_Id = self.gather_nd(inputs, _get_gather_inds(B_im_x1, B_im_y1))

            # Calculate bilinear weights
            B_wa = (B_im_x1_f - B_im_x) * (B_im_y1_f - B_im_y)
            # print("wa.size()",wa.size())
            B_wb = (B_im_x1_f - B_im_x) * (B_im_y - B_im_y0_f)
            B_wc = (B_im_x - B_im_x0_f) * (B_im_y1_f - B_im_y)
            B_wd = (B_im_x - B_im_x0_f) * (B_im_y - B_im_y0_f)
            B_wa = torch.reshape(B_wa, [-1, 1])
            B_wb = torch.reshape(B_wb, [-1, 1])
            B_wc = torch.reshape(B_wc, [-1, 1])
            B_wd = torch.reshape(B_wd, [-1, 1])
            Bilinear_result = (B_wa * B_Ia + B_wb * B_Ib + B_wc * B_Ic + B_wd * B_Id)
            Bilinear_result = torch.reshape(Bilinear_result, [int(self.batch_size / self.patch_num), self.patch_num,
                                                              self.gp_y, self.gp_x, fdim])
            # add a mask:
            self.Ibilin.append(torch.mul(Bilinear_result, self.view_gp_mask))
            # self.Ibilin.append(Bilinear_result)  # no mask needed.
        # result = torch.stack(self.Ibilin)
        return torch.stack(self.Ibilin), self.view_gp_mask

    # def gather_nd(self, x, indices):
    #     x = x.detach()
    #     newshape = indices.shape[:-1] + x.shape[indices.shape[-1]:]
    #     # print(newshape)
    #     indices = indices.view(-1, indices.shape[-1]).tolist()
    #     # print(len(indices))
    #     out = torch.cat([x.__getitem__(tuple(i)) for i in indices])
    #     # print("out", out.requires_grad)
    #     # print(" out", out.grad)
    #     return out.reshape(newshape)

    def gather_nd(self, x, coords):
        x = x.contiguous()
        # print("x",x.size())
        inds = coords.type(torch.float32).mv(torch.FloatTensor([0, x.size()[2], 1]).to(self.device)).type(torch.int64)
        x_gather = torch.index_select(x.view(-1, x.size()[-1]), 0, inds)
        return x_gather
