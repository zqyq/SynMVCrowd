# from sklearn import feature_extraction
import json
import os
import random

import cv2
# import sys
import numpy as np
import torch
from scipy.stats import multivariate_normal
from torch.nn import functional as F
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

from scripts.vis import show_tensor_images  # noqa


# import h5py
# from scipy import ndimage
# import scipy
# import scipy.ndimage
# import scipy.io as sio


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, num_cam=3, frames=200, scale=100, **kwargs):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce
        self.scale = scale
        self.base = base
        self.train = train
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        # CVCS dataset parameters:
        data_file = '/mnt/d/Datasets/LCVCS'
        label_file = '/mnt/d/Datasets/LCVCS'
        self.transform = transform

        # read images
        if train:
            self.file_path = os.path.join(data_file, 'train')
            self.label_file_path = os.path.join(label_file, 'train')

            # self.depthMap_file_path = depthMap_file + 'train/'
        else:
            self.file_path = os.path.join(data_file, 'val')
            self.label_file_path = os.path.join(label_file, 'val')

            # self.depthMap_file_path = depthMap_file + 'val/'

        self.gt_fpath = self.file_path

        self.cropped_size = [200, 200]
        self.patch_num = 3

        self.batch_size = 1

        self.num_cam = num_cam
        self.view_size = self.num_cam

        self.r = 5
        # 2 # 0.5m/pixel
        # 10: 0.1m/pixel
        # 5: 0.2m/pixel

        self.a = 5
        self.b = 5
        # cropped_size, r, a, b, patch_num

        random.seed(14)

        # list M scenes:
        self.scene_name_list = sorted(os.listdir(self.file_path))  # 2scenes model training
        self.nb_scenes = len(self.scene_name_list)
        #
        # if train:
        #     self.nb_samplings = 5 # int(nb_frames/view_size/batch_size + 1)#*3 for epochFixed *3; for epochFixed2, not.
        # else:
        #     self.nb_samplings = 1  # int(nb_frames/view_size/batch_size + 1)
        self.nb_samplings = 5
        self.frames = frames

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]

        # only use head labels:
        self.img_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        # self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        pass

    @staticmethod
    def read_json_frame(coords_info, cropped_size, r, a, b, patch_num):

        # set the wld map paras
        wld_map_paras = coords_info['wld_map_paras']
        s, r0, a0, b0, h4, w4, d_delta, d_mean, w_min, h_min = wld_map_paras  # old 640*480 labels

        # reconstruct wld_map_paras:
        w_max = (4 * w4 - 2 * a0) / r0 + w_min
        h_max = (4 * h4 - 2 * b0) / r0 + h_min

        # actual size:
        w_actual = int((w_max - w_min + 2 * a)) * r
        h_actual = int((h_max - h_min + 2 * b)) * r

        h_actual = int(max(h_actual, cropped_size[0] + patch_num))
        w_actual = int(max(w_actual, cropped_size[1] + patch_num))

        # create patch_num patches' bbox coordinates:
        ## h: 0~h_actual-cropped_size[0], w: 0~w_actual-cropped_size[1]
        h_range = range(0, h_actual - cropped_size[0] + 1)
        w_range = range(0, w_actual - cropped_size[1] + 1)

        h_random = random.sample(h_range, k=patch_num)
        w_random = random.sample(w_range, k=patch_num)
        hw_random = np.asarray([h_random, w_random])

        wld_map_paras = [r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max]

        # get people 2D and 3D coords:
        coords = coords_info['image_info']

        # coords_2d_all = [] #np.zeros((1, 2))
        coords_3d_id_all = []  # np.zeros((1, 3))

        # id = 0
        for point in coords:
            id = point['idx']
            coords_3d_id = point['world'] + [id]
            coords_3d_id_all.append(coords_3d_id)

        coords_3d_id_all = np.asarray(coords_3d_id_all, dtype='float32')

        # form the para list:
        return coords_3d_id_all, wld_map_paras, hw_random

    @staticmethod
    def read_json_view(coords_info):

        # # get camera location
        camera_loc = np.asarray(coords_info['camera']['location'])
        #
        # # get camera rotation
        # rotation = np.asarray(coords_info['camera']['rotation'])

        # get camera matrix
        cameraMatrix = np.asarray(coords_info['cameraMatrix'])
        fx = cameraMatrix[0][0]
        fy = cameraMatrix[1][1]
        u = cameraMatrix[0][2]
        v = cameraMatrix[1][2]

        # get camera matrix:
        distCoeffs = coords_info['distCoeffs']

        rvec = coords_info['rvec']
        tvec = coords_info['tvec']

        camera_paras = [fx] + [fy] + [u] + [v] + distCoeffs + rvec + tvec
        camera_paras = np.asarray(camera_paras)

        # get people 2D and 3D coords:
        coords = coords_info['image_info']

        coords_2d_all = []
        coords_3d_id_all = []

        for point in coords:
            id = point['idx']

            coords_2d0 = point['pixel']
            if coords_2d0 == None:
                continue
            coords_2d = [coords_2d0[1] / 1920.0, coords_2d0[0] / 1080.0]

            coords_3d_id = point['world'] + [id]

            coords_2d_all.append(coords_2d)
            coords_3d_id_all.append(coords_3d_id)
            # id = id + 1

        # form the para list:
        return coords_3d_id_all, coords_2d_all, camera_paras, camera_loc

    @staticmethod
    def density_map_creation(map_gt, w, h):
        if map_gt.size == 0:
            img_map_gt_i = np.zeros((h, w))
            density_map = np.asarray(img_map_gt_i).astype('f')

        else:

            map_gt = np.asarray(map_gt)
            img_dmap = np.zeros((h, w))

            x = (map_gt[:, 0] * w).astype('int')
            y = (map_gt[:, 1] * h).astype('int')
            img_dmap[y, x] = 1

            density_map = np.asarray(img_dmap).astype('f')

        return density_map

    # @staticmethod
    def GP_density_map_creation(self, wld_coords, crop_size, wld_map_paras, hw_random, cam_location):
        # we need to resize the images
        h = int(crop_size[0])
        w = int(crop_size[1])

        r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max = wld_map_paras

        h_actual, w_actual = int(h_actual), int(w_actual)
        patch_num = hw_random.shape[1]

        whole_plane_map = np.zeros((h_actual, w_actual))
        if wld_coords.size == 0:
            print('wld_coords.size == 0!!!!!!!!')
            img_map_gt_i = np.zeros((h_actual, w_actual))
            GP_density_map = []
            for p in range(patch_num):
                hw = hw_random[:, p]
                GP_density_map_i = img_map_gt_i[hw[0]:hw[0] + h, hw[1]:hw[1] + w]
                GP_density_map.append(GP_density_map_i)
            GP_density_map = np.asarray(GP_density_map)

        else:
            wld_coords_transed = np.zeros(wld_coords.shape)
            wld_coords_transed[:, 0] = (wld_coords[:, 0] - w_min + a) * r
            wld_coords_transed[:, 1] = (wld_coords[:, 1] - h_min + b) * r
            wld_coords_transed = wld_coords_transed.astype('int')

            cl = cam_location.copy()

            for i in range(self.view_size):
                cur_cl = cl[i]
                cur_cl[0] = (cam_location[i][0] - w_min + a) * r
                cur_cl[1] = (cam_location[i][1] - h_min + b) * r
                # cl = cl.astype('int')

            assert min(wld_coords_transed[:, 0]) >= 0 and max(wld_coords_transed[:, 0]) < w_actual
            assert min(wld_coords_transed[:, 1]) >= 0 and max(wld_coords_transed[:, 1]) < h_actual

            whole_plane_map[wld_coords_transed[:, 1], wld_coords_transed[:, 0]] = 1

            GP_density_map_0 = whole_plane_map

            GP_density_map = []
            for p in range(patch_num):
                hw = hw_random[:, p]
                GP_density_map_i = GP_density_map_0[hw[0]:hw[0] + h, hw[1]:hw[1] + w]
                GP_density_map.append(GP_density_map_i)
            GP_density_map = np.asarray(GP_density_map)

        cl = np.asarray(cl)
        cl_pats = []
        for i in range(patch_num):
            # 注意camera坐标是x， y
            cur = cl.copy()
            cur[:, 0] = cl[:, 0] - hw_random[:, i][1]
            cur[:, 1] = cl[:, 1] - hw_random[:, i][0]
            cl_pats.append(cur)

        return GP_density_map, cl_pats, whole_plane_map

    @staticmethod
    def id_unique(coords_array):
        # intilize a null list
        coords_array = np.asarray(coords_array)

        unique_list = [[-1, -1, -1, -1]]

        id = coords_array[:, -1]
        n = id.shape[0]
        # traverse for all elements

        for i in range(n):
            id_i = id[i]
            coords_array_i = coords_array[i]

            id_current_unique_list = list(np.asarray(unique_list)[:, -1])
            if id_i not in id_current_unique_list:
                unique_list.append(coords_array_i)

        unique_list = unique_list[1:]
        return unique_list

    @staticmethod
    def id_diff(coords_arrayA, coords_arrayB):  # coords_arrayA is larger
        coords_arrayA = np.asarray(coords_arrayA)
        coords_arrayB = np.asarray(coords_arrayB)

        unique_list = []  # [[-1, -1, -1, -1]]

        if coords_arrayB.size == 0:
            unique_list = coords_arrayA
        else:

            idA = coords_arrayA[:, -1]
            idB = coords_arrayB[:, -1]

            n = idA.shape[0]
            # traverse for all elements

            for i in range(n):
                id_i = idA[i]
                coords_array_i = coords_arrayA[i]

                # id_current_unique_list = list(np.asarray(unique_list)[:, -1])
                if id_i not in idB:  # id_current_unique_list:
                    unique_list.append(coords_array_i)

            # unique_list = unique_list[1:]
            unique_list = np.asarray(unique_list)
        return unique_list

    def get_one_sample(self):
        file_path='/mnt/d/yunfei/Daijie_code/Baseline_OT/IJCV_Results/Multi-view detection'
        img_list = os.listdir(file_path)
        img_list.sort()
        if 'scene78' in img_list[0]:
            data_root = '/mnt/d/Datasets/LCVCS/val/scene_78'
        else:
            data_root = '/mnt/d/Datasets/LCVCS/test/scene_1'
        # example: scene78_frame131_view5.jpg
        img_views = []
        camera_paras = []
        wld_coords = []
        cam_locations = []
        single_view_dmaps = []

        for i, img_dir in enumerate(img_list):
            scene = img_dir.split('_')[0].split('scene')[-1]
            frame = img_dir.split('_')[1].split('frame')[-1]
            view = img_dir.split('_')[2].split('view')[-1].split('.')[0]
            print(f'scene{scene} f{frame} v{view}: ', end='')
            label_dir = os.path.join(data_root, frame, 'jsons', f'{view}.json')
            # label_dirs.append(label_dir)
            if i == 0:
                label_path_j = label_dir
                with open(label_path_j, 'r') as data_file:
                    coords_info_frame = json.load(data_file)
                coords_3d_id_all_frame, wld_map_paras_frame, hw_random = self.read_json_frame(coords_info_frame,
                                                                                              self.cropped_size,
                                                                                              self.r, self.a, self.b,
                                                                                              self.patch_num)
                wld_map_paras_frame = np.asarray(wld_map_paras_frame)
                hw_random = np.asarray(hw_random)
                wld_map_paras = wld_map_paras_frame
            # read images
            img_path = os.path.join(file_path, img_dir)
            img = cv2.imread(img_path)
            img = img[:, :, (2, 1, 0)]  # BGR -> RGB
            img = img.astype('float32')
            img = img / 255.0
            img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)
            img = torch.nn.functional.interpolate(img[None, :, :, :], [720, 1280])
            img_views.append(img)
            # show_tensor_images(img, num_images=1, size=(1, 3), title=f'scene{scene} f{frame} v{view}')
            # read labels
            with open(label_dir, 'r') as data_file:
                coords_info = json.load(data_file)
            coords, coords_2d, paras, cam_loc = self.read_json_view(coords_info)
            cam_loc = np.asarray(cam_loc)
            coords_2d = np.asarray(coords_2d)
            single_view_dmaps_i = self.density_map_creation(coords_2d, w=640, h=360)
            single_view_dmaps_i = np.expand_dims(single_view_dmaps_i, axis=0)
            single_view_dmaps_i = torch.from_numpy(single_view_dmaps_i)
            single_view_dmaps.append(single_view_dmaps_i)
            cam_locations.append(cam_loc)
            camera_paras.append(paras)
            wld_coords = wld_coords + coords

        # get the wld coords
        wld_coords = self.id_unique(wld_coords)
        wld_coords = np.asarray(wld_coords)

        # create the map_gts, instead of density maps
        GP_point_map, cam_location, whole_pl_map = self.GP_density_map_creation(wld_coords,
                                                                                self.cropped_size,
                                                                                wld_map_paras,
                                                                                hw_random,
                                                                                cam_location=cam_locations)

        img_views=torch.cat(img_views, dim=0)
        camera_paras = np.asarray(camera_paras)


        cam_location = [c / self.cropped_size[-2] for c in cam_location]
        cam_location = torch.tensor(np.array(cam_location))

        single_view_dmaps = np.asarray(single_view_dmaps)
        single_view_dmaps = torch.from_numpy(single_view_dmaps)
        with torch.no_grad():
            heatmap = torch.from_numpy(whole_pl_map)[None, None].float()
            human_count = heatmap.sum()
            heatmap = F.conv2d(torch.tensor(heatmap), self.map_kernel.float(),
                               padding=int((self.map_kernel.shape[-1] - 1) / 2))
            heatmap_sum = heatmap.sum()
            # norm
            heatmap = heatmap / heatmap_sum * human_count
            GP_density_map = heatmap.squeeze(0).numpy() * self.scale
        # all transform to tensor
        GP_density_map = torch.from_numpy(GP_density_map).float()
        whole_pl_map = torch.from_numpy(whole_pl_map).float()
        hw_random=torch.from_numpy(hw_random).float()
        camera_paras=torch.from_numpy(camera_paras).float()
        wld_map_paras=torch.from_numpy(wld_map_paras).float()
        whole_pl_map=whole_pl_map.float()
        GP_point_map=torch.from_numpy(GP_point_map).float()
        return img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_point_map, GP_density_map, cam_location, whole_pl_map
        pass

    def __getitem__(self, index):
        if self.train:
            scene_index = int(index / (self.nb_samplings * 10))
        else:
            scene_index = int(index / (self.nb_samplings * self.frames))
        # selection_index = int((index - scene_index * self.nb_samplings * 100) / 100)

        scene_i = self.scene_name_list[scene_index]
        # if scene_i == 'scene_24':
        #     frame_index = (index - scene_index * self.nb_samplings * 20) % 92
        # else:
        #     frame_index = (index - scene_index * self.nb_samplings * 20) % 100

        scene_path = os.path.join(self.file_path, scene_i)
        scene_path_label = os.path.join(self.label_file_path, scene_i)

        # list N frames:
        frame_name_list = sorted([int(num) for num in os.listdir(scene_path_label)])
        # select views first:
        # frame_j = frame_name_list[frame_index]

        if self.train:
            frame_j = random.sample(frame_name_list, k=1)[0]
        else:
            frame_index = (index - scene_index * self.nb_samplings * self.frames) % self.frames
            frame_j = frame_name_list[frame_index]

        label_path_j = os.path.join(self.label_file_path, scene_i, str(frame_j), 'jsons/')
        label_path_list_j = os.listdir(label_path_j)
        label_path_list_sampling = random.sample(label_path_list_j, k=self.view_size)

        frame_path = os.path.join(scene_path, str(frame_j))
        img_path = frame_path + '/jpgs/'
        label_path = os.path.join(self.label_file_path, scene_i, str(frame_j), 'jsons/')

        label_name_j = label_path_list_j[0]
        label_path_name_j = os.path.join(label_path, label_name_j)

        with open(label_path_name_j, 'r') as data_file:
            coords_info_frame = json.load(data_file)
        # coords_3d_id_all_frame = read_json_all(coords_info_frame)
        coords_3d_id_all_frame, wld_map_paras_frame, hw_random = self.read_json_frame(coords_info_frame,
                                                                                      self.cropped_size,
                                                                                      self.r, self.a, self.b,
                                                                                      self.patch_num)
        wld_map_paras_frame = np.asarray(wld_map_paras_frame)
        hw_random = np.asarray(hw_random)
        wld_map_paras = wld_map_paras_frame

        img_views = []
        camera_paras = []
        wld_coords = []
        cam_locations = []
        single_view_dmaps = []

        def exist_none_img(img_name_list):
            for p in img_name_list:
                label_name = p
                img_name = label_name[0:-5] + '.jpg'
                # read images
                img_path_name = os.path.join(img_path, img_name)
                img = cv2.imread(img_path_name)
                if img is None:
                    return True
            return False

        flag = exist_none_img(label_path_list_sampling)

        if flag:
            while flag:
                print('exist_none')
                label_path_list_sampling = random.sample(label_path_list_j, k=self.view_size)
                flag = exist_none_img(label_path_list_sampling)

        for p in label_path_list_sampling:
            label_name = p
            img_name = label_name[0:-5] + '.jpg'

            # read images
            img_path_name = os.path.join(img_path, img_name)

            # if not self.train:
            # print(img_path_name)

            img = cv2.imread(img_path_name)
            if img is None:
                raise ValueError('img is None')

            img = img[:, :, (2, 1, 0)]  # BGR -> RGB
            img = img.astype('float32')

            img = img / 255.0
            img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
            img_views.append(img)

            # read labels
            label_path_name = os.path.join(label_path, label_name)
            with open(label_path_name, 'r') as data_file:
                coords_info = json.load(data_file)
            coords, coords_2d, paras, cam_loc = self.read_json_view(coords_info)
            cam_locations.append(cam_loc)

            coords_2d = np.asarray(coords_2d)
            single_view_dmaps_i = self.density_map_creation(coords_2d, w=640, h=360)
            single_view_dmaps_i = np.expand_dims(single_view_dmaps_i, axis=0)
            single_view_dmaps.append(single_view_dmaps_i)

            # form the camera paras list
            camera_paras.append(paras)

            # get the wld_coords:
            wld_coords = wld_coords + coords

        # get the wld coords
        wld_coords = self.id_unique(wld_coords)
        wld_coords = np.asarray(wld_coords)

        # create the map_gts, instead of density maps
        GP_point_map, cam_location, whole_pl_map = self.GP_density_map_creation(wld_coords,
                                                                                self.cropped_size,
                                                                                wld_map_paras,
                                                                                hw_random,
                                                                                cam_location=cam_locations)

        img_views = np.asarray(img_views)
        camera_paras = np.asarray(camera_paras)

        img_views = np.transpose(img_views, (0, 3, 1, 2))
        img_views = torch.from_numpy(img_views)
        img_views = torch.nn.functional.interpolate(img_views, [720, 1280])

        cam_location = [c / self.cropped_size[-2] for c in cam_location]
        cam_location = torch.tensor(np.array(cam_location))

        single_view_dmaps = np.asarray(single_view_dmaps)

        # GP_density_map = []
        if self.train:
            with torch.no_grad():
                heatmap = torch.from_numpy(GP_point_map)[:, None].float()
                human_count = heatmap.sum()
                heatmap = F.conv2d(heatmap, self.map_kernel.float(),
                                   padding=int((self.map_kernel.shape[-1] - 1) / 2))
                heatmap_sum = heatmap.sum()
                # norm
                heatmap = heatmap / heatmap_sum * human_count
                GP_density_map = heatmap.squeeze().numpy() * self.scale
            return img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_point_map, GP_density_map, cam_location, whole_pl_map
        else:
            with torch.no_grad():
                heatmap = torch.from_numpy(whole_pl_map)[None, None].float()
                human_count = heatmap.sum()
                heatmap = F.conv2d(torch.tensor(heatmap), self.map_kernel.float(),
                                   padding=int((self.map_kernel.shape[-1] - 1) / 2))
                heatmap_sum = heatmap.sum()
                # norm
                heatmap = heatmap / heatmap_sum * human_count
                GP_density_map = heatmap.squeeze(0).numpy() * self.scale
            return img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_point_map, GP_density_map, cam_location, whole_pl_map

    def __len__(self):
        if self.train:
            len_num = self.nb_scenes * 10 * self.nb_samplings
        else:
            len_num = self.nb_scenes * self.frames * self.nb_samplings

        return len_num


def test():
    from scripts.multiview_detector.datasets.lcvcs.Wildtrack import Wildtrack
    trainset = frameDataset(Wildtrack(os.path.expanduser('/mnt/d/Datasets/Wildtrack')), train=True)
    testset = frameDataset(Wildtrack(os.path.expanduser('/mnt/d/Datasets/Wildtrack')), train=False)
    data = testset.get_one_sample('/mnt/d/yunfei/Daijie_code/Baseline_OT/IJCV_Results/Multi-view detection')

    for i, (img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_point_map, GP_density_map,
            cam_location_) in enumerate(trainset):
        if i % 10 == 0:
            print(f"Sample {i}:")
            print(f"Image Views Shape: {img_views.shape}")
            print(f"Single View Dmaps Shape: {single_view_dmaps.shape}")
            print(f"Camera Paras Shape: {camera_paras.shape}")
            print(f"WLD Map Paras: {wld_map_paras}")
            print(f"HW Random: {hw_random}")
            print(f"GP Point Map Shape: {GP_point_map.shape}")
            print(f"GP Density Map Shape: {GP_density_map.shape}")
            print(f"Cam Location Shape: {cam_location_.shape}\n")
            break
    for i, (img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_point_map, GP_density_map,
            cam_location_, whole_pl_map) in enumerate(testset):
        if i % 10 == 0:
            print(f"Sample {i}:")
            print(f"Image Views Shape: {img_views.shape}")
            print(f"Single View Dmaps Shape: {single_view_dmaps.shape}")
            print(f"Camera Paras Shape: {camera_paras.shape}")
            print(f"WLD Map Paras: {wld_map_paras}")
            print(f"HW Random: {hw_random}")
            print(f"GP Point Map Shape: {GP_point_map.shape}")
            print(f"GP Density Map Shape: {GP_density_map.shape}")
            print(f"Cam Location Shape: {cam_location_.shape}")
            print(f"Whole Plane Map Shape: {whole_pl_map.shape}\n")
            break


if __name__ == '__main__':
    test()
