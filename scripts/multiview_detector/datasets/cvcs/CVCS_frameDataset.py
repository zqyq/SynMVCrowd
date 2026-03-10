# import os
# import json
# from scipy.stats import multivariate_normal
# from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor

from multiview_detector.loss.projasa import *
from multiview_detector.utils.projection import *
# import h5py
# from scipy import ndimage
import scipy
# import scipy.ndimage
# import scipy.io as sio

# from sklearn import feature_extraction
import json
import cv2
import random
from scipy.stats import multivariate_normal
from datetime import datetime
import collections

seed_Set = [12, 24, 8, 89, 335]
frame_label_list = [['36', '28', '45', '39', '42', '11', '17', '7', '97', '29'],
                    ['90', '14', '36', '64', '73', '3', '99', '93', '47', '31',
                     '58', '81', '33', '67', '92', '55', '91', '77', '4', '22'],
                    ['12', '8', '38', '24', '67']]


class frameDataset(VisionDataset):
    def __init__(self, base, rate, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True, test_num=1, scale=10):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.scale = scale
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce
        self.train = train
        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        # CVCS dataset parameters:
        data_file = '/mnt/d/common/Datasets/CVCS'
        # label_file = '/mnt/d/common/Datasets/CVCS/labels/100frames_labels_reproduce_640_480_CVCS/100frames_labels_reproduce_640_480_CVCS/'
        label_file = '/mnt/d/common/Datasets/CVCS/labels/100frames_labels_reproduce_640_480_CVCS/100frames_labels_reproduce_640_480_CVCS'
        # depthMap_file = '/opt/visal/home/qzhang364/Cross_View-Cross_Scene/dataset/' \
        #                 'Cross_view_cross_sscene_data/smallscene3_full/100frames_depthMaps_reproduce_640_480/'
        # data_file = '/home/zq/codes/SSHFS/URL_my/Cross_View-Cross_Scene/dataset/Cross_view_cross_sscene_data/smallscene3_full/100frames_smallSize/'
        # label_file = '/home/zq/codes/SSHFS/URL_my/Cross_View-Cross_Scene/dataset/Cross_view_cross_sscene_data/smallscene3_full/100frames_labels_reproduce_640_480/'
        # depthMap_file = '/home/zq/codes/SSHFS/URL_my/Cross_View-Cross_Scene/dataset/Cross_view_cross_sscene_data/smallscene3_full/100frames_depthMaps_reproduce_640_480/'

        # read images
        if train:
            # self.file_path = data_file + 'train/'
            # self.label_file_path = label_file + 'train/'
            # self.depthMap_file_path = depthMap_file + 'train/'
            self.file_path = os.path.join(data_file, 'train')
            self.label_file_path = os.path.join(label_file, 'train')
        else:
            # self.file_path = data_file + 'val/'  #
            # self.label_file_path = label_file + 'val/'
            # self.depthMap_file_path = depthMap_file + 'val/'
            self.file_path = os.path.join(data_file, 'val')
            self.label_file_path = os.path.join(label_file, 'val')
        self.gt_fpath = self.file_path

        # ind_scene = 0
        # nb_batch_used = 0
        # nb_view_used = 0

        # wld_h = int(720/2) #480  #360
        # wld_w = int(640/2) #640

        self.batch_size = 1

        # if train:
        #     self.output_size = [160, 180]
        # else:
        #     self.output_size = [200, 180]

        self.output_size = [200, 200]

        self.view_size = 5
        self.rate = rate
        self.map_sigma = 2
        self.num_cam = self.view_size
        self.is_train = train

        self.patch_num = 1  # 5

        self.r = 2
        # 2 # 0.5m/pixel
        # 10: 0.1m/pixel
        # 5: 0.2m/pixel

        self.a = 5
        self.b = 5
        # cropped_size, r, a, b, patch_num

        img_views_list = []
        camera_paras_list = []
        wld_map_paras_list = []
        hw_random_list = []
        depthMap_views_list = []

        GP_list = []
        single_view_dmaps_list = []

        random.seed(14)

        # list M scenes:
        self.scene_name_list = os.listdir(self.file_path)  # 2scenes model training
        # self.scene_name_list = self.scene_name_list[0:1]

        self.nb_scenes = len(self.scene_name_list)

        if train:
            self.nb_samplings = 5  # int(nb_frames/view_size/batch_size + 1)#*3 for epochFixed *3; for epochFixed2, not.
        else:
            self.nb_samplings = 1  # int(nb_frames/view_size/batch_size + 1)
            # self.nb_samplings = 21

        # if train:
        #     frame_range = range(0, int(self.num_frame * train_ratio))
        # else:
        #     frame_range = range(int(self.num_frame * train_ratio), self.num_frame)
        #
        # self.img_fpaths = self.base.get_image_fpaths(frame_range)
        # self.map_gt = {}
        # self.imgs_head_foot_gt = {}
        # self.download(frame_range)
        #
        # self.gt_fpath = os.path.join(self.root, 'gt.txt')
        # if not os.path.exists(self.gt_fpath) or force_download:
        #     self.prepare_gt()

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
        # self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        # self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        # self.img_kernel[1, 1] = torch.from_numpy(img_kernel)

        # only use head labels:
        self.img_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        # self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        pass

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                    [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                    [[] for _ in range(self.num_cam)], \
                    [[] for _ in range(self.num_cam)]
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in range(self.num_cam):
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def read_json_frame(self, coords_info, cropped_size, r, a, b, patch_num):

        # set the wld map paras
        wld_map_paras = coords_info['wld_map_paras']
        s, r0, a0, b0, h4, w4, d_delta, d_mean, w_min, h_min = wld_map_paras  # old 640*480 labels

        # reconstruct wld_map_paras:
        # a = (w - (w_max - w_min) * r) / 2
        # b = (h - (h_max - h_min) * r) / 2
        w_max = (4 * w4 - 2 * a0) / r0 + w_min
        h_max = (4 * h4 - 2 * b0) / r0 + h_min

        # r = int(4 / 2)
        # a = 5
        # b = 5
        # patch_num = 5
        # cropped_size = [180, 160]

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
        # h_random[0] = h_actual
        w_random = random.sample(w_range, k=patch_num)
        # w_random[0] = w_actual
        hw_random = np.asarray([h_random, w_random])

        # h_random = np.linspace(0, h_actual-1, h_actual)
        # w_random = np.linspace(0, w_actual-1, h_actual)
        # h_coordinates, w_coordinates = np.meshgrid(h_random, w_random)

        wld_map_paras = [r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max]

        coords = coords_info['image_info']

        # coords_2d_all = [] #np.zeros((1, 2))
        coords_3d_id_all = []  # np.zeros((1, 3))

        # id = 0
        for point in coords:
            id = point['idx']
            coords_3d_id = point['world'] + [id]
            coords_3d_id_all.append(coords_3d_id)

        coords_3d_id_all = np.asarray(coords_3d_id_all, dtype='float32')

        return coords_3d_id_all, wld_map_paras, hw_random

    def read_json_view(self, coords_info):

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

        coords_2d_all = []  # np.zeros((1, 2))
        coords_3d_id_all = []  # np.zeros((1, 3))

        for point in coords:
            id = point['idx']

            coords_2d0 = point['pixel']
            if coords_2d0 == None:
                continue
            coords_2d = [coords_2d0[1] / 1920.0, coords_2d0[0] / 1080.0]

            coords_3d_id = point['world'] + [id]

            # coords_2d_all = np.concatenate((coords_2d_all, np.expand_dims(coords_2d, axis=0)), axis=0)
            # coords_3d_all = np.concatenate((coords_3d_all, np.expand_dims(coords_3d, axis=0)), axis=0)
            coords_2d_all.append(coords_2d)
            coords_3d_id_all.append(coords_3d_id)
            # id = id + 1

        # coords_2d_all = coords_2d_all[1:, :].astype('float32')
        # coords_3d_all = coords_3d_all[1:, :].astype('float32')
        # coords_2d_all = np.asarray(coords_2d_all, dtype='float32')
        # coords_3d_all = np.asarray(coords_3d_all, dtype='float32')

        # form the para list:
        return coords_3d_id_all, coords_2d_all, camera_paras

    def density_map_creation(self, pmap, w, h):
        if pmap.size == 0:
            img_pmap_i = np.zeros((h, w))
            density_map = np.asarray(img_pmap_i).astype('f')

        else:
            density_map = []
            img_id_all = pmap[:, 0].shape[0]

            pmap = np.asarray(pmap)
            img_dmap = np.zeros((h, w))

            x = (pmap[:, 0] * w).astype('int')
            y = (pmap[:, 1] * h).astype('int')
            img_dmap[y, x] = 1

            # count_i = 0

            # for i in range(img_id_all):
            #     pmap_i = pmap[i, :]
            #
            #     x = int(pmap_i[0]*w)
            #     y = int(pmap_i[1]*h)
            #
            #     sigma = [5, 5]
            #     # sigmax = pmap_i[2]
            #     # sigmay = pmap_i[3]
            #     # sigma = [sigmax/2, sigmay/2]
            #
            #     img_pmap_i = np.zeros((h, w))
            #     if x>=w or x<0 or y>=h or y<0:
            #         continue
            #     else:
            #         count_i = count_i + 1
            #         img_pmap_i[y, x] = 1
            #
            # count = count_i
            density_map = np.asarray(img_dmap).astype('f')

        # plt.figure()
        # plt.imshow(img_dmap_i)
        # plt.savefig('GP.png')
        return density_map

    def GP_density_map_creation(self, wld_coords, output_size, wld_map_paras, hw_random):
        # we need to resize the images
        h = int(output_size[0])
        w = int(output_size[1])

        r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max = wld_map_paras

        h_actual, w_actual = int(h_actual), int(w_actual)
        patch_num = hw_random.shape[1]

        if wld_coords.size == 0:
            img_pmap_i = np.zeros((h_actual, w_actual))

            GP_density_map = []
            for p in range(patch_num):
                hw = hw_random[:, p]
                GP_density_map_i = img_pmap_i[hw[0]:hw[0] + h, hw[1]:hw[1] + w]
                GP_density_map.append(GP_density_map_i)
            GP_density_map = np.asarray(GP_density_map)

            GP_MAP = np.zeros(output_size)
            GP_map_counts = 0

        else:

            wld_coords_transed = np.zeros(wld_coords.shape)
            wld_coords_transed[:, 0] = (wld_coords[:, 0] - w_min + a) * r
            wld_coords_transed[:, 1] = (wld_coords[:, 1] - h_min + b) * r
            wld_coords_transed = wld_coords_transed.astype('int')

            # wld_coords_transed = wld_coords_transed[wld_coords_transed[:, 0] < w_actual, :]
            # wld_coords_transed = wld_coords_transed[wld_coords_transed[:, 0] >=0, :]
            # wld_coords_transed = wld_coords_transed[wld_coords_transed[:, 1] < h_actual, :]
            # wld_coords_transed = wld_coords_transed[wld_coords_transed[:, 1] >=0, :]

            assert min(wld_coords_transed[:, 0]) >= 0 and max(wld_coords_transed[:, 0]) < w_actual
            assert min(wld_coords_transed[:, 1]) >= 0 and max(wld_coords_transed[:, 1]) < h_actual

            # img_pmap = np.zeros((h_actual, w_actual, img_id_all))
            # img_pmap[wld_coords_transed[:, 1], wld_coords_transed[:, 0], range(img_id_all)] = 1

            img_id_all = wld_coords_transed[:, 0].shape[0]

            img_pmap = np.zeros((h_actual, w_actual))
            img_pmap[wld_coords_transed[:, 1], wld_coords_transed[:, 0]] = 1
            # plt.imshow(img_pmap)
            # plt.show()

            sg = 2
            sigma = [sg, sg]
            sg_size = 4 * sg

            dmap_i = np.zeros((2 * sg_size + 1, 2 * sg_size + 1))
            dmap_i[sg_size, sg_size] = 1
            dmap_i = scipy.ndimage.gaussian_filter(dmap_i, sigma, mode='reflect')
            # print(sum(dmap_i.flatten()))

            img_dmap = []  # np.zeros((h, w, img_id_all))
            for i in range(img_id_all):
                img_dmap_i = np.zeros((h_actual + 2 * sg_size, w_actual + 2 * sg_size))
                img_dmap_i[wld_coords_transed[i, 1]:(wld_coords_transed[i, 1] + 2 * sg_size + 1),
                wld_coords_transed[i, 0]:(wld_coords_transed[i, 0] + 2 * sg_size + 1)] = dmap_i
                img_dmap.append(img_dmap_i[sg_size:h_actual + sg_size, sg_size:w_actual + sg_size])
            img_dmap = np.asarray(img_dmap)
            GP_density_map_0 = np.sum(img_dmap, axis=0).astype('f')  # -1

            # patch_num = hw_random.shape[1]
            GP_density_map = []
            for p in range(patch_num):
                hw = hw_random[:, p]
                GP_density_map_i = GP_density_map_0[hw[0]:hw[0] + h, hw[1]:hw[1] + w]
                GP_density_map.append(GP_density_map_i)

            GP_density_map = np.asarray(GP_density_map)
        return GP_density_map

        #     GP_MAP = np.zeros(output_size)
        #     G_test = np.zeros((h_actual, w_actual))
        #     for i in range(len(wld_coords_transed)):
        #         cx = round(wld_coords_transed[i, 0] / w_actual * output_size[1])
        #         cy = round(wld_coords_transed[i, 1] / h_actual * output_size[0])
        #         map_counting(GP_MAP, (cx, cy), sigma=self.map_sigma)
        #         # cx = wld_coords_transed[i,0]
        #         # cy = wld_coords_transed[i,1]
        #         # map_counting(G_test, (cx, cy), sigma=self.map_sigma)
        #
        #     # plt.imshow(GP_MAP)
        #     # plt.show()
        #     # plt.imshow(G_test)
        #     # plt.show()
        #     GP_map_counts = round(GP_MAP.sum())
        # return GP_MAP, GP_map_counts

    def id_unique(self, coords_array):
        # intilize a null list
        coords_array = np.asarray(coords_array)
        # print("coords_array", coords_array.shape)
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

    def id_diff(self, coords_arrayA, coords_arrayB):  # coords_arrayA is larger
        coords_arrayA = np.asarray(coords_arrayA)
        coords_arrayB = np.asarray(coords_arrayB)

        unique_list = []  # [[-1, -1, -1, -1]]

        if coords_arrayB.size == 0:
            unique_list = coords_arrayA
        else:

            idA = coords_arrayA[:, -1]
            idB = coords_arrayB[:, -1]

            # print(idA)
            # print(idB)

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

    def mask_gt(self, GP_dmap):
        GP_dmap = torch.from_numpy(GP_dmap)
        Count_nonzero = torch.count_nonzero(GP_dmap).item()
        val_sort, _ = torch.sort(GP_dmap.reshape(-1), descending=True)
        top30_thr = val_sort[int(Count_nonzero * 0.3)]
        top50_thr = val_sort[int(Count_nonzero * 0.5)]
        top100_thr = val_sort[int(Count_nonzero * 1)]

        GP_top30_dmap = (GP_dmap > top30_thr).long()
        GP_top50_dmap = (GP_dmap > top50_thr).long()
        GP_top100_dmap = (GP_dmap > top100_thr).long()
        # plt.imshow(GP_dmap)
        # plt.show()
        # plt.imshow(GP_top30_dmap)
        # plt.show()
        # plt.imshow(GP_top50_dmap)
        # plt.show()
        # plt.imshow(GP_top100_dmap)
        # plt.show()
        return GP_top30_dmap, GP_top50_dmap, GP_top100_dmap

    def __getitem__(self, index):
        scene_index = int(index / (self.nb_samplings * 100))
        scene_i = self.scene_name_list[scene_index]
        # scene_i = 'scene_37'

        scene_path = os.path.join(self.file_path, scene_i)
        scene_path_label = os.path.join(self.label_file_path, scene_i)
        # list N frames:
        frame_index = int(index - scene_index * self.nb_samplings * 100) % 100  # 0-100
        label_flag = 1
        if self.rate == 0.1:
            if frame_index < int(self.rate * 100):
                frame_name_list = frame_label_list[0]
                label_flag = 1
            else:
                frame_index -= int(self.rate * 100)
                frame_name_list = os.listdir(scene_path_label)
                for i in range(len(frame_label_list[0])):
                    frame_name_list.remove(frame_label_list[0][i])
                label_flag = 0
        elif self.rate == 0.2:
            if frame_index < int(self.rate * 100):
                frame_name_list = frame_label_list[1]
                label_flag = 1
            else:
                frame_index -= int(self.rate * 100)
                frame_name_list = os.listdir(scene_path_label)
                for i in range(len(frame_label_list[1])):
                    frame_name_list.remove(frame_label_list[1][i])
                label_flag = 0
        elif self.rate == 0.05:
            if frame_index < int(self.rate * 100):
                frame_name_list = frame_label_list[2]
                label_flag = 1
            else:
                frame_index -= int(self.rate * 100)
                frame_name_list = os.listdir(scene_path_label)
                for i in range(len(frame_label_list[2])):
                    frame_name_list.remove(frame_label_list[2][i])
                label_flag = 0
        else:
            frame_name_list = os.listdir(scene_path_label)
        if scene_i == "scene_24":
            frame_name_list = sorted(frame_name_list)
            frame_name_list = frame_name_list[0:90]

        # select views first:
        frame_0 = '0'  # frame_name_list[0]
        # frame_path0 = os.path.join(scene_path, frame_0)
        # img_path0 = frame_path0 + '/jpgs/'
        label_path0 = os.path.join(self.label_file_path, scene_i, frame_0, 'json_paras/')
        label_path_list0 = os.listdir(label_path0)
        if self.train:
            label_path_list_sampling = random.sample(label_path_list0, k=self.view_size)
        else:
            label_path_list_sampling = label_path_list0[0:self.view_size]
        # print("--------------------")
        # print(scene_i)
        # print(label_path_list_sampling)
        # label_path_list_sampling = ['27.json']

        if frame_index < len(frame_name_list):
            frame_j = frame_name_list[frame_index]
        else:
            frame_j = random.sample(frame_name_list, k=1)[0]
        # frame_j = '77' # zq
        frame_path = os.path.join(scene_path, frame_j)
        # print(frame_j)
        # print("--------------------")
        img_path = frame_path + '/jpgs/'
        # # label_path = view_path + '/json_paras/'
        label_path = os.path.join(self.label_file_path, scene_i, frame_j, 'json_paras/')

        # decide the whole crowd GP density maps of the frame:
        # read all people
        # label_path_list = os.listdir(label_path)
        label_name0 = label_path_list0[0]
        label_path_name0 = os.path.join(label_path, label_name0)

        with open(label_path_name0, 'r') as data_file:
            coords_info_frame = json.load(data_file)
        # coords_3d_id_all_frame = read_json_all(coords_info_frame)
        coords_3d_id_all_frame, wld_map_paras_frame, hw_random = self.read_json_frame(coords_info_frame,
                                                                                      self.output_size,
                                                                                      self.r, self.a, self.b,
                                                                                      self.patch_num)
        wld_map_paras_frame = np.asarray(wld_map_paras_frame)

        # hw_random = [[194], [215]] # zq
        hw_random = np.asarray(hw_random)
        wld_map_paras = wld_map_paras_frame

        # read the images:
        img_views_list = []
        camera_paras_list = []
        wld_map_paras_list = []
        hw_random_list = []
        # depthMap_views_list = []

        GP_list = []
        single_view_dmaps_list = []

        img_views = []
        camera_paras = []
        wld_coords = []

        single_view_dmaps = []
        # print("label_path_list_sampling",label_path_list_sampling)
        view1_5_GT = []
        the_img_name = []
        for p in label_path_list_sampling:
            # img_name = p
            # label_name = 'pedInfo' + img_name[5:-6] +'.json'

            label_name = p
            img_name = label_name[0:-5] + '.jpg'
            the_img_name.append(label_name[0:-5])
            # depthMap_name = label_name[0:-5] + '.h5'

            # read images
            img_path_name = os.path.join(img_path, img_name)
            if os.path.exists(img_path_name) == False:
                img = np.zeros((640, 360, 3))
            else:
                img = cv2.imread(img_path_name)
                img = cv2.resize(img, (640, 360))
            # plt.imshow(img[:,:,0])
            # plt.show()
            # if img is None:
            #     break

            # (height, width, _) = img.shape # CSR-net_multi-view_counting_1output_lowRes_9_9_loadWeights
            # resized_width, resized_height = int(width/3), int(height/3)
            # img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
            img = img[:, :, (2, 1, 0)]  # BGR -> RGB
            img = img.astype('float32')

            img = img / 255.0
            img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

            img_views.append(img)

            # # read depthMap:
            # depthMap_path_name = os.path.join(depthMap_file_path2, depthMap_name)
            # with h5py.File(depthMap_path_name, 'r') as f:
            #     depthMap_i = f['depthMap'].value
            #     depthMap_i = np.expand_dims(depthMap_i, axis=-1)
            # depthMap_views.append(depthMap_i)

            # read labels
            label_path_name = os.path.join(label_path, label_name)
            with open(label_path_name, 'r') as data_file:
                try:
                    coords_info = json.load(data_file)
                except:
                    print(data_file, img_path)

            coords, coords_2d, paras = self.read_json_view(coords_info)

            # label_path2 = label_path0
            # label_path_name2 = os.path.join(label_path2, label_name)
            # with open(label_path_name2, 'r') as data_file:
            #     coords_info = json.load(data_file)
            # _, paras, _ = read_json(coords_info)

            coords_2d = np.asarray(coords_2d)
            hfwf = (90, 160)
            single_view_dmaps_i = np.zeros(hfwf)
            for x0, x1 in coords_2d:
                cx = x0 * 160
                cy = x1 * 90
                #     view1_5.add((cx, cy))
                # # print(f" {p} = ",len(view1_5))
                # for the_p in view1_5:
                #     # print(the_p)
                map_counting(single_view_dmaps_i, (cx, cy), sigma=3)
            # plt.imshow(single_view_dmaps_i)
            # plt.show()
            # single_view_dmaps_0 = self.density_map_creation(coords_2d, w = 480, h = 270)
            # plt.imshow(single_view_dmaps_0)
            # plt.show()
            # plt.imshow(single_view_dmaps_i)
            # plt.show()
            single_view_dmaps_i = np.expand_dims(single_view_dmaps_i, axis=0)
            single_view_dmaps_i *= self.scale
            single_view_dmaps.append(single_view_dmaps_i)

            # if wld_flag==0:
            #     wld_map_paras = wld_map_paras0 #decide_wld_map(coords_info, wld_h, wld_w)
            #     wld_flag = 1

            # form the camera paras list
            camera_paras.append(paras)

            # get the wld_coords:
            # print("coords",len(coords))
            wld_coords = wld_coords + coords
            # print("wld_coords",len(wld_coords))
            view1_5_wld_coords = wld_coords
            if len(view1_5_wld_coords) != 0:
                view1_5_wld_coords = self.id_unique(view1_5_wld_coords)
            else:
                view1_5_wld_coords = []
            # print("view1_5_wld_coords",len(view1_5_wld_coords))
            view1_5_wld_coords = np.asarray(view1_5_wld_coords)
            view1_5_GP_density_map = self.GP_density_map_creation(view1_5_wld_coords,
                                                                  self.output_size,
                                                                  wld_map_paras,
                                                                  hw_random)
            view1_5_GP_density_map *= self.scale
            view1_5_GT.append([view1_5_GP_density_map, np.sum(view1_5_GP_density_map)])
        # get the wld coords
        wld_coords = self.id_unique(wld_coords)
        wld_coords = np.asarray(wld_coords)

        # create the pmaps, instead of density maps
        GP_density_map = self.GP_density_map_creation(wld_coords,
                                                      self.output_size,
                                                      wld_map_paras,
                                                      hw_random)
        # GP_density_map = np.expand_dims(GP_density_map, axis=1)
        GP_density_map *= self.scale
        GP_top30_mask, GP_top50_mask, GP_top100_mask = self.mask_gt(GP_density_map)
        camera_paras = np.asarray(camera_paras)

        try:
            img_views = np.asarray(img_views)
            img_views = np.transpose(img_views, (0, 3, 1, 2))
            # img_views *= 100
        except:

            img_views = np.zeros((5, 3, 360, 640))
            GP_density_map = np.zeros((160, 180))

        # all_imformation = scen+frame+label_path_list_sampling
        return (img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_density_map, 0,
                view1_5_GT, the_img_name, label_flag, scene_i, frame_j,
                label_path_list_sampling)  # , GP_top30_mask, GP_top50_mask, GP_top100_mask

    def __len__(self):
        # return len(self.map_gt.keys())
        len_num = self.nb_scenes * 100 * self.nb_samplings
        # len_num = 1 * 100 * self.nb_samplings
        return len_num

# def test():
#     from multiview_detector.datasets.Wildtrack import Wildtrack
#     # from multiview_detector.datasets.MultiviewX import MultiviewX
#     from multiview_detector.utils_funcs.projection import get_worldcoord_from_imagecoord
#     dataset = frameDataset(Wildtrack(os.path.expanduser('/data/Wildtrack')), 0.2, train=False)
#     # test projection
#     world_grid_maps = []
#     xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
#     H, W = xx.shape
#     image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
#     import matplotlib.pyplot as plt
#     for i in range(20):
#         img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_density_map, GP_map_counts, _, _, _ = dataset.__getitem__(i+20)
#
#     pass
#
#
# if __name__ == '__main__':
#     test()
