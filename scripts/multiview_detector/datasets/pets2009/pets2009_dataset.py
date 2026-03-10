import os

import numpy as np
import torch
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
from torchvision import transforms as T

from scripts.multiview_detector.datasets.pets2009.camera_proj_PETS2009 import calculate_camera_position


def z_score_normalize(tensor, dim=None, eps=1e-8):
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    return (tensor - mean) / (std + eps)


class Pets2009Dataset(Dataset):
    def __init__(self, train=True):
        """
        Args:
            root (string): Root directory of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        root = '/mnt/d/Datasets/PETS2009'
        self.root = root
        self._train = 'Train' if train else 'Test'
        self.file_root = os.path.join(root, f'image_frames/{self._train}')
        self.label_root = os.path.join(root, f'labels/{self._train}')
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Load all image files, sorting them to ensure that they are aligned
        self.image_files = {'view1': [], 'view2': [], 'view3': []}
        self.labels = {'view1': [], 'view2': [], 'view3': []}
        self.num_cam = 3  # Total 3 views
        self.img_shape = [768, 576]
        self.gp_grid_shape = [177, 152]

        # Example: Assuming images are in 'images' subdirectory and labels in 'labels.txt'
        self._read_img_lbl_file_dir()
        self.gp_gt = self.prepare_gp_gt()
        self.gt_fpath = None
        # cam location
        self.cam1_pos = calculate_camera_position('view1')
        self.cam2_pos = calculate_camera_position('view2')
        self.cam3_pos = calculate_camera_position('view3')
        # self.cam_loc=[self.cam1_pos, self.cam2_pos, self.cam3_pos]  # 相机位置列表
        # 归一化相机位置并存为[3, 3]的tensor
        self.cam_loc = torch.stack([z_score_normalize(torch.tensor(self.cam1_pos)),
                                         z_score_normalize(torch.tensor(self.cam2_pos)),
                                         z_score_normalize(torch.tensor(self.cam3_pos))], dim=0).float()

        map_sigma, map_kernel_size = 20 / 4, 20
        img_sigma, img_kernel_size = 10 / 8, 10
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

    def _read_img_lbl_file_dir(self):
        series_list = os.listdir(self.file_root)
        series_list.sort()
        for series in series_list:
            if not series.startswith('S'):
                continue
            series_path = os.path.join(self.file_root, series)
            label_path = os.path.join(self.label_root, series)
            time_parts = os.listdir(series_path)
            time_parts.sort()
            for time_part in time_parts:
                if os.path.isdir(os.path.join(series_path, time_part)) is False:
                    continue
                time_part_path = os.path.join(series_path, time_part)
                label_part_path = os.path.join(label_path, time_part)
                view_parts = os.listdir(time_part_path)
                for vp in view_parts:
                    if 'View' in vp:
                        view_num = int(vp.replace('View_', ''))
                        view_path = os.path.join(time_part_path, vp)
                        label_view_path = os.path.join(label_part_path, f'via_region_data_view{view_num}.json')
                        img_files = os.listdir(view_path)
                        img_files.sort()
                        # label json
                        label_data = self.load_json(label_view_path)
                        # assert len(label_data) == len(img_files), f'Label and image count mismatch in {label_view_path}'
                        for img_file in img_files:
                            if img_file.endswith('.jpg'):
                                self.image_files[f'view{view_num}'].append(os.path.join(view_path, img_file))
                                img_gt = self.generate_img_gt(label_data[img_file])
                                self.labels[f'view{view_num}'].append(img_gt)
        print(f'We have {len(self.image_files["view1"])} images in {self._train} set for each view.')

    def prepare_gp_gt(self):
        gp_gt_dir = os.path.join(self.root, 'labels/Ground_Plane_KeyPoint')
        gp_gt_path = os.path.join(gp_gt_dir, f'PETS2009_GroundPlanePoint_xyzdis_{self._train}.npy')
        gp_gt = []
        gp_gt_txt = []
        # if os.path.exists(gp_gt_path):
        data = np.load(gp_gt_path, allow_pickle=True)
        for i in range(len(data)):
            npd = data[i]
            frame_gt = np.zeros(self.gp_grid_shape, dtype=np.float32)
            for xyd in npd:
                x, y, d = xyd
                x, y = int(x), int(y)
                if 0 <= x < self.gp_grid_shape[1] and 0 <= y < self.gp_grid_shape[0]:
                    frame_gt[y, x] = 1
                    gp_gt_txt.append([i, x, y])
            gp_gt.append(frame_gt)

        if self._train == 'Test':
            ground_plane_coord_dir = os.path.join(self.label_root, 'Ground_Plane_Coordinate.txt')
            if not os.path.exists(ground_plane_coord_dir):
                print(f'Ground plane coordinate txt file does not exist. Creating: {ground_plane_coord_dir}')
                gp_gt_txt = np.array(gp_gt_txt)
                np.savetxt(ground_plane_coord_dir, gp_gt_txt, fmt='%d', delimiter=' ')
            else:
                print(f'Ground plane coordinate txt file already exists: {ground_plane_coord_dir}')
            self.gt_fpath = ground_plane_coord_dir
        return gp_gt

    def load_json(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def generate_img_gt(self, dct: dict):
        # num_people = len(dct['regions'])
        img_gt = np.zeros(self.img_shape)
        for k, v in dct['regions'].items():
            shape_attr = v['shape_attributes']
            cx, cy = shape_attr['cx'], shape_attr['cy']
            cx, cy = int(cx), int(cy)
            if 0 <= cx < self.img_shape[1] and 0 <= cy < self.img_shape[0]:
                img_gt[cy, cx] = 1
        return img_gt

    def __len__(self):
        return len(self.image_files['view1'])

    def __getitem__(self, idx):
        # Load images and labels
        images = []
        imgs_gt = []
        frame_id = idx
        for cam in range(1, self.num_cam + 1):
            img_path = self.image_files[f'view{cam}'][idx]
            label = self.labels[f'view{cam}'][idx]
            image = self.load_image(img_path)
            if self.transform:
                image = self.transform(image)

            images.append(image)
            imgs_gt.append(torch.from_numpy(label))
        images = torch.stack(images)
        imgs_gt = torch.stack(imgs_gt)
        gp_gt = torch.from_numpy(self.gp_gt[frame_id])
        return images, imgs_gt, gp_gt, frame_id

    def load_image(self, path):
        from PIL import Image
        return Image.open(path).convert('RGB')


if __name__ == '__main__':
    from scripts.vis import show_tensor_images  # noqa

    train_set = Pets2009Dataset(train=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    # example_json = '/mnt/d/Datasets/PETS2009/labels/Train/S1L3/14_17/via_region_data_view1.json'
    # data = train_set.load_json(example_json)
    for i, (imgs, lbls, gp, fid) in enumerate(train_loader):
        print(f'Frame {fid}:')
        print(f'  Image shapes: {[img.shape for img in imgs]}')
        print(f'  Label shapes: {[lbl.shape for lbl in lbls]}')
        print(f'  GP shape: {gp.shape}')
        if i == 2:
            break
