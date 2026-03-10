"""
辅助显示数据集
"""
import os

import cv2
import numpy as np
import torch

from scripts.vis import show_tensor_images

# 显示当前帧所有视角图像

data_root = '/mnt/d/Datasets/LCVCS'
split = 'test'
scenes = os.listdir(os.path.join(data_root, split))
scenes.sort()
output_dir = data_root + f'/overview/{split}'
for scene in scenes:
    scene_path = os.path.join(data_root, split, scene)
    frames = os.listdir(scene_path)
    frames.sort()
    for frame in frames:
        frame_r = os.path.join(scene_path, frame)
        frame_path = os.path.join(scene_path, frame + '/jpgs')
        cams = os.listdir(frame_path)
        cams.sort()
        img_list = []
        sub_plot_title = []
        for cam in cams:
            if not cam.endswith('.jpg') and not cam.endswith('.png'):
                continue
            img = cv2.imread(os.path.join(frame_path, cam))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)
            sub_plot_title.append(cam.split('.')[0])
        imgs = torch.from_numpy(np.asarray(img_list)).permute(0, 3, 1, 2)
        show_tensor_images(imgs, save_dir=f'{output_dir}/{scene}_{frame}', nrow=5, axis='off',
                           sub_plot_title=sub_plot_title)
        break
    print(f'{scene} done!')
pass
