"""
寻找和目标图像最相似的图像
"""
# scene1 f10 v1: [10, 7.50043539833225, 122.01922320956743, 10.0, 120, 160, 1.75, 31.33624453258954, -219.04312133789062, -826.59033203125]}
# scene1 f1 v1: [10, 7.81693172070948, 111.62043130531467, 10.0, 120, 160, 1.75, 30.76046860084105, -218.4771270751953, -824.7172241210938]}
import os

import cv2
from skimage.metrics import structural_similarity as ssim

# target_img_dir = '/mnt/d/yunfei/Daijie_code/Baseline_OT/IJCV_Results/Multi-view detection/scene1_frame62_view35.png'
target_img_dir='/mnt/d/yunfei/Daijie_code/lb_CountFormer_semi_mvcc/examples/cvcs5.png'
# search_root_dir = '/mnt/d/Datasets/LCVCS/test/scene_1'
search_root_dir = '/mnt/d/Datasets/CVCS/val/scene_37'
frame_list = os.listdir(search_root_dir)
frame_list.sort()
# view = '5'
target_img = cv2.imread(target_img_dir)
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

best_match = None
highest_score = -1
for frame in frame_list:
    if '84' not in frame:
        continue
    view_list= os.listdir(os.path.join(search_root_dir, frame, 'jpgs'))
    view_list.sort()
    for view in view_list:
        a_img_dir = os.path.join(search_root_dir, frame, f'jpgs/{view}')
        if not os.path.exists(a_img_dir):
            continue
        # read the compared image
        compare_img = cv2.imread(a_img_dir)
        compare_img = cv2.cvtColor(compare_img, cv2.COLOR_BGR2GRAY)
        if compare_img.shape != target_img.shape:
            compare_img = cv2.resize(compare_img, (target_img.shape[1], target_img.shape[0]))
        # compute the similarity score using template matching
        sim = ssim(target_img, compare_img)
        # update the best match if the current score is higher
        if sim > highest_score:
            highest_score = sim
            best_match = a_img_dir
        print(f'Comparing with {a_img_dir}, similarity score: {sim}')
print(f'Best match: {best_match} with similarity score: {highest_score}')
