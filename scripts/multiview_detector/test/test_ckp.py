import os
import cv2
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
# from test.post_proc import nms
from scripts.multiview_detector.test.post_proc import nms
#from dataset.dataset import conv_process
from model.City.Backbone import Backbone


class Test:
    def __init__(self, mode='separate', save=True):
        self.root = os.getcwd()
        self.gt_p = self.root + '/data/gp_point/'
        self.img_p1 = self.root + '/data/camera1/test/'
        self.img_p2 = self.root + '/data/camera2/test/'
        self.img_p3 = self.root + '/data/camera3/test/'


        self.save_p = self.root + '/result'
        self.mask_P = self.root + '/data/mask/view1_GP_mask.npz'
        self.mode = mode
        self.save = save
        self.h = 384
        self.w = 320
        self.down_radio = 2
        self.sig_thresold = 10
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #transforms.Resize([760, 1352]) #citystreet
            transforms.Resize([720, 1280])
        ])
        self.device = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

    def __call__(self, ckp_p):

        print(self.root + '/' + ckp_p)
        id_std = []
        pred_data = {}
        gt_data = {}

        with open(self.root + '/data/img_test.txt') as f:
            for line in f.readlines():
                img_name = line.strip()
                id_std.append(img_name[0:4])
                gt_point = np.load(
                    self.gt_p + img_name.replace('jpg', 'npy'))

                img_t1 = Image.open(self.img_p1 + img_name).convert(
                    'RGB')
                img_t1 = self.trans(img_t1).cuda().to(self.device[0])
                img_t1 = img_t1.unsqueeze(0)

                img_t2 = Image.open(self.img_p2 + img_name).convert(
                    'RGB')
                img_t2 = self.trans(img_t2).cuda().to(self.device[1])
                img_t2 = img_t2.unsqueeze(0)

                img_t3 = Image.open(self.img_p3 + img_name).convert(
                    'RGB')
                img_t3 = self.trans(img_t3).cuda().to(self.device[2])
                img_t3 = img_t3.unsqueeze(0)



                model = Backbone()
                model.FCN1.cuda().to(self.device[0])
                model.FCN2.cuda().to(self.device[1])
                model.FCN3.cuda().to(self.device[2])
                model.after_proj.cuda().to(self.device[3])



                checkpoint = torch.load(self.root + '/' + ckp_p)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                with torch.set_grad_enabled(False):
                    out = model(img_t1, img_t2, img_t3)
                    out = torch.squeeze(out)

                    out_den = out.detach().cpu().numpy()


                    plt.show()


                    #crop
                    # 左上
                    #out_den = out_den[0:192, 0: 160]
                    # 右上
                    #out_den = out_den[0:192, 160:]
                    # 左下
                    #out_den= out_den[192:, 0:160]
                    # 右下
                    #out_den = out_den[192:, 160:]


                    out, points = nms(out_den)
                    pre_count = np.sum(out)

                pred_data[img_name[0:4]] = {'num': pre_count, 'points': points}

                #denmap = conv_process(denmap)

                view_gp_mask = np.load(self.mask_P)
                view_gp_mask = cv2.resize(view_gp_mask.f.arr_0, (self.w, self.h))
                view_gp_mask = np.array(view_gp_mask).astype('float32')
                #view_gp_mask = view_gp_mask[..., None]
                #denmap = denmap * view_gp_mask

                gt_map = np.zeros((self.h, self.w))
                gt_point[:, [0, 1]] = gt_point[:, [1, 0]]
                gt_point[:, 0] = gt_point[:, 0] / self.down_radio
                gt_point[:, 1] = gt_point[:, 1] / self.down_radio
                gt_point = np.round(gt_point).astype(int)



                gt_map[gt_point[:, 0], gt_point[:, 1]] = 1

                #左上
                #gt_map = gt_map[0:192, 0: 160]
                #右上
                #gt_map = gt_map[0:192, 160:]
                #左下
                #gt_map = gt_map[192:, 0:160]
                #右下
                #gt_map = gt_map[192:, 160:]



                #gt_map = gt_map * view_gp_mask

                gt_point = np.nonzero(gt_map)
                gt_point = np.asarray(gt_point).transpose()

                gt_num = gt_point.shape[0]
                sigma = self.sig_thresold * np.ones_like(gt_point)

                gt_data[img_name[0:4]] = {'num': gt_num, 'points': gt_point, 'sigma': sigma}

                if self.save:
                    if self.mode == 'separate':
                        plt.figure()
                        plt.subplot(1, 2, 2)
                        plt.imshow(out_den)
                        plt.subplot(1, 2, 1)
                        plt.imshow(gt_map)
                    else:
                        plt.imshow(out,  cmap='Blues')
                        plt.imshow(gt_map, alpha='0.65')

                print("img: {}, gt: {}, pre: {}, res: {}".format(img_name, gt_num, pre_count, gt_num - pre_count))
                f = plt.gcf()
                if not os.path.exists(self.save_p):
                    os.makedirs(self.save_p)
                f.savefig(self.save_p + '/' + img_name, dpi=500.0)
                plt.close('all')
                f.clear()

        return pred_data, gt_data, id_std



