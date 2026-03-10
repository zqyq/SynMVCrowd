# fig, ax = plt.subplots()
# ax.set_xticks([])
# ax.set_yticks([])
# ax.axis('off')
#
# ax.imshow(map_res_mse[0, 0].cpu().detach().numpy().squeeze(), cmap='binary')
# fig.savefig(os.path.join(self.logdir, f'result_{scene_idx}_frame{int(frame % 10)}_moda{str(moda)}.jpg'), dpi=1000.0,
#             pad_inches=0.0, bbox_inches='tight')
#
# # gt = map_gt[0].cpu().detach().numpy().squeeze()
# # point = np.asarray(np.nonzero(gt)).transpose()
#
# ax.imshow(map_gt[0].cpu().detach().numpy().squeeze(), cmap='binary')
# fig.savefig(os.path.join(self.logdir, f'gt_{scene_idx}_{int(frame % 10)}.jpg'), dpi=1000.0, pad_inches=0.0,
#             bbox_inches='tight')
# frame = 1 + frame
import os
import time

import math
import numpy as np
import torch
from torch import nn

from scripts.multiview_detector.evaluation.evaluate import evaluate
# from scripts.multiview_detector.loss import *
from scripts.multiview_detector.loss.gaussian_mse import GaussianMSE
from scripts.multiview_detector.loss.ot_loss.genloss import GeneralizedLoss
from scripts.multiview_detector.utils.image_utils import img_color_denormalize
from scripts.multiview_detector.utils.meters import AverageMeter
from scripts.multiview_detector.utils.nms import nms
from scripts.vis import c1vis, show_tensor_images


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, logdir, cls_thres=0.4, alpha=1.0, use_mse=False, id_ratio=0, scale=100, dst='ed',
                 devices=('cuda:0', 'cuda:1')):
        super(BaseTrainer, self).__init__()

        self.devices = devices
        self.model = model.to(self.devices[0])
        self.scale = scale
        self.ot_loss_ed = GeneralizedLoss(type=dst, scale=scale, device=devices[0])
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.criterion = GaussianMSE().cuda()
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.alpha = alpha
        self.use_mse = use_mse
        self.id_ratio = id_ratio
        # logdir
        img_pred_dir = os.path.join(logdir, 'img_pred')
        img_gt_dir = os.path.join(logdir, 'img_gt')
        map_gt_dir = os.path.join(logdir, 'map_gt')
        map_res_dir = os.path.join(logdir, 'map_res')
        os.makedirs(img_pred_dir, exist_ok=True)
        os.makedirs(img_gt_dir, exist_ok=True)
        os.makedirs(map_gt_dir, exist_ok=True)
        os.makedirs(map_res_dir, exist_ok=True)
        self.img_pred_dir = img_pred_dir
        self.img_gt_dir = img_gt_dir
        self.map_gt_dir = map_gt_dir
        self.map_res_dir = map_res_dir
        self.best_mae = float('inf')
        self.best_moda = 0

    def train(self, epoch, dataloader, optimizer, scheduler=None, is_debug=False):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        print(f'label rate== 0.3')
        for idx, (data, imgs_gt, map_gt_gt, frame) in enumerate(dataloader):
            if idx >= len(dataloader) * 0.3:
                print(f'break at idx:{idx}')
                break
            data = data.to(self.devices[0])
            imgs_gt = [img.to(self.devices[0]) for img in imgs_gt]
            map_gt_gt = map_gt_gt.to(self.devices[0])
            map_res, imgs_res = self.model(data, train=True)

            loss_img = 0
            loss_w = 0
            for b in range(data.shape[0]):
                cur_map_gt = map_gt_gt[b]
                for p in range(cur_map_gt.shape[0]):
                    # loss_w = loss_w + self.mse_loss(map_res[b][p:p + 1, 0:1, ...].squeeze(), 10 * dmap_gt[b, p].float())
                    cur_patch = cur_map_gt[p].squeeze()
                    points = torch.nonzero(cur_patch)  # y, x index
                    if points.shape[0] == 0:
                        loss_w = loss_w + torch.abs(map_res[b][p:p + 1, 0:1, ...]).mean().to(self.devices[0])
                    else:
                        points = points / float(cur_map_gt.shape[-2])
                        points[:, [0, 1]] = points[:, [1, 0]]
                        points = points.unsqueeze(0).unsqueeze(0)
                        # for cam in range(5):
                        #     loss_w = loss_w + 0.2 * self.ot_loss_md(map_res[b][p:p + 1, 0:1, ...], points, cam_loc[b, p,cam])
                        loss_w = loss_w + self.ot_loss_ed(map_res[b][p:p + 1, 0:1, ...], points,
                                                          dataloader.dataset.cam_loc.to(self.devices[0]))
            # 2D gt
            imgs_gt_conv = []
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                loss_img_i, target_gt = self.criterion(img_res[None], img_gt.to(img_res.device),
                                                       dataloader.dataset.img_kernel)
                imgs_gt_conv.append(target_gt)
                loss_img += loss_img_i
            imgs_gt = torch.cat(imgs_gt_conv)
            loss_total = loss_w / (data.shape[0] * 3.0) + loss_img / (data.shape[0] * 7.0)
            t_f = time.time()
            t_forward += t_f - t_b

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losses += loss_total.item()

            t_b = time.time()
            t_backward += t_b - t_f

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + idx / len(dataloader))

            if (idx) % 40 == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train Epoch: {epoch}, Batch:{(idx + 1)}, loss: {losses / (idx + 1):.6f}, '
                      f'Time: {t_epoch:.1f}, maxima: {map_res.max():.3f}')
                # vis
                c1vis(map_res[0],
                      save_dir=os.path.join(self.map_res_dir, f'train/{epoch}_idx{idx}_map_res'))
                c1vis(map_gt_gt.permute(1, 0, 2, 3),
                      save_dir=os.path.join(self.map_gt_dir, f'train/{epoch}_idx{idx}_map_gt_gt'))
                show_tensor_images(imgs_res,
                                   save_dir=os.path.join(self.img_pred_dir, f'train/{epoch}_idx{idx}_imgs_res'))
                show_tensor_images(imgs_gt,
                                   save_dir=os.path.join(self.img_gt_dir, f'train/{epoch}_idx{idx}_imgs_gt'))
            if is_debug and idx > 2:
                print(f'Break...')
                break

    def test_detect(self, epoch, data_loader, res_fpath=None, gt_fpath=None):
        self.model.eval()
        print(f'Testing...')
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        all_gt_list = []
        MAE = 0.0
        NAE = 0.0
        MSE = 0.0
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        frame = 0
        log_interval = 10
        for batch_idx, (data, imgs_gt, map_gt, frame,) in enumerate(data_loader):
            data = data.to(self.devices[0])
            imgs_gt = [img.to(self.devices[0]) for img in imgs_gt]
            map_gt = map_gt.to(self.devices[0])
            with torch.no_grad():
                map_res, imgs_res = self.model(data, train=False)
                map_res_sum = map_res.sum() / self.scale
                map_gt_sum = map_gt.sum()
                mae = torch.abs(map_res_sum - map_gt_sum).item()
                mse = torch.square(map_res_sum - map_gt_sum).item()
                nae = torch.abs(map_res_sum - map_gt_sum) / map_gt_sum
                if nae > 1:
                    nae = 1.0
                MAE += mae
                MSE += mse
                NAE += nae

                map_res = (map_res - torch.min(map_res)) / (torch.max(map_res) - torch.min(map_res) + 1e-8)
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()

                # gt:
                map_grid_gt = map_gt.detach().cpu().squeeze()
                v_s_gt = map_grid_gt[map_grid_gt > self.cls_thres].unsqueeze(1)
                grid_ij_gt = (map_grid_gt > self.cls_thres).nonzero()

                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                    grid_xy_gt = grid_ij_gt[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                    grid_xy_gt = grid_ij_gt

                frame = 1 + frame

                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float(), v_s], dim=1))
                all_gt_list.append(torch.cat([(torch.ones_like(v_s_gt) * frame).float(), grid_xy_gt.float()], dim=1))

            if batch_idx % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Test Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.6f}, '
                      f'Time: {t_epoch:.1f}, maxima: {map_res.max():.3f}')
                # vis
                from scripts.vis import show_tensor_images
                show_tensor_images(map_res[0],
                                   save_dir=os.path.join(self.map_res_dir,
                                                         f'val/epo{epoch}_idx{batch_idx}_map_res.jpg'))
                show_tensor_images(map_gt,
                                   save_dir=os.path.join(self.map_gt_dir,
                                                         f'val/epo{epoch}_idx{batch_idx}_map_gt_gt.jpg'))
                show_tensor_images(imgs_res,
                                   save_dir=os.path.join(self.img_pred_dir,
                                                         f'val/epo{epoch}_idx{batch_idx}_imgs_res.jpg'))
                show_tensor_images(torch.cat(imgs_gt),
                                   save_dir=os.path.join(self.img_gt_dir,
                                                         f'val/epo{epoch}_idx{batch_idx}_imgs_gt.jpg'))

            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        t1 = time.time()
        t_epoch = t1 - t0

        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            all_gt_list = torch.cat(all_gt_list, dim=0)

            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%0.5f')
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_gt.txt', all_gt_list.numpy(), '%d')

            res_list = []
            gt_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 5, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])

            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                     os.path.abspath(os.path.dirname(res_fpath)) + '/all_gt.txt',
                                                     data_loader.dataset.base.__name__,
                                                     hd=5)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            print(
                f'moda: {moda:.3f}, modp: {modp:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f},')

        print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
            losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))
        # save best model
        self.save_model_moda(moda)
        MAE = MAE / len(data_loader)
        NAE = NAE / len(data_loader)
        MSE = math.sqrt(MSE / len(data_loader))
        print(f'Test Epoch: {epoch}, MAE: {MAE:.3f}, NAE: {NAE:.3f}, MSE: {MSE:.3f}, '
              f'Time: {t_epoch:.3f}s')
        return losses / len(data_loader), moda, modp

    # 用于人群计数
    def test(self, batch, epoch, data_loader,
             res_fpath=None, gt_fpath=None, visualize=False, is_debug=False):
        self.model.eval()
        # 显示当前时间
        date_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f'Test Start: {date_now}, Epoch: {epoch}')
        MAE = 0.0
        NAE = 0.0
        MSE = 0.0
        t0 = time.time()
        for idx, (data, imgs_gt, map_gt_gt, frame,) in enumerate(data_loader):
            data = data.to(self.devices[0])
            imgs_gt = [img.to(self.devices[0]) for img in imgs_gt]
            map_gt_gt = map_gt_gt.to(self.devices[0])
            with torch.no_grad():
                map_res, imgs_res = self.model(data, train=False)
                map_res_sum = map_res.sum() / self.scale
                map_gt_sum = map_gt_gt.sum()
                mae = torch.abs(map_res_sum - map_gt_sum).item()
                mse = torch.square(map_res_sum - map_gt_sum).item()
                nae = torch.abs(map_res_sum - map_gt_sum) / map_gt_sum
                if nae > 1:
                    nae = 1.0
                MAE += mae
                MSE += mse
                NAE += nae

            imgs_gt_conv = []
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                _, target_gt = self.criterion(img_res[None], img_gt.to(img_res.device),
                                              data_loader.dataset.img_kernel)
                imgs_gt_conv.append(target_gt)
                # loss_img += loss_img_i
            imgs_gt = torch.cat(imgs_gt_conv)
            if idx % 10 == 0:
                print(f'Test Epoch: {epoch} [{(idx + 1) * len(data)}/{len(data_loader.dataset)}, '
                      f'MAE: {mae:.3f}, NAE: {nae:.3f}, MSE: {mse:.3f}, '
                      f'maxima: {map_res.max().item():.3f}, time: {time.time() - t0:.3f}s]')
                c1vis(map_res[0],
                      save_dir=os.path.join(self.map_res_dir, f'val/{epoch}_idx{idx}_map_res'))
                c1vis(map_gt_gt.permute(1, 0, 2, 3),
                      save_dir=os.path.join(self.map_gt_dir, f'val/{epoch}_idx{idx}_map_gt_gt'))
                show_tensor_images(imgs_res,
                                   save_dir=os.path.join(self.img_pred_dir, f'val/{epoch}_idx{idx}_imgs_res'))
                show_tensor_images(imgs_gt,
                                   save_dir=os.path.join(self.img_gt_dir, f'val/{epoch}_idx{idx}_imgs_gt'))
            if is_debug and idx > 2:
                print(f'Break...')
                break
        #
        MAE = MAE / len(data_loader)
        NAE = NAE / len(data_loader)
        MSE = math.sqrt(MSE / len(data_loader))
        t1 = time.time()
        print(f'Test Epoch: {epoch}, MAE: {MAE:.3f}, NAE: {NAE:.3f}, MSE: {MSE:.3f}, '
              f'Time: {t1 - t0:.3f}s')
        # save best model
        self.save_model(MAE)

    def save_model(self, mae):
        model_state = self.model.state_dict()
        if mae < self.best_mae:
            self.best_mae = mae
            torch.save(model_state, os.path.join(self.logdir, 'best.pth'))
            print(f'Save best model with MAE: {self.best_mae:.3f} to {self.logdir}/best.pth')
        else:
            torch.save(model_state, os.path.join(self.logdir, 'last.pth'))
            print(f'Save last model with MAE: {mae:.3f} to {self.logdir}/last.pth')

    def save_model_moda(self, moda):
        model_state = self.model.state_dict()
        if moda > self.best_moda:
            self.best_moda = moda
            torch.save(model_state, os.path.join(self.logdir, 'best.pth'))
            print(f'Save best model with MODA: {self.best_moda:.3f} to {self.logdir}/best.pth')
        else:
            torch.save(model_state, os.path.join(self.logdir, 'last.pth'))
            print(f'Save last model with MODA: {moda:.3f} to {self.logdir}/last.pth')
