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
from scripts.multiview_detector.loss import *
from scripts.multiview_detector.loss.gaussian_mse import GaussianMSE, GaussianMSE_norm
from scripts.multiview_detector.loss.ot_loss.genloss import GeneralizedLoss
from scripts.multiview_detector.utils.image_utils import img_color_denormalize
from scripts.multiview_detector.utils.meters import AverageMeter
from scripts.multiview_detector.utils.nms import nms
from scripts.vis import show_tensor_images


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, args, model, logdir, cls_thres=0.4, alpha=1.0, use_mse=False, id_ratio=0, scale=100, dst='ed',
                 devices=('cuda:0', 'cuda:1'), **kwargs):
        super(BaseTrainer, self).__init__()
        # self.mse_loss = nn.MSELoss()
        self.args = args
        self.focal_loss = FocalLoss()
        self.regress_loss = RegL1Loss()
        self.ce_loss = RegCELoss()
        self.devices = devices
        self.model = model.to(self.devices[0])
        self.scale = scale
        self.ot_loss_ed = GeneralizedLoss(type=dst, scale=scale)
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.criterion = GaussianMSE().cuda()
        self.criterion_norm = GaussianMSE_norm()
        self.cls_thres = cls_thres  # 0.4 * 100 = 40
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
        self.loss_type = self.args.loss_type  # mse represent that we use mse loss to optimize the model, else use ot loss

        self.detect_metric = 1

    def train(self, epoch, dataloader, optimizer, scheduler=None, is_debug=False):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        MAE = 0
        MSE = 0
        NAE = 0
        for idx, (data, imgs_gt, camera_paras, wld_map_paras, hw_random, map_gt_gt, dmap_gt, cam_loc,
                  scene_frame_cam) in enumerate(
            dataloader):
            imgs_gt = [img.to(self.devices[0]) for img in imgs_gt]
            data, camera_paras, wld_map_paras, hw_random = data.to(self.devices[0]), camera_paras.to(
                self.devices[0]), wld_map_paras.to(self.devices[0]), hw_random.to(self.devices[0])
            map_gt_gt, dmap_gt, cam_loc = map_gt_gt.to(self.devices[0]), dmap_gt.to(self.devices[0]), cam_loc.to(
                self.devices[0])  # cam_loc: [1, 3, 5, 3] (x, y, z) for each camera
            map_res, imgs_res = self.model(data, camera_paras, wld_map_paras, hw_random)

            loss_img = 0
            loss_w = 0
            if self.loss_type == 'ot':
                for b in range(data.shape[0]):
                    cur_map_gt = map_gt_gt[b]
                    for p in range(cur_map_gt.shape[0]):
                        # loss_w = loss_w + self.mse_loss(map_res[b][p:p + 1, 0:1, ...].squeeze(), 10 * dmap_gt[b, p].float())
                        cur_patch = cur_map_gt[p]
                        points = torch.nonzero(cur_patch)  # y, x index
                        if points.shape[0] == 0:
                            loss_w = loss_w + torch.abs(map_res[b][p:p + 1, 0:1, ...]).mean()
                        else:
                            points = points / float(cur_map_gt.shape[-2])
                            points[:, [0, 1]] = points[:, [1, 0]]
                            points = points.unsqueeze(0).unsqueeze(0)
                            # for cam in range(5):
                            #     loss_w = loss_w + 0.2 * self.ot_loss_md(map_res[b][p:p + 1, 0:1, ...], points, cam_loc[b, p,cam])
                            loss_w = loss_w + self.ot_loss_ed(map_res[b][p:p + 1, 0:1, ...], points, cam_loc[b, p])
            elif self.loss_type == 'mse':
                # loss_w = self.mse_loss(map_res[0], dmap_gt.float())
                loss_w, map_gt_conv = self.criterion_norm(map_res[0], map_gt_gt.permute(1, 0, 2, 3).float(),
                                                          dataloader.dataset.map_kernel)
            img_gt_conv = []
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                loss_img_i, img_gt_i = \
                    self.criterion_norm(img_res, img_gt.to(img_res.device), dataloader.dataset.img_kernel)
                img_gt_conv.append(img_gt_i)
                loss_img += loss_img_i
            img_gt_conv = torch.cat(img_gt_conv, dim=0)
            # loss_img = loss_img.to(torch.float32)
            if self.loss_type == 'ot':
                loss_total = loss_w / (data.shape[0] * 3.0) + loss_img / (data.shape[0] * 5.0)
            else:
                loss_total = loss_w + loss_img / 5.0
            t_f = time.time()
            t_forward += t_f - t_b
            # check nan
            if torch.isnan(loss_total):
                print(f'WARNING: received NaN loss, setting loss value to 0, skipping update at idx:{idx}')
                # loss_total = torch.zeros_like(loss_total)
                continue
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            # calcualte metric
            map_res_sum = map_res.sum() / self.scale
            map_gt_sum = map_gt_conv.sum() / self.scale
            mae = torch.abs(map_res_sum - map_gt_sum).item()
            mse = torch.square(map_res_sum - map_gt_sum).item()
            nae = torch.abs(map_res_sum - map_gt_sum) / map_gt_sum
            if nae > 1:
                nae = 1.0
            MAE += mae
            MSE += mse
            NAE += nae
            losses += loss_total.item()

            t_b = time.time()
            t_backward += t_b - t_f

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + idx / len(dataloader))

            if (idx) % 50 == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train Epoch: {epoch}, Batch:{(idx + 1)}, loss: {losses / (idx + 1):.6f}, '
                      f'MAE: {MAE / (idx + 1):.2f}, MSE: {math.sqrt(MSE / (idx + 1)):.2f}, NAE: {NAE / (idx + 1):.3f}, '
                      f'mae:{mae:.2f}, mse:{math.sqrt(mse):.2f}, nae:{nae:.3f}, '
                      f'map_res_sum: {map_res_sum:.3f}, map_gt_sum: {map_gt_sum:.3f}, '
                      f'Time: {t_epoch:.1f}, maxima: {map_res.max():.3f}')
                # pass
                # vis
                show_tensor_images(map_res[0],
                                   save_dir=os.path.join(self.map_res_dir, f'train/{epoch}_idx{idx}_map_res'))
                show_tensor_images(map_gt_conv,
                                   save_dir=os.path.join(self.map_gt_dir, f'train/{epoch}_idx{idx}_map_gt_gt'))
                show_tensor_images(imgs_res[0],
                                   save_dir=os.path.join(self.img_pred_dir, f'train/{epoch}_idx{idx}_imgs_res'))
                show_tensor_images(img_gt_conv,
                                   save_dir=os.path.join(self.img_gt_dir, f'train/{epoch}_idx{idx}_imgs_gt'))
            if is_debug and idx > 5:
                print(f'Break...')
                break
        MAE /= (idx + 1)
        NAE /= (idx + 1)
        MSE = math.sqrt(MSE / (idx + 1))
        print(f'Train Epoch {epoch}, MAE: {MAE:.2f}, NAE: {NAE:.3f}, MSE: {MSE:.2f}')

    def test(self, batch, epoch, data_loader, res_fpath=None, gt_fpath=None, visualize=False, is_debug=False):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        all_gt_list = []

        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None

        frame = 0
        log_interval = 50
        MAE = 0.0
        NAE = 0.0
        MSE = 0.0
        # test one sample
        # (data, imgs_gt, camera_paras, wld_map_paras, hw_random,
        #  map_gt_gt, dmap_gt, cam_loc, whole_pl_map) = data_loader.dataset.get_one_sample()
        # imgs_gt = imgs_gt.cuda()
        # data, camera_paras, wld_map_paras, hw_random = data.cuda(), camera_paras.cuda(), wld_map_paras.cuda(), hw_random.cuda()
        # map_gt_gt, dmap_gt, cam_loc = map_gt_gt.cuda(), dmap_gt.cuda(), cam_loc.cuda()  # cam_loc: [1, 3, 5, 3] (x, y, z) for each camera
        # dmap_gt = dmap_gt[None]  # [B, 1, H, W] -> [1, B, H, W]
        # with torch.no_grad():
        #     gp_res, imgs_res = self.model(data[None], camera_paras[None], wld_map_paras[None],
        #                                   hw_random[None],
        #                                   train=False)
        #     # show_tensor_images(view_gp_res, nrow=5, save_dir=os.path.join(self.logdir, f'one_sample_view_gp_res.png'))
        #     show_tensor_images(gp_res[0], save_dir=os.path.join(self.logdir, f'one_sample_gp_res.png'))
        #     show_tensor_images(dmap_gt, save_dir=os.path.join(self.logdir, f'one_sample_dmap_gt.png'))
        #
        # return
        for batch_idx, (
                data, imgs_gt, camera_paras, wld_map_paras, hw_random, map_gt, dmap_gt, cam_loc,
                whole_plane_map) in enumerate(
            data_loader):
            # imgs_gt = [img.cuda() for img in imgs_gt]
            data, camera_paras, wld_map_paras, hw_random = data.to(self.devices[0]), camera_paras.to(
                self.devices[0]), wld_map_paras.to(self.devices[0]), hw_random.to(self.devices[0])
            map_gt, cam_loc, whole_plane_map, dmap_gt = (map_gt.cuda(), cam_loc.cuda(), whole_plane_map.cuda(),
                                                         dmap_gt.cuda().float())
            if map_gt.flatten().max == 0:  # zq
                continue

            with torch.no_grad():
                map_res_mse, imgs_res = self.model(data, camera_paras, wld_map_paras, hw_random, train=False)
                # count
                map_res_sum = map_res_mse.sum() / self.scale
                map_gt_sum = whole_plane_map.sum()
                mae = torch.abs(map_res_sum - map_gt_sum).item()
                mse = torch.square(map_res_sum - map_gt_sum).item()
                nae = torch.abs(map_res_sum - map_gt_sum) / map_gt_sum
                if nae > 1:
                    nae = 1.0
                MAE += mae
                MSE += mse
                NAE += nae

                loss_img = 0
                loss_w = 0
                if self.loss_type == 'ot':
                    for b in range(batch):
                        cur_map_res = map_res_mse[b][:, :, :int(wld_map_paras[0, 3].item()),
                        :int(wld_map_paras[0, 4].item())]
                        cur_map_gt = whole_plane_map[b]

                        points = torch.nonzero(cur_map_gt)  # y, x index
                        if points.shape[0] == 0:
                            loss_w = loss_w + torch.abs(cur_map_res).mean()
                        else:
                            points = points / float(cur_map_gt.shape[-2])
                            points[:, [0, 1]] = points[:, [1, 0]]
                            points = points.unsqueeze(0).unsqueeze(0)
                            loss_w = loss_w + self.ot_loss_ed(map_res_mse[b], points, cam_loc[b, 0])  # 注意这里的相机参数无效

                    # for img_res, img_gt in zip(imgs_res, imgs_gt):
                    #     loss_img += self.criterion(img_res, img_gt.to(img_res.device), dataloader.dataset.img_kernel)
                    # loss_w = loss_w + loss_img / 5.0
                elif self.loss_type == 'mse':
                    # loss_w = self.mse_loss(map_res_mse[0], dmap_gt[None]).float()
                    loss_w, map_gt_conv = self.criterion_norm(map_res_mse[0], map_gt.permute(1, 0, 2, 3).float(),
                                                              data_loader.dataset.map_kernel)
                img_gt_conv = []
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    loss_img_i, img_gt_i = self.criterion_norm(img_res, img_gt.to(img_res.device),
                                                               data_loader.dataset.img_kernel)
                    loss_img += loss_img_i
                    img_gt_conv.append(img_gt_i)
                img_gt_conv = torch.cat(img_gt_conv, dim=0)
                losses = losses + loss_w.item()

                map_gt = whole_plane_map
                map_res_mse = map_res_mse[0][:, :, :int(wld_map_paras[0, 3].item()), :int(wld_map_paras[0, 4].item())]
                map_res_mse = (map_res_mse - torch.min(map_res_mse)) / (
                        torch.max(map_res_mse) - torch.min(map_res_mse) + 1e-8)

                # map_res = map_res[0][:, :, :int(wld_map_paras[0, 3].item()), :int(wld_map_paras[0, 4].item())]
                # map_res = (map_res - torch.min(map_res)) / (torch.max(map_res) - torch.min(map_res) + 1e-8)

            if res_fpath is not None and self.detect_metric:
                map_grid_res = map_res_mse.detach().cpu().squeeze()
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
            #
            if batch_idx % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Test Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.6f}, '
                      f'MAE: {MAE / (batch_idx + 1):.2f}, MSE: {math.sqrt(MSE / (batch_idx + 1)):.2f}, NAE: {NAE / (batch_idx + 1):.3f}, '
                      f'Time: {t_epoch:.1f}, maxima: {map_res_mse.max():.3f}')
                # vis
                from scripts.vis import show_tensor_images
                show_tensor_images(map_res_mse,
                                   save_dir=os.path.join(self.map_res_dir,
                                                         f'val/epo{epoch}_idx{batch_idx}_map_res.jpg'))
                # show_tensor_images(map_gt_conv,
                #                    save_dir=os.path.join(self.map_gt_dir,
                #                                          f'val/epo{epoch}_idx{batch_idx}_map_gt_gt.jpg'))
                show_tensor_images(imgs_res[0],
                                   save_dir=os.path.join(self.img_pred_dir,
                                                         f'val/epo{epoch}_idx{batch_idx}_imgs_res.jpg'))
                show_tensor_images(img_gt_conv,
                                   save_dir=os.path.join(self.img_gt_dir,
                                                         f'val/epo{epoch}_idx{batch_idx}_imgs_gt.jpg'))
            #
            pred = (map_res_mse > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)
            if is_debug and batch_idx == 2:
                break
        # count
        MAE /= (batch_idx + 1)
        NAE /= (batch_idx + 1)
        MSE = math.sqrt(MSE / (batch_idx + 1))
        print(
            f'Test Epoch: {epoch}, MAE: {MAE:.2f}, NAE: {NAE:.3f}, MSE: {MSE:.2f}, Cost time: {time.time() - t0:.1f}s')
        self.save_model(MAE)
        # save model

        # t1 = time.time()
        # t_epoch = t1 - t0

        # del data, imgs_gt, camera_paras, wld_map_paras, hw_random, map_gt, cam_loc, whole_plane_map, map_res_mse, imgs_res
        # torch.cuda.empty_cache()
        #
        moda = 0
        if res_fpath is not None and self.detect_metric:
            all_res_list = torch.cat(all_res_list, dim=0)
            all_gt_list = torch.cat(all_gt_list, dim=0)

            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%0.5f')
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_gt.txt', all_gt_list.numpy(), '%d')

            res_list = []
            gt_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 2.5, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])

            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                     os.path.abspath(os.path.dirname(res_fpath)) + '/all_gt.txt',
                                                     data_loader.dataset.base.__name__)
            f1_score = 2 * precision * recall / (precision + recall + 1e-4)
            print(
                f'moda: {moda:.1f}%, modp: {modp:.1f}%, precision:'
                f' {precision:.1f}%, recall: {recall:.1f}%, f1: {f1_score:.1f}%')

        print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
            losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))
        #
        # # del data, imgs_gt, camera_paras, wld_map_paras, hw_random, map_gt, cam_loc, whole_plane_map, map_res_mse, imgs_res, map_res
        # # torch.cuda.empty_cache()
        #
        # return losses / len(data_loader), moda, modp

    # # 用于人群计数
    # def test_count(self, epoch, data_loader, is_debug=True):
    #     self.model.eval()
    #     # 显示当前时间
    #     date_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #     print(f'Test Start: {date_now}, Epoch: {epoch}')
    #     MAE = 0.0
    #     NAE = 0.0
    #     MSE = 0.0
    #     t0 = time.time()
    #     for idx, (
    #             data, imgs_gt, camera_paras, wld_map_paras, hw_random, map_gt, cam_loc, whole_plane_map) in enumerate(
    #         data_loader):
    #         # imgs_gt = [img.cuda() for img in imgs_gt]
    #         data, camera_paras, wld_map_paras, hw_random = data.to(self.devices[0]), camera_paras.to(
    #             self.devices[0]), wld_map_paras.to(self.devices[0]), hw_random.to(self.devices[0])
    #         map_gt, cam_loc, whole_plane_map = map_gt.cuda(), cam_loc.cuda(), whole_plane_map.cuda()
    #         if map_gt.flatten().max == 0:  # zq
    #             continue
    #
    #         with torch.no_grad():
    #             map_res, imgs_res = self.model(data, camera_paras, wld_map_paras, hw_random, train=False)
    #             map_res_sum = map_res.sum() / self.scale
    #             map_gt_sum = whole_plane_map.sum()
    #             mae = torch.abs(map_res_sum - map_gt_sum).item()
    #             mse = torch.square(map_res_sum - map_gt_sum).item()
    #             nae = torch.abs(map_res_sum - map_gt_sum) / map_gt_sum
    #             if nae > 1:
    #                 nae = 1.0
    #             MAE += mae
    #             MSE += mse
    #             NAE += nae
    #
    #         if idx % 10 == 0:
    #             print(f'Test Epoch: {epoch} [{(idx + 1) * len(data)}/{len(data_loader.dataset)}, '
    #                   f'MAE: {mae:.3f}, NAE: {nae:.3f}, MSE: {mse:.3f}, '
    #                   f'maxima: {map_res.max().item():.3f}, time: {time.time() - t0:.3f}s]')
    #             show_tensor_images(map_res, save_dir=os.path.join(self.map_res_dir, f'val/{epoch}_idx{idx}_map_res'))
    #             show_tensor_images(whole_plane_map,
    #                                save_dir=os.path.join(self.map_gt_dir, f'val/{epoch}_idx{idx}_map_gt_gt.jpg'))
    #             show_tensor_images(imgs_res[0],
    #                                save_dir=os.path.join(self.img_pred_dir, f'val/{epoch}_idx{idx}_imgs_res'))
    #             show_tensor_images(imgs_gt[0],
    #                                save_dir=os.path.join(self.img_gt_dir, f'val/{epoch}_idx{idx}_imgs_gt'))
    #         # if is_debug and idx > 2:
    #         #     print(f'Break...')
    #         #     break
    #     #
    #     MAE = MAE / len(data_loader)
    #     NAE = NAE / len(data_loader)
    #     MSE = math.sqrt(MSE / len(data_loader))
    #     t1 = time.time()
    #     print(f'Test Epoch: {epoch}, MAE: {MAE:.3f}, NAE: {NAE:.3f}, MSE: {MSE:.3f}, '
    #           f'Time: {t1 - t0:.3f}s')
    #     # save best model
    #     self.save_model(MAE)

    def save_model(self, mae):
        model_state = self.model.state_dict()
        if mae < self.best_mae:
            self.best_mae = mae
            torch.save(model_state, os.path.join(self.logdir, 'best.pth'))
            print(f'Save best model with MAE: {self.best_mae:.3f} to {self.logdir}/best.pth')
        else:
            torch.save(model_state, os.path.join(self.logdir, 'last.pth'))
            print(f'Save last model with MAE: {mae:.3f} to {self.logdir}/last.pth')


def diff_cls_nt_dt_detect(gt_path, res_path, work_dir):
    """
    calculate the difference of cls, nt, dt
    """
    # all_gt_list = np.loadtxt(gt_path)
    # [frame, cx, cy, score]
    # 剔除score小于cls的点
    cls_list = [0.4, 0.5, 0.6, 0.8]
    t0 = time.time()
    nt = 5
    hd = 5
    os.makedirs(work_dir, exist_ok=True)
    log_dir = os.path.join(work_dir, 'detect.txt')
    with open(log_dir, 'a') as f:
        f.write('Detection Metrics under different parameters\n')
    for cls in cls_list:
        print(f'cls: {cls}, nt: {nt}, hd: {hd}')
        all_res_list = np.loadtxt(res_path)
        all_res_list = all_res_list[all_res_list[:, 3] > cls]
        all_res_list = torch.from_numpy(all_res_list)
        res_list = []
        # gt_list = []
        for frame in np.unique(all_res_list[:, 0]):
            res = all_res_list[all_res_list[:, 0] == frame, :]
            positions, scores = res[:, 1:3], res[:, 3]
            ids, count = nms(positions, scores, nt, np.inf)
            res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))

        res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])

        res_refresh_dir = work_dir + '/all_res.txt'
        # gt_dir=work_dir + '/all_gt.txt'

        np.savetxt(res_refresh_dir, res_list, '%d')
        # np.savetxt(all_gt_list,gt_dir,'%d')

        recall, precision, moda, modp = evaluate(res_refresh_dir,
                                                 gt_path,
                                                 'lcvcs',
                                                 hd=hd,
                                                 )
        f1_score = 2 * precision * recall / (precision + recall + 1e-4)
        print(
            f'cls: {cls}, nt: {nt}, hd: {hd}, '
            f'time cost: {time.time() - t0:.1f}s, '
            f'moda: {moda:.1f}%, modp: {modp:.1f}%, precision:'
            f' {precision:.1f}%, recall: {recall:.1f}%, f1: {f1_score:.1f}%')
        with open(log_dir, 'a') as f:
            f.write(
                f'cls: {cls}, nt: {nt}, hd: {hd}, '
                f'time cost: {time.time() - t0:.1f}s, '
                f'moda: {moda:.1f}%, modp: {modp:.1f}%, precision:'
                f' {precision:.1f}%, recall: {recall:.1f}%, f1: {f1_score:.1f}%\n')


if __name__ == '__main__':
    script_root = os.path.dirname(os.getcwd())
    # res_root= '/mnt/d/yunfei/Daijie_code/Baseline_OT/scripts/logs/lcvcs/resnet18-mse/2025-09-17_22-30-29-viewnum-5_dst-ed_test0'
    res_path = os.path.join(script_root,
                            'logs/lcvcs/resnet18-mse/2025-09-17_22-30-29-viewnum-5_dst-ed_test0/all_res.txt')
    gt_path = os.path.join(script_root,
                           'logs/lcvcs/resnet18-mse/2025-09-17_22-30-29-viewnum-5_dst-ed_test0/all_gt.txt')
    work_dir = os.path.dirname(res_path) + '/work'
    diff_cls_nt_dt_detect(gt_path, res_path, work_dir)
