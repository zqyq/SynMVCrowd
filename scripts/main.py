import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append('../')
os.environ['CUDA_VISIBLE_DEVICES'] = "3,7"
import argparse
import datetime
import tqdm
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch import optim
from torch.utils.data import DataLoader
# from scripts.
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.str2bool import str2bool
# from scripts.multiview_detector.mvdetr_datasets.lcvcs.Wildtrack import Wildtrack
# from scripts.multiview_detector.mvdetr_datasets.lcvcs.frameDataset import frameDataset

import torchvision.transforms as T


def is_debugging():
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def main(args):
    # check if in debug mode
    is_debug = is_debugging()
    print(f'Debug mode: {is_debug}')
    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(False)
    else:
        torch.backends.cudnn.benchmark = True

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.ToTensor(), normalize, T.Resize([360, 640]), ])
    if is_debug:
        # args.dataset = 'lcvcs'
        args.dataset = 'wildtrack'

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def custom_collate_fn(batch):
        # batch = [b for b in batch if b is not None]
        imgs, img_gt, map_gt, frame = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        img_gt = torch.stack(img_gt, dim=0).permute(1, 0, 2, 3)
        map_gt = torch.stack(map_gt, dim=0)[0]
        return imgs, img_gt, map_gt, frame[0]

    # dataset
    if 'lcvcs' in args.dataset:
        from scripts.multiview_detector.datasets.lcvcs.Wildtrack import Wildtrack
        from scripts.multiview_detector.datasets.lcvcs.frameDataset import frameDataset
        base = Wildtrack(os.path.expanduser('/mnt/d/Datasets/Wildtrack'))
        train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=args.world_reduce,
                                 num_cam=args.num_cam)
        # test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=args.world_reduce,
        #                         num_cam=args.num_cam, frames=10)
        test_set = frameDataset(base, train=False, transform=train_trans,
                                grid_reduce=args.world_reduce,
                                num_cam=args.num_cam, frames=200)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True,
                                                   worker_init_fn=seed_worker,
                                                   collate_fn=custom_collate_fn if 'citystreet' in args.dataset else None)
        test_loader = torch.utils.data.DataLoader(test_set, 1, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True,
                                                  worker_init_fn=seed_worker,
                                                  collate_fn=custom_collate_fn if 'citystreet' in args.dataset else None)
    elif 'wildtrack' in args.dataset:
        from scripts.multiview_detector.datasets.wildtrack.Wildtrack import Wildtrack
        from scripts.multiview_detector.datasets.wildtrack.frameDataset import frameDataset
        base = Wildtrack(os.path.expanduser('/mnt/d/Datasets/Wildtrack'))
        train_set = frameDataset(base, train=True)
        test_set = frameDataset(base, train=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True,
                                                   worker_init_fn=seed_worker)
        test_loader = torch.utils.data.DataLoader(test_set, 1, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True,
                                                  worker_init_fn=seed_worker)
    else:
        raise NotImplementedError

    # logging
    if args.resume is None:
        logdir = (f'logs/{args.dataset}/{args.arch}-{args.loss_type}/{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}-'
                  f'viewnum-{args.num_cam}_dst-{args.distance}_test{1 if args.test else 0}')
        # copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        # for script in os.listdir('.'):
        #     if script.split('.')[-1] == 'py':
        #         dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
        #         shutil.copyfile(script, dst_file)

    if is_debug:
        logdir = f'logs/{args.dataset}/debug'

    sys.stdout = Logger(os.path.join(logdir, 'log.txt'))
    os.makedirs(logdir, exist_ok=True)
    print(logdir)
    print('Settings:')
    print(vars(args))
    devices = list(map(int, args.devices.split(',')))
    devices = [f'cuda:{d}' for d in devices]  # 转换为cuda设备字符串
    if len(devices) == 1:
        devices = [devices[0], devices[0]]
    print(f'Using devices: {devices}')
    # model
    pretrained_dir = '../checkpoints/46_maxmoda45.14009309117459_better.pth'
    # pretrained_dir ='../checkpoints'

    # scaler = GradScaler()

    if args.dataset == 'lcvcs':
        from scripts.multiview_detector.models.mvdetr import MVDeTr
        from scripts.multiview_detector.trainer import PerspectiveTrainer
        model = MVDeTr(args, args.arch, world_feat_arch=args.world_feat,
                       bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, droupout=args.dropout,
                       num_cam=args.num_cam, devices=devices)
        print(f'Loading pretrained model from {pretrained_dir}')
        model_state = torch.load(pretrained_dir, map_location=devices[0])
        model.load_state_dict(model_state, strict=False)
        trainer = PerspectiveTrainer(args, model, logdir, args.cls_thres, args.alpha,
                                     args.use_mse, args.id_ratio,
                                     scale=args.scale, dst=args.distance,
                                     devices=devices)
    elif args.dataset == 'wildtrack':
        from scripts.multiview_detector.models.mvdetr_wildtrack import MVDeTr
        from scripts.multiview_detector.trainer_wildtrack import PerspectiveTrainer
        model = MVDeTr(train_set, args.arch, world_feat_arch=args.world_feat,
                       bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, droupout=args.dropout,
                       num_cam=7, devices=devices)
        print(f'Loading pretrained model from {pretrained_dir}')
        model_state = torch.load(pretrained_dir, map_location=devices[0])
        model.load_state_dict(model_state, strict=False)
        trainer = PerspectiveTrainer(model, logdir, args.cls_thres, args.alpha,
                                     args.use_mse, args.id_ratio,
                                     scale=args.scale, dst=args.distance,
                                     devices=devices)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # # 部分测试
    # trainer.test(batch=1, epoch=0, data_loader=test_loader, res_fpath=os.path.join(logdir, 'test.txt'),
    #              gt_fpath=train_set.gt_fpath, visualize=args.visualize, is_debug=False)
    # return
    if args.resume is None:
        trainer.test(batch=1, epoch=0, data_loader=test_loader,
                     res_fpath=os.path.join(logdir, 'test.txt'),
                     gt_fpath=train_set.gt_fpath, visualize=args.visualize, is_debug=is_debug)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print(
                'learning rate: {}'.format(optimizer.param_groups[0]['lr']))
            print('Training...')
            # train_loss = trainer.train(args.batch_size, epoch, train_loader, optimizer, scaler, scheduler)
            trainer.train(epoch, train_loader, optimizer, scheduler=None, is_debug=is_debug)

            print('Testing...')
            trainer.test(batch=1, epoch=epoch, data_loader=test_loader,
                         res_fpath=os.path.join(logdir, 'test.txt'),
                         gt_fpath=train_set.gt_fpath, visualize=args.visualize, is_debug=is_debug)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--semi_supervised', type=float, default=0)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'transformer'])
    parser.add_argument('-d', '--dataset', type=str, default='lcvcs',
                        choices=['wildtrack', 'multiviewx', 'citystreet', 'lcvcs', 'lcvcs'])
    parser.add_argument('-j', '--num_workers', type=int, default=2)
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--num_cam', type=int, default=5, help='number of cameras')
    parser.add_argument('--distance', '-dst', type=str, choices=['ed', 'md'], default='ed')

    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--base_lr_ratio', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--world_feat', type=str, default='deform_trans',
                        choices=['conv', 'trans', 'deform_conv', 'deform_trans', 'aio'])
    parser.add_argument('--bottleneck_dim', type=int, default=512)
    parser.add_argument('--outfeat_dim', type=int, default=64)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    parser.add_argument('--devices', type=str, default='0,1', help='cuda devices')
    parser.add_argument('--scale', type=float, default=100.0, help='scale for ot loss')
    parser.add_argument('--test', type=bool, default=False, help='test mode')
    parser.add_argument('--pretrained_dir', '-pd', type=str,
                        default='/mnt/d/yunfei/Daijie_code/Baseline_OT/scripts/logs/lcvcs/2025-08-16_00-12-01-viewnum-9_dst-ed/best.pth',
                        help='path to the pretrained model')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['bce', 'ot', 'mse'])
    args = parser.parse_args()

    main(args)
