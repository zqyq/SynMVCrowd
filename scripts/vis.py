import os

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


def recover_from_normalized(img):
    """
    img: shape [H, W, C], 归一化后的图像数据，值域[-2, 2]
    """
    # 反向标准化
    img[:, :, 0] = img[:, :, 0] * 0.229 + 0.485
    img[:, :, 1] = img[:, :, 1] * 0.224 + 0.456
    img[:, :, 2] = img[:, :, 2] * 0.225 + 0.406

    # 缩放到0-255范围
    img = img * 255.0

    # 转换为uint8类型
    img = img.astype('uint8')

    # RGB -> BGR转换（OpenCV默认格式）
    img = img[:, :, (2, 1, 0)]

    return img


def denormalized_tensors_to_images(tensors):
    """
    args:
        tensors: shape [N, C, H, W], 归一化后的图像数据，值域[-2, 2]
    """
    images = []
    for tensor in tensors:
        img = tensor.permute(1, 2, 0).cpu().numpy()  # 转换为 [H, W, C]
        img = recover_from_normalized(img)
        images.append(torch.from_numpy(img).float())
    images = torch.stack(images, dim=0)
    return images


def show_original_from_normalized(normalized_tensor, save_dir='original_image'):
    """
    显示经过ImageNet标准归一化后的图像原图

    参数:
        normalized_tensor: 经过Normalize的tensor [C, H, W]
    """
    # 定义逆归一化变换
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    # 执行逆归一化
    unnormalized = inv_normalize(normalized_tensor)

    # 将tensor转为numpy数组并调整维度顺序
    image = unnormalized.numpy().transpose((1, 2, 0))

    # 裁剪到[0,1]范围（处理可能的浮点误差）
    image = np.clip(image, 0, 1)

    # 显示图像
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(save_dir + '.png')
    # plt.show()
    plt.close()


def reconstructed_from_patches(patches: torch.Tensor, patch_size: int = 384) -> torch.Tensor:
    """
    从分块重建图像（用于可视化检查）

    Args:
        patches: 分块列表 [V, P, C, H, W], V为视角数，P为分块数，C为通道数，H和W为分块的高度和宽度
        patch_size: 每个分块的大小

    Returns:
        重建后的图像 [V, C, H, W*P]，在水平方向上拼接所有分块
    """
    # 获取输入维度
    V, P, C, H, W = patches.shape

    # 验证分块大小是否匹配
    if H != patch_size or W != patch_size:
        raise ValueError(f"Patch size mismatch. Expected {patch_size}x{patch_size}, got {H}x{W}")

    # 调整维度顺序 [V, P, C, H, W] -> [V, C, H, P*W]
    reconstructed = patches.permute(0, 2, 3, 1, 4)  # [V, C, H, P, W]
    reconstructed = reconstructed.reshape(V, C, H, -1)  # 在宽度维度拼接

    return reconstructed


# 加强显示点图dot map
def vis_dot_map(dot_map: torch.Tensor, save_dir='dot_map.png', dpi=300):
    """
    Args:
        dot_map: 点图张量 [1, H, W] 或 [1, 1, H, W]
        save_dir: 保存路径
        dpi: 图像分辨率
    点图的每个点用红色圆点表示，大小为3像素。
    """
    dot_map = dot_map.squeeze()
    H, W = dot_map.shape  # 获取高度和宽度
    # 创建一个RGB图像
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
    # 将点图转换为二值图像
    binary_map = (dot_map > 0).cpu().numpy().astype(np.uint8)
    # 在RGB图像上绘制红色圆点
    for y in range(H):
        for x in range(W):
            if binary_map[y, x] > 0:
                # 在位置(x, y)绘制红色圆点
                rgb_image[y, x] = [255, 0, 0]
    # 显示图像
    plt.imshow(rgb_image)
    plt.axis('off')  # 不显示坐标轴
    if save_dir is not None:
        plt.savefig(save_dir, dpi=dpi, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()


def show_patches_normalized(patches: torch.Tensor, save_dir=None, dpi=300):
    # pathches: [V, P, C, H, W]
    assert save_dir is not None, "save_dir must be specified to save the image."

    # 将分块转换为重建图像
    reconstructed = reconstructed_from_patches(patches)
    # 显示重建后的图像
    show_original_from_normalized(reconstructed, save_dir=save_dir)


def show_tensor_images(image_tensor, nrow=1, figsize=(10, 10), save_dir=None, dpi=300, force_show=False,axis='on',
                       sub_plot_title=None,color_bar=False):
    """
    参数:
    image_tensor -- 输入张量 [N, C, H, W] (C=1或3)
    nrow         -- 每行显示的图像数量
    figsize      -- 图像显示尺寸
    """
    # assert save_dir is not None, "save_dir must be specified to save the image."
    # to tensor
    if isinstance(image_tensor, list):
        if isinstance(image_tensor[0], np.ndarray):
            image_tensor = [torch.from_numpy(img) for img in image_tensor]
        image_tensor = torch.stack(image_tensor)

    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.from_numpy(image_tensor)
    if image_tensor.ndim != 4:
        # 补齐到4维
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    N, C, H, W = image_tensor.shape
    images = image_tensor.detach().cpu().numpy()
    if N == 1 and C == 1:  # 单通道单图像
        c1vis(image_tensor.squeeze(), save_dir=save_dir)
    else:
        if images.shape[1] == 1:  # 单通道灰度图
            images = images.squeeze(1)  # [N, H, W]
        else:  # 三通道RGB
            images = images.transpose(0, 2, 3, 1)  # [N, H, W, C]

        # 计算网格行列数
        if N > 4:
            # 设置一个合适的nrow
            sr = math.sqrt(N)
            nrow = int(sr)
        ncol = min(nrow, len(images))
        nrows = int(np.ceil(len(images) / ncol))

        # 创建画布
        fig, axes = plt.subplots(nrows, ncol, figsize=figsize, dpi=dpi)
        if nrows == 1:
            axes = axes.reshape(1, -1)  # 确保axes总是二维

        # 显示每张图像
        for idx, ax in enumerate(axes.flat):
            if idx < len(images):
                # 添加子图标题序号
                if sub_plot_title is not None:
                    ax.set_title(sub_plot_title[idx])
                if images[idx].ndim == 2:  # 灰度图
                    # ax.imshow(images[idx], vmin=0, vmax=1)
                    ax.imshow(images[idx])
                else:  # RGB图
                    if images.max()<=1.0:
                        ax.imshow(np.clip(images[idx], 0, 1))
                    else:
                        ax.imshow(images[idx].astype(np.uint8))
            if axis=='off':
                ax.axis('off')

        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)  # 确保目录存在
            if not save_dir.endswith('.png') or not save_dir.endswith('.jpg'):
                save_dir += '.png'
            plt.savefig(save_dir)
            if force_show==True:
                if color_bar==True:
                    plt.colorbar()
                plt.show()
        else:
            if color_bar == True:
                plt.colorbar()
            plt.show()
        plt.close()


def c3vis(img, name='test'):
    """
    可视化函数，展示图像
    :param img: shape [3, H, W] 的图像张量，表示三个通道的图像。
    :return: None
    """
    if isinstance(img, torch.Tensor):
        img = img.squeeze().cpu().numpy()  # 转换为 NumPy 数组
    elif isinstance(img, np.ndarray):
        img = img.squeeze()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # 转换为 [H, W, C] 格式
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)  # 将像素值转换为 [0, 255] 范围

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    if name is not None:
        plt.savefig(name + '.png')
    else:
        plt.show()  # 显示图像
    plt.close()


def multiImg_show(img: torch.tensor, name='test.png'):
    """
    可视化函数，展示多通道图像
    :param img: shape [N, 3, H, W] 的图像张量，表示多张图像。
    :return: None
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()  # 转换为 NumPy 数组

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, img.shape[0], figsize=(15, 5))
    for i in range(img.shape[0]):
        axes[i].imshow(img[i].transpose(1, 2, 0))  # 转换为 [H, W, C] 格式
        # axes[i].axis('off')  # 不显示坐标轴
        axes[i].set_title(f'Image {i + 1}')

    plt.savefig(name)
    plt.show()  # 显示图像
    plt.close()


def c1vis(img, save_dir=None):
    """
    可视化函数，展示单通道图像
    :param img: shape [H, W] 的图像张量，表示单通道图像。
    :return: None
    """
    if isinstance(img, torch.Tensor):
        img = img.squeeze().detach().cpu().numpy()
    elif isinstance(img, np.ndarray):
        img = img.squeeze()
    import matplotlib.pyplot as plt
    plt.imshow(img)
    # plt.axis('off')  # 不显示坐标轴
    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)  # 确保目录存在
        plt.savefig(save_dir + '.png')
    else:
        plt.show()  # 显示图像
    plt.close()


def rowshow(imgs, name):
    """
    可视化函数，展示多张图像
    :param imgs: shape [N, 3, H, W] 的图像张量，表示多张图像。
    :return: None
    """


def foldlinechart(losses, name):
    """
    绘制损失曲线
    :param losses: 损失列表
    :param name: 保存的文件名
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(name)
    # plt.show()
    plt.close()


def reconstruct_img_from_patches(patches: torch.tensor):
    """
    从分块重建图像（用于可视化检查）

    Args:
        patches: 分块列表 [N, C, H, W]

    Returns:
        重建后的图像 [C, H, W*N]
    """
    assert patches.dim() == 4, "Patches should be a 4D tensor [N, C, H, W]"
    N, C, H, W = patches.shape
    # 将分块沿宽度方向拼接
    return patches.permute(1, 2, 0).reshape(C, H, W * N)  # [C, H, W*N]


def _devide_patches(img: torch.tensor, patch_size: int = 384) -> torch.tensor:
    """
    Args:
        img: 输入图像张量 [C, H, W]
        patch_size: 分块大小
    Returns:
        分块后的图像张量 [N, C, H, W]，其中N为分块数量
    """
    assert img.dim() == 3, "Input image should be a 3D tensor [C, H, W]"
    C, H, W = img.shape
    # 计算分块数量
    n_patches_h = (H + patch_size - 1) // patch_size
    n_patches_w = (W + patch_size - 1) // patch_size
    patches = []

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            h_start = i * patch_size
            h_end = min(h_start + patch_size, H)
            w_start = j * patch_size
            w_end = min(w_start + patch_size, W)
            patches.append(img[:, h_start:h_end, w_start:w_end])

    return torch.stack(patches)  # [N, C, H', W']


def denormalize(tensor: torch.Tensor,
                mean: list = [0.485, 0.456, 0.406],
                std: list = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    将归一化的Tensor逆转换回可可视化的图像数据

    Args:
        tensor: 输入张量 (C,H,W) 或 (B,C,H,W)，值域应为归一化后范围
        mean: 使用的均值参数（需与transform中完全一致）
        std: 使用的标准差参数（需与transform中完全一致）

    Returns:
        np.ndarray: 反归一化后的图像 (H,W,C) 或 (B,H,W,C)，值域[0,255]，uint8类型

    处理流程：
    1. 克隆张量避免污染输入
    2. 反标准化计算： tensor = (tensor * std) + mean
    3. 截断到[0,1]范围
    4. 转换为numpy数组并调整通道顺序
    5. 缩放回0-255整数
    """
    # 输入校验
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"输入应为torch.Tensor，实际得到 {type(tensor)}")
    if tensor.dim() not in [3, 4]:
        raise ValueError(f"输入张量应为3D(C,H,W)或4D(B,C,H,W)，实际维度 {tensor.dim()}")

    # 参数转换为Tensor
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)

    # 反归一化计算
    tensor = tensor.clone()  # 避免修改原始张量
    if tensor.device != mean_tensor.device:
        mean_tensor = mean_tensor.to(tensor.device)
        std_tensor = std_tensor.to(tensor.device)

    tensor.mul_(std_tensor).add_(mean_tensor)

    # 处理值域并转换格式
    tensor = torch.clamp(tensor, 0, 1) * 255.0  # 重要！防止溢出

    return tensor.cpu().numpy().astype(np.uint8)


if __name__ == '__main__':
    import PIL.Image
    import torchvision
    from vis_tools import show_tensor_images, c3vis  # noqa

    exp = '/mnt/d/Datasets/CityStreet/image_frames/camera1/frame_0670.jpg'
    img = PIL.Image.open(exp).convert('RGB')
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize
    ])
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    denormalized_img = denormalize(img_tensor)
    # show_tensor_images(denormalized_img, nrow=1, save_dir='denormalized_image.png')
    c3vis(denormalized_img.squeeze(), name='c3vis_denormalized_image.png')
