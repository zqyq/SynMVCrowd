import torch


class EDCost:
    def __init__(self) -> None:
        pass

    def __call__(self, x, y, calc):
        X, Y = x.clone(), y.clone()
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        C = torch.sqrt(torch.sum((x_col - y_row) ** 2, -1) + 1e-8)
        # condition = C < (8 / float(200))
        # C = torch.where(condition, 0, C)

        #relax
        # direcY = Y - calc[:, :-1].unsqueeze(1)
        # abs_dy = torch.sqrt(torch.sum(direcY ** 2, -1))
        #
        # condi = (abs_dy > 0.5).any(0).repeat(C.shape[1], 1).unsqueeze(0)
        # C = torch.where(condi, C / (torch.exp(-(C / float(2.0*1*1))**1)+1e-8), C)
        C = torch.exp(C) - 1.0
        return C



class MDCost:
    def __init__(self):
        super().__init__()

    def __call__(self, x, y, calc):
        """
        calc: x, y index, have already normalized
        """
        X, Y = x.clone(), y.clone()
        X, Y = X.squeeze(0), Y.squeeze(0)

        '''
        转换图像坐标系原点至图像左下角, 转换坐标表示转换为（x, y）
        '''
        # X[:, [0, 1]] = X[:, [1, 0]]
        X[:, 1] = 1 - X[:, 1]

        # Y[:, [0, 1]] = Y[:, [1, 0]]
        Y[:, 1] = 1 - Y[:, 1]
        calc = calc[:, :-1].float()  # remove z axis
        calc[:, 1] = 1 - calc[:, 1]
        maha_d = []

        gt2cam_dis = []

        for i in range(5):

            direcY = Y - calc[i]

            rot_col1 = direcY.clone()
            abs_dy = torch.sqrt(torch.sum(direcY ** 2, -1))
            gt2cam_dis.append(abs_dy)

            abs_dy = (abs_dy - torch.min(abs_dy)) / (torch.max(abs_dy) - torch.min(abs_dy) + 1e-8)

            abs_dy = torch.exp(abs_dy)

            # abs_dy_sig1 = torch.exp(0.5 * abs_dy)
            # abs_dy_sig2 = torch.exp(1.5 * abs_dy)
            #
            # sigma1 = abs_dy_sig1
            # sigma2 = abs_dy_sig2

            sigma1 = 1 / abs_dy #torch.ones_like(abs_dy) * 1.0
            sigma2 = 1 / abs_dy

            inv_var = torch.zeros((abs_dy.shape[0], 2, 2)).cuda()
            inv_var[:, 0, 0] = 1 / sigma1
            inv_var[:, 1, 1] = 1 / sigma2

            rot_col1 = rot_col1 / torch.sqrt(rot_col1[:, 0] ** 2 + rot_col1[:, 1] ** 2).unsqueeze(-1)

            '''
            (x, y) -> (-y, x)
            rotation = [x,  -y
                        y   x]
            '''
            rot_col2 = rot_col1.clone()
            rot_col2[:, [0, 1]] = rot_col2[:, [1, 0]]
            rot_col2[:, 0] = -1 * rot_col2[:, 0]

            rot_col1 = rot_col1.unsqueeze(-1)
            rot_col2 = rot_col2.unsqueeze(-1)

            rot_mat = torch.concat((rot_col1, rot_col2), dim=-1)
            rot_mat_T = torch.permute(rot_mat, (0, 2, 1))

            inv_cov = torch.matmul(torch.matmul(rot_mat, inv_var), rot_mat_T)

            x_col = X.unsqueeze(-2)
            y_row = Y.unsqueeze(-3)
            diff_left = (x_col - y_row).unsqueeze(-2)
            diff_right = (x_col - y_row).unsqueeze(-1)

            #mahalanobis_dis = torch.matmul(torch.matmul(diff_left, inv_cov), diff_right).squeeze(-1).squeeze(-1)
            mahalanobis_dis = torch.sqrt(
                torch.matmul(torch.matmul(diff_left, inv_cov), diff_right).squeeze(-1).squeeze(-1) + 1e-8)

            maha_d.append(mahalanobis_dis)


        maha_d = torch.stack(maha_d, dim=0)
        nearest_cam = torch.stack(gt2cam_dis, dim=0).min(0)[1]


        final_ma = torch.zeros_like(maha_d)[0]
        for i in range(nearest_cam.shape[0]):
            ind = nearest_cam[i]
            final_ma[:, i] = maha_d[ind, :, i]
        C = torch.exp(final_ma) - 1.

        C = C.unsqueeze(0)
        return C











#
#
#
#
#
# class MDCost:
#     def __init__(self):
#         super().__init__()
#
#     def __call__(self, x, y, calc):
#         """
#         calc: x, y index, have already normalized
#         """
#         calc = calc[:-1].float()  # remove z axis
#         calc[1] = 1 - calc[1]
#
#         X, Y = x.clone(), y.clone()
#         X, Y = X.squeeze(0), Y.squeeze(0)
#
#         '''
#         转换图像坐标系原点至图像左下角, 转换坐标表示转换为（x, y）
#         '''
#         # X[:, [0, 1]] = X[:, [1, 0]]
#         X[:, 1] = 1 - X[:, 1]
#
#         # Y[:, [0, 1]] = Y[:, [1, 0]]
#         Y[:, 1] = 1 - Y[:, 1]
#
#         direcY = Y - calc
#
#         rot_col1 = direcY.clone()
#         abs_dy = torch.sqrt(torch.sum(direcY ** 2, -1))
#
#         abs_dy = (abs_dy - torch.min(abs_dy)) / (torch.max(abs_dy) - torch.min(abs_dy) + 1e-8)
#         abs_dy = torch.exp(abs_dy)
#
#         sigma1 = abs_dy
#         sigma2 = torch.ones_like(abs_dy) * 1.0
#
#         inv_var = torch.zeros((abs_dy.shape[0], 2, 2)).cuda()
#         inv_var[:, 0, 0] = 1 / sigma1
#         inv_var[:, 1, 1] = 1 / sigma2
#
#         rot_col1 = rot_col1 / torch.sqrt(rot_col1[:, 0] ** 2 + rot_col1[:, 1] ** 2).unsqueeze(-1)
#
#         '''
#         (x, y) -> (-y, x)
#         rotation = [x,  -y
#                     y   x]
#         '''
#         rot_col2 = rot_col1.clone()
#         rot_col2[:, [0, 1]] = rot_col2[:, [1, 0]]
#         rot_col2[:, 0] = -1 * rot_col2[:, 0]
#
#         rot_col1 = rot_col1.unsqueeze(-1)
#         rot_col2 = rot_col2.unsqueeze(-1)
#
#         rot_mat = torch.concat((rot_col1, rot_col2), dim=-1)
#         rot_mat_T = torch.permute(rot_mat, (0, 2, 1))
#
#         inv_cov = torch.matmul(torch.matmul(rot_mat, inv_var), rot_mat_T)
#
#         x_col = X.unsqueeze(-2)
#         y_row = Y.unsqueeze(-3)
#         diff_left = (x_col - y_row).unsqueeze(-2)
#         diff_right = (x_col - y_row).unsqueeze(-1)
#
#         mahalanobis_dis = torch.sqrt(
#             torch.matmul(torch.matmul(diff_left, inv_cov), diff_right).squeeze(-1).squeeze(-1) + 1e-8)
#         # mahalanobis_dis = mahalanobis_dis / torch.unsqueeze((abs_dy), 0)
#
#         # print(f'maxd = {torch.max(mahalanobis_dis)}')
#         # print(f'mind = {torch.min(mahalanobis_dis)}')
#
#         C = torch.exp(mahalanobis_dis) - 1.
#         #C = mahalanobis_dis
#         #
#         # print(f'maxd = {torch.max(C)}')
#         # print(f'mind = {torch.min(C)}')
#
#         C = C.unsqueeze(0)
#         return C

# class MDCost:
#     def __init__(self):
#         super().__init__()
#
#     def __call__(self, x, y, calc):
#         """
#         calc: x, y index, have already normalized
#         """
#         X, Y = x.clone(), y.clone()
#         X, Y = X.squeeze(0), Y.squeeze(0)
#
#         '''
#         转换图像坐标系原点至图像左下角, 转换坐标表示转换为（x, y）
#         '''
#         # X[:, [0, 1]] = X[:, [1, 0]]
#         X[:, 1] = 1 - X[:, 1]
#
#         # Y[:, [0, 1]] = Y[:, [1, 0]]
#         Y[:, 1] = 1 - Y[:, 1]
#         calc = calc[:-1].float()  # remove z axis
#         calc[1] = 1 - calc[1]
#
#         direcY = Y - calc
#
#         rot_col1 = direcY.clone()
#         abs_dy = torch.sqrt(torch.sum(direcY ** 2, -1))
#
#         abs_dy = (abs_dy - torch.min(abs_dy)) / (torch.max(abs_dy) - torch.min(abs_dy) + 1e-8)
#
#         abs_dy = torch.exp(0.2 * abs_dy)
#
#         sigma1 = torch.ones_like(abs_dy) * 1.0
#         sigma2 = abs_dy
#
#         inv_var = torch.zeros((abs_dy.shape[0], 2, 2)).cuda()
#         inv_var[:, 0, 0] = 1 / sigma1
#         inv_var[:, 1, 1] = 1 / sigma2
#
#         rot_col1 = rot_col1 / torch.sqrt(rot_col1[:, 0] ** 2 + rot_col1[:, 1] ** 2).unsqueeze(-1)
#
#         '''
#         (x, y) -> (-y, x)
#         rotation = [x,  -y
#                     y   x]
#         '''
#         rot_col2 = rot_col1.clone()
#         rot_col2[:, [0, 1]] = rot_col2[:, [1, 0]]
#         rot_col2[:, 0] = -1 * rot_col2[:, 0]
#
#         rot_col1 = rot_col1.unsqueeze(-1)
#         rot_col2 = rot_col2.unsqueeze(-1)
#
#         rot_mat = torch.concat((rot_col1, rot_col2), dim=-1)
#         rot_mat_T = torch.permute(rot_mat, (0, 2, 1))
#
#         inv_cov = torch.matmul(torch.matmul(rot_mat, inv_var), rot_mat_T)
#
#         x_col = X.unsqueeze(-2)
#         y_row = Y.unsqueeze(-3)
#         diff_left = (x_col - y_row).unsqueeze(-2)
#         diff_right = (x_col - y_row).unsqueeze(-1)
#
#         mahalanobis_dis = torch.sqrt(
#             torch.matmul(torch.matmul(diff_left, inv_cov), diff_right).squeeze(-1).squeeze(-1) + 1e-8)
#
#         C = torch.exp(mahalanobis_dis) - 1.
#         C = C.unsqueeze(0)
#         return C