
# 衰减函数 方法
import numpy
import torch


class Attenuation(object):
    @staticmethod
    def func_attenuation_1(views, batch_size, xi=0):
        """
            这个是处理log衰减函数的
            view : 输入一张图的节点集合的in and out traffic flow信息
            xi：超参数，用于调整衰减函数的权重，当xi=0时，表示不增加权重
            return : 加权后的节点embedding
        """
        batch_views=[]
        # views=torch.tensor([item.cpu().detach().numpy() for item in views]).cuda()
        for view in views:
            batch_view = []
            for i in range(len(view)):
                new_view = []
                t = len(view)-i+1
                # print(xi*torch.log_(torch.tensor(t+1, dtype=torch.float)))
                # print(pow(torch.log_((torch.tensor(t+1, dtype=torch.float))), xi))
                detal = 1 / (pow(numpy.log(t + 1), xi))
                for n in range(len(view[i])):
                    new_view.append(torch.stack([detal * view[i][n][0], detal * view[i][n][1]]))
                    # print(f"测试一下衰减计算效果={[detal * view[i][0], detal * view[i][1]]}")
                    # print(f"new_view={new_view}")
                batch_view.append(torch.stack(new_view))
            batch_views.append(torch.stack(batch_view))

        return torch.stack(batch_views)



    @staticmethod
    def func_attenuation_2(view,batch_size, xi=0):
        """
            这个是处理exp衰减函数的
            view : 输入一张图的节点集合的in and out traffic flow信息
            xi：超参数，用于调整衰减函数的权重，当xi=0时，表示不增加权重
            return : 加权后的节点embedding
        """


        pass

    @staticmethod
    def func_attenuation_3(view,batch_size, xi=0):
        """
            这个是处理inv衰减函数的
            view : 输入一张图的节点集合的in and out traffic flow信息
            xi：超参数，用于调整衰减函数的权重，当xi=0时，表示不增加权重
            return : 加权后的节点embedding
        """


        pass

    @staticmethod
    def func_attenuation_log_fast(views, batch_size, xi=0):
        """
            这个是处理log衰减函数的
            view : 输入一张图的节点集合的in and out traffic flow信息
            xi：超参数，用于调整衰减函数的权重，当xi=0时，表示不增加权重
            return : 加权后的节点embedding
        """
        batch_views = []
        # views=torch.tensor([item.cpu().detach().numpy() for item in views]).cuda()
        for view in views:
            batch_view = []
            for i in range(len(view)):
                new_view = []
                t = len(view) - i + 1
                # print(xi*torch.log_(torch.tensor(t+1, dtype=torch.float)))
                # print(pow(torch.log_((torch.tensor(t+1, dtype=torch.float))), xi))
                detal = 1 / (pow(numpy.log(t + 1), xi))

                batch_view.append(torch.from_numpy(detal*view[i].cpu().numpy()).cuda())
            # print(batch_view)
            batch_views.append(torch.stack(batch_view))

        return torch.stack(batch_views)

    @staticmethod
    def func_attenuation_exp_fast(views, batch_size, xi=0):
        """
            这个是处理exp衰减函数的
            view : 输入一张图的节点集合的in and out traffic flow信息
            xi：超参数，用于调整衰减函数的权重，当xi=0时，表示不增加权重
            return : 加权后的节点embedding
        """
        batch_views = []
        # views=torch.tensor([item.cpu().detach().numpy() for item in views]).cuda()
        for view in views:
            batch_view = []
            for i in range(len(view)):
                new_view = []
                t = len(view) - i + 1
                # print(xi*torch.log_(torch.tensor(t+1, dtype=torch.float)))
                # print(pow(torch.log_((torch.tensor(t+1, dtype=torch.float))), xi))
                detal = numpy.exp(-xi * t)

                batch_view.append(torch.from_numpy(detal * view[i].cpu().numpy()).cuda())
            # print(batch_view)
            batch_views.append(torch.stack(batch_view))
        # for view in views:
        #     t = numpy.arange(len(view), 0, -1)  # 生成 t 的数组
        #     detal = numpy.exp(-xi * t)  # 计算所有衰减因子
        #
        #     # 使用广播将 detal 应用于整个 view
        #     weighted_view = detal[:, numpy.newaxis] * view  # 保持维度一致
        #     batch_views.append(weighted_view)

        return torch.stack(batch_views)

    @staticmethod
    def func_attenuation_inv_fast(views, batch_size, xi=0):
        """
            这个是处理inv衰减函数的
            view : 输入一张图的节点集合的in and out traffic flow信息
            xi：超参数，用于调整衰减函数的权重，当xi=0时，表示不增加权重
            return : 加权后的节点embedding
        """
        batch_views = []
        # views=torch.tensor([item.cpu().detach().numpy() for item in views]).cuda()
        for view in views:
            batch_view = []
            for i in range(len(view)):
                new_view = []
                t = len(view) - i + 1
                # print(xi*torch.log_(torch.tensor(t+1, dtype=torch.float)))
                # print(pow(torch.log_((torch.tensor(t+1, dtype=torch.float))), xi))
                detal = 1 / pow(t, xi)

                batch_view.append(torch.from_numpy(detal * view[i].cpu().numpy()).cuda())
            # print(batch_view)
            batch_views.append(torch.stack(batch_view))

        return torch.stack(batch_views)