import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import torch.utils.checkpoint as cp
import  torch.autograd as autograd
import  numpy as np
import copy

from    rnnmodel import LSTM
from    copy import deepcopy
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Meta(nn.Module):
    def __init__(self, args, config, device):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.train_way
        self.k_spt = args.train_shot
        self.k_qry = args.train_query
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.config = config
        self.device = device
        self.net = LSTM(*config)
        self.domain_net = nn.Linear(128, 1)
        self.label0 = torch.zeros(8, 1).cuda()
        self.label1 = torch.ones(8, 1).cuda()
        self.tau = -999999

    def single_task_loss(self, data_in):
        # support_x, support_y, meta_x, meta_y = data_in
        support_x = data_in[0]
        support_y = data_in[1]
        meta_x = data_in[2]
        meta_y = data_in[3]
        support_xu = data_in[4]
        support_xaug = data_in[5]
        meta_xu = data_in[6]
        meta_xaug = data_in[7]

        meta_loss = []
        out = self.net(support_x)
        latent_reversed = ReverseLayerF.apply(latent, 0.5)
        domain = self.domain_net(latent_reversed)
        dloss0 = F.binary_cross_entropy_with_logits(domain, self.label0)
        pseudo = self.net(support_xu).double()
        pred = self.net(support_xaug)
        latent_reversed = ReverseLayerF.apply(latent, 0.5)
        domain = self.domain_net(latent_reversed)
        dloss1 = F.binary_cross_entropy_with_logits(domain, self.label1)
        mask = torch.gt(torch.abs(pseudo), self.tau)
        pseudo = torch.gt(pseudo, 0)
        loss = F.binary_cross_entropy_with_logits(out, support_y) + F.binary_cross_entropy_with_logits(pred*mask, pseudo*mask) + dloss0 #+ dloss1
        self.net.zero_grad()
        grad = autograd.grad(loss, self.net.parameters(), create_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
        out, _ = self.net(meta_x, vars=fast_weights, train=True)
        pseudo, _ = self.net(meta_xu, vars=fast_weights, train=True)
        pred, _ = self.net(meta_xaug, vars=fast_weights, train=True)
        mask = torch.gt(torch.abs(pseudo), self.tau)
        pseudo = torch.gt(pseudo, 0)
        meta_loss.append(F.binary_cross_entropy_with_logits(out, meta_y) + F.binary_cross_entropy_with_logits(pred*mask, pseudo*mask))
        for k in range(1, self.update_step):
            out, latent = self.net(support_x, vars = fast_weights, train=True)
            latent_reversed = ReverseLayerF.apply(latent, 0.5)
            domain = self.domain_net(latent_reversed)
            dloss0 = F.binary_cross_entropy_with_logits(domain, self.label0)
            pseudo, _ = self.net(support_xu, vars=fast_weights, train=True)
            pred, latent = self.net(support_xaug, vars=fast_weights, train=True)
            latent_reversed = ReverseLayerF.apply(latent, 0.5)
            domain = self.domain_net(latent_reversed)
            dloss1 = F.binary_cross_entropy_with_logits(domain, self.label1)
            mask = torch.gt(torch.abs(pseudo), self.tau)
            pseudo = torch.gt(pseudo, 0)
            loss = F.binary_cross_entropy_with_logits(out, support_y) + F.binary_cross_entropy_with_logits(pred*mask, pseudo*mask) + dloss1 + dloss0
            self.net.zero_grad()
            grad = autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            out, _ = self.net(meta_x, vars=fast_weights, train=True)
            pseudo, _ = self.net(meta_xu, vars=fast_weights, train=True)
            pred, _ = self.net(meta_xaug, vars=fast_weights, train=True)
            mask = torch.gt(torch.abs(pseudo), self.tau)
            pseudo = torch.gt(pseudo, 0)
            meta_loss.append(F.binary_cross_entropy_with_logits(out, meta_y) + F.binary_cross_entropy_with_logits(pred*mask, pseudo*mask))
        return meta_loss

    def forward(self, data, meta_train=True, fast_weights=None):
        support_x = data[0]
        support_y = data[1]
        meta_x = data[2]
        meta_y = data[3]
        support_xu = data[4]
        support_xaug = data[5]
        meta_xu = data[6]
        meta_xaug = data[7]
        if(meta_train):
            """
            :param support_x:   [b, setsz, c_, h, w]
            :param support_y:   [b, setsz]
            :param meta_x:      [b, setsz, c_, h, w]
            :param meta_y:      [b, setsz]
            """
            # assert(len(support_x.shape) == 5)
            # task_num_now = support_x.size(0)
            task_num_now = len(support_x)
            n_task_meta_loss = list(map(self.single_task_loss, zip(support_x,support_y,meta_x,meta_y,support_xu,support_xaug,meta_xu,meta_xaug)))
            re = n_task_meta_loss[0][-1].view(1,1)
            for i in range(1, task_num_now):
                re = torch.cat([re, n_task_meta_loss[i][-1].view(1,1)], dim = 0)
            return re
        elif fast_weights is None:
            """
            :param support_x:   [b, setsz,   c_, h, w]
            :param support_y:   [b, setsz  ]
            :param qx:          [b, querysz, c_, h, w]
            :param qy:          [b, querysz]
            :return:            [b, acc_dim]
            """
            fast_weights = list(self.net.parameters())
            for _ in range(self.update_step_test):
                # out = self.net(support_x, vars = fast_weights, train=True)
                # pseudo = self.net(support_xu, vars=fast_weights, train=True)
                # pred = self.net(support_xaug, vars=fast_weights, train=True)
                # loss = F.multilabel_soft_margin_loss(out, support_y) + F.multilabel_soft_margin_loss(pseudo, pred)
                out, _ = self.net(support_x, vars = fast_weights, train=True)
                loss = F.binary_cross_entropy_with_logits(out, support_y)
                self.net.zero_grad()
                grad = autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            return fast_weights
        else:
            out_q, _ = self.net(meta_x, vars=fast_weights)
            return out_q


def main():
    pass


if __name__ == '__main__':
    main()
