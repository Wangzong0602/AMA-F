import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CALoss(nn.Module):
    def __init__(self, margin=0.1, dist_type='l2', num=2, weight=0.2):
        super(CALoss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')

        self.num = num
        self.fc1 = nn.Sequential(nn.Linear(2048, 256, bias=False))
        self.fc2 = nn.Sequential(nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 256), nn.BatchNorm1d(256))
        self.weight=weight


    def forward(self, feat1, feat2, label1, label2):

        x1 = feat1
        x2 = feat2
        # 分布约束
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            if i == 0:
                dist = max(0, self.dist(center1, center2) - self.margin)
            else:
                dist += max(0, self.dist(center1, center2) - self.margin)

        # 语义约束
        x = torch.cat((x1, x2), 0)
        x = F.normalize(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.normalize(x)
        loss1 = 0
        num = int(x.size(0) / self.num)
        for i in range(self.num):
            for j in range(self.num):
                if i<j:
                    loss1 += ((x[i*num:(i+1)*num,:] - x[j*num:(j+1)*num,:]).norm(dim=1,keepdim=True)).mean()

        x1, x2 = torch.chunk(x, 2)
        x1 = x1.chunk(label_num, 0)
        x2 = x2.chunk(label_num, 0)
        loss2 = 0
        for i in range(label_num):
            center1 = torch.mean(x1[i], dim=0)
            center2 = torch.mean(x2[i], dim=0)
            loss2 += max(0, self.dist(center1, center2) - self.margin)
        loss2 = loss2/label_num
        loss = (loss1 + loss2)/2
        loss = dist + loss*self.weight
        return loss

# class SCLoss(nn.Module):
#     def __init__(self, num=2, weight=0.55):
#         super(SCLoss, self).__init__()
#         self.num = num
#         self.weight = weight
#         self.fc1 = nn.Sequential(nn.Linear(2048, 256, bias=False))
#         self.fc2 = nn.Sequential(nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 256), nn.BatchNorm1d(256))
#
#     def forward(self, x, label1):
#         label_num = len(label1.unique())
#         k = int(label1.size(0)/label_num)
#         x = F.normalize(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = F.normalize(x)
#         loss = 0
#         num = int(x.size(0) / self.num)
#         for i in range(num):
#             index = i//k
#             for j in range(k):
#                 if i%k==j:
#                     loss += (x[i] - x[num+index*k+j]).norm(dim=0, keepdim=True)*self.weight
#                 else:
#                     loss += (x[i] - x[num+index*k+j]).norm(dim=0, keepdim=True)*(1-self.weight)/(k-1)
#         return loss/num


# 在loss.py中添加以下代码
class SCLoss(nn.Module):
    def __init__(self, num=3, weights=(0.55, 0.55), variance_aware=True):
        super(SCLoss, self).__init__()
        self.num = num
        self.weights = weights
        self.variance_aware = variance_aware
        self.lambda_i = 0.2
        #self.lambda_d =1
        # 共享特征变换层
        self.fc1 = nn.Sequential(nn.Linear(2048, 256, bias=False))
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )

    def compute_center(self, features):
        """计算模态特征中心"""
        return torch.mean(features, dim=0, keepdim=True)

    def dynamic_weight(self, f_v, f_i):
        """方差感知动态权重"""
        var_v = torch.var(f_v, dim=1).mean()
        var_i = torch.var(f_i, dim=1).mean()
        total_var = var_v + var_i
        beta = var_i / total_var
        return beta

    def forward(self, f_v, f_i, f_t, label):
        """
        双路径语义一致性损失
        Args:
            f_v: 可见光特征 (batch_size, 2048)
            f_i: 红外特征 (batch_size, 2048)
            f_t: 过渡模态特征 (batch_size, 2048)
            label: 样本标签
        """
        # 1. 特征归一化和变换
        f_v = F.normalize(f_v)
        f_i = F.normalize(f_i)
        f_t = F.normalize(f_t)

        # 2. 特征变换
        f_v_trans = self.fc2(self.fc1(f_v))
        f_i_trans = self.fc2(self.fc1(f_i))
        f_t_trans = self.fc2(self.fc1(f_t))

        f_v_trans = F.normalize(f_v_trans)
        f_i_trans = F.normalize(f_i_trans)
        f_t_trans = F.normalize(f_t_trans)

        # 3. 计算动态权重
        if self.variance_aware:
            beta = self.dynamic_weight(f_v, f_i)
            i_t_weight = beta * self.weights[1]
        else:
            i_t_weight = self.weights[1]

        # 4. 双路径损失计算
        loss_v_t = self.calculate_path_loss(f_v_trans, f_t_trans, label, self.weights[0])
        loss_i_t = self.lambda_i * self.calculate_path_loss(f_i_trans, f_t_trans, label, i_t_weight)

        total_loss = loss_v_t + loss_i_t

        return total_loss

    def calculate_path_loss(self, anchor, target, label, weight):
        """计算单一路径损失"""
        loss = 0
        label_num = len(label.unique())
        k = int(label.size(0) / label_num)

        for i in range(len(anchor)):
            current_label = label[i]
            target_indices = (label == current_label)
            same_id_targets = target[target_indices]

            # 加权距离计算
            for j, t_feat in enumerate(same_id_targets):
                if j == i % k:  # 同位置样本
                    loss += weight * (anchor[i] - t_feat).norm(p=2)
                else:  # 同ID不同位置
                    loss += (1 - weight) / (k - 1) * (anchor[i] - t_feat).norm(p=2)

        return loss / len(anchor)



class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge myloss0.5
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct    
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative  = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha =1, gamma = 1, square = 0):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap*self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an*self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        
        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)
        
        # squared difference
        if self.square ==0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow =torch.clamp_max(diff_pow, max=88)

            # Compute ranking hinge myloss0.5
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            
            loss = self.ranking_loss(diff_pow, y)
        
        # myloss0.5 = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

      
class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss
       
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx