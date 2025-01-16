import os
import os.path as osp
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import random
from TruncatedLoss import TruncatedLoss
from utils import AverageMeter, accuracy
from wrapper import wrapper
from cifar import CIFAR100
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from numpy.testing import assert_array_almost_equal
import numpy as np
import os
import torch
import random
import mlconfig

from models import model_dict

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train SSKD student network.')
parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--t-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=128)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--t-lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150,180,210])
parser.add_argument('--t-milestones', type=int, nargs='+', default=[30,45])

parser.add_argument('--save-interval', type=int, default=40)
parser.add_argument('--ce-weight', type=float, default=0.5) # cross-entropy
parser.add_argument('--kd-weight', type=float, default=0.5) # knowledge distillation
parser.add_argument('--tf-weight', type=float, default=2.7) # transformation
parser.add_argument('--ss-weight', type=float, default=21.0) # self-supervision

parser.add_argument('--kd-T', type=float, default=4.0) # temperature in KD
parser.add_argument('--tf-T', type=float, default=4.0) # temperature in LT
parser.add_argument('--ss-T', type=float, default=0.5) # temperature in SS

parser.add_argument('--ratio-tf', type=float, default=1.0) # keep how many wrong predictions of LT
parser.add_argument('--ratio-ss', type=float, default=0.75) # keep how many wrong predictions of SS
parser.add_argument('--s-arch', type=str) # student architecture
parser.add_argument('--t-path', type=str) # teacher checkpoint path

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


t_name = osp.abspath(args.t_path).split('/')[-1]
t_arch = '_'.join(t_name.split('_')[1:-1])
exp_name = 'sskd_student_{}_weight{}+{}+{}+{}_T{}+{}+{}_ratio{}+{}_seed{}_{}'.format(\
            args.s_arch, \
            args.ce_weight, args.kd_weight, args.tf_weight, args.ss_weight, \
            args.kd_T, args.tf_T, args.ss_T, \
            args.ratio_tf, args.ratio_ss, \
            args.seed, t_name)
exp_path = './experiments/{}'.format(exp_name)
os.makedirs(exp_path, exist_ok=True)




def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data, _ in tqdm(loader):

        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])

trainset = CIFAR100('./data', train=True, transform=transform_train)#, download=True, asym=False, seed =123, nosiy_rate=0.8)
nosiy_rate =0.8
n_samples = len(trainset.targets)
n_noisy = int(nosiy_rate * n_samples)
print("%d Noisy samples" % (n_noisy))
class_index = [np.where(np.array(trainset.targets) == i)[0] for i in range(100)]
class_noisy = int(n_noisy / 100)
noisy_idx = []
for d in range(100):
    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
    noisy_idx.extend(noisy_class_index)
    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
for i in noisy_idx:
    trainset.targets[i] = other_class(n_classes=100, current_class=trainset.targets[i])
print(len(noisy_idx))
print("Print noisy label generation statistics:")
for i in range(100):
    n_noisy = np.sum(np.array(trainset.targets) == i)
    print("Noisy class %s, has %s samples." % (i, n_noisy))


valset = CIFAR100('./data', train=False, transform=transform_test)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=4, pin_memory=False)


ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
t_model = model_dict[t_arch](num_classes=100).cuda()
state_dict = torch.load(ckpt_path)['state_dict']
t_model.load_state_dict(state_dict)
t_model = wrapper(module=t_model).cuda()

t_optimizer = optim.SGD([{'params':t_model.backbone.parameters(), 'lr':0.0},
                        {'params':t_model.proj_head.parameters(), 'lr':args.t_lr}],
                        momentum=args.momentum, weight_decay=args.weight_decay)
t_model.eval()
t_scheduler = MultiStepLR(t_optimizer, milestones=args.t_milestones, gamma=args.gamma)

logger = SummaryWriter(osp.join(exp_path, 'events'))

acc_record = AverageMeter()
loss_record = AverageMeter()
start = time.time()
for x, target in val_loader:

    x = x[:,0,:,:,:].cuda()
    target = target.cuda()
    with torch.no_grad():
        output, _, feat = t_model(x)
        loss = F.cross_entropy(output, target)

    batch_acc = accuracy(output, target, topk=(1,))[0]
    acc_record.update(batch_acc.item(), x.size(0))
    loss_record.update(loss.item(), x.size(0))

run_time = time.time() - start
info = 'teacher cls_acc:{:.2f}\n'.format(acc_record.avg)
print(info)

# train ssp_head
for epoch in range(args.t_epoch):

    t_model.eval()
    loss_record = AverageMeter()
    acc_record = AverageMeter()

    start = time.time()
    for x, _ in train_loader:

        t_optimizer.zero_grad()

        x = x.cuda()
        c,h,w = x.size()[-3:]
        x = x.view(-1, c, h, w)

        _, rep, feat = t_model(x, bb_grad=False)
        batch = int(x.size(0) / 4)
        nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
        aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

        nor_rep = rep[nor_index]
        aug_rep = rep[aug_index]
        nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
        simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
        target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        loss = F.cross_entropy(simi, target)

        loss.backward()
        t_optimizer.step()

        batch_acc = accuracy(simi, target, topk=(1,))[0]
        loss_record.update(loss.item(), 3*batch)
        acc_record.update(batch_acc.item(), 3*batch)

    logger.add_scalar('train/teacher_ssp_loss', loss_record.avg, epoch+1)
    logger.add_scalar('train/teacher_ssp_acc', acc_record.avg, epoch+1)

    run_time = time.time() - start
    info = 'teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\t'.format(
        epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
    print(info)

    t_model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, _ in val_loader:

        x = x.cuda()
        c,h,w = x.size()[-3:]
        x = x.view(-1, c, h, w)

        with torch.no_grad():
            _, rep, feat = t_model(x)
        batch = int(x.size(0) / 4)
        nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
        aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

        nor_rep = rep[nor_index]
        aug_rep = rep[aug_index]
        nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
        simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
        target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        loss = F.cross_entropy(simi, target)

        batch_acc = accuracy(simi, target, topk=(1,))[0]
        acc_record.update(batch_acc.item(),3*batch)
        loss_record.update(loss.item(), 3*batch)

    run_time = time.time() - start
    logger.add_scalar('val/teacher_ssp_loss', loss_record.avg, epoch+1)
    logger.add_scalar('val/teacher_ssp_acc', acc_record.avg, epoch+1)

    info = 'ssp_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\n'.format(
            epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
    print(info)

    t_scheduler.step()


name = osp.join(exp_path, 'ckpt/teacher.pth')
os.makedirs(osp.dirname(name), exist_ok=True)
torch.save(t_model.state_dict(), name)


s_model = model_dict[args.s_arch](num_classes=100)
s_model = wrapper(module=s_model).cuda()
optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

best_acc = 0
for epoch in range(args.epoch):

    # train
    s_model.train()
    loss1_record = AverageMeter()
    loss2_record = AverageMeter()
    loss3_record = AverageMeter()
    loss4_record = AverageMeter()
    cls_acc_record = AverageMeter()
    ssp_acc_record = AverageMeter()
    
    start = time.time()
    for x, target in train_loader:

        optimizer.zero_grad()

        c,h,w = x.size()[-3:]
        x = x.view(-1,c,h,w).cuda()
        target = target.cuda()

        batch = int(x.size(0) / 4)
        nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
        aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

        output, s_feat, _ = s_model(x, bb_grad=True)
        log_nor_output = F.log_softmax(output[nor_index] / args.kd_T, dim=1)
        log_aug_output = F.log_softmax(output[aug_index] / args.tf_T, dim=1)
        with torch.no_grad():
            knowledge, t_feat, _ = t_model(x)
            nor_knowledge = F.softmax(knowledge[nor_index] / args.kd_T, dim=1)
            aug_knowledge = F.softmax(knowledge[aug_index] / args.tf_T, dim=1)

        # error level ranking
        aug_target = target.unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        rank = torch.argsort(aug_knowledge, dim=1, descending=True)
        rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        index = torch.argsort(rank)
        tmp = torch.nonzero(rank, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = 3*batch - wrong_num
        wrong_keep = int(wrong_num * args.ratio_tf)
        index = index[:correct_num+wrong_keep]
        distill_index_tf = torch.sort(index)[0]

        s_nor_feat = s_feat[nor_index]
        s_aug_feat = s_feat[aug_index]
        s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
        s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)

        t_nor_feat = t_feat[nor_index]
        t_aug_feat = t_feat[aug_index]
        t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
        t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)

        t_simi = t_simi.detach()
        aug_target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        rank = torch.argsort(t_simi, dim=1, descending=True)
        rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        index = torch.argsort(rank)
        tmp = torch.nonzero(rank, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = 3*batch - wrong_num
        wrong_keep = int(wrong_num * args.ratio_ss)
        index = index[:correct_num+wrong_keep]
        distill_index_ss = torch.sort(index)[0]

        log_simi = F.log_softmax(s_simi / args.ss_T, dim=1)
        simi_knowledge = F.softmax(t_simi / args.ss_T, dim=1)
        
        q=0.7
        num_classes=100
        scale=1.0
        #loss1 = F.cross_entropy(output[nor_index], target)
        ce = F.cross_entropy(output[nor_index], target)

        # RCE
        pred = F.softmax(output[nor_index], dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(target, num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), q)
        denominators = num_classes - pred.pow(q).sum(dim=1)
        ngce = numerators / denominators
        normalizor = 1 / 4 * (num_classes - 1)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss1 = 10.0 * scale * ngce.mean() + 0.1 *normalizor* rce.mean()
        loss2 = F.kl_div(log_nor_output, nor_knowledge, reduction='batchmean') * args.kd_T * args.kd_T #+ F.cosine_similarity(log_nor_output.view(1,-1), nor_knowledge.view(1,-1), dim=1)* args.ss_T * args.ss_T
        loss3 = F.kl_div(log_aug_output[distill_index_tf], aug_knowledge[distill_index_tf], \
                        reduction='batchmean') * args.tf_T * args.tf_T

        loss4 = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss], \
                        reduction='batchmean') * args.ss_T * args.ss_T + F.cosine_similarity(log_simi[distill_index_ss].view(1,-1), simi_knowledge[distill_index_ss].view(1,-1), dim=1)* args.ss_T * args.ss_T
        loss = args.ce_weight * loss1 + args.kd_weight * loss2 + args.tf_weight * loss3 + args.ss_weight * loss4

        loss.backward()
        optimizer.step()

        cls_batch_acc = accuracy(output[nor_index], target, topk=(1,))[0]
        ssp_batch_acc = accuracy(s_simi, aug_target, topk=(1,))[0]
        loss1_record.update(loss1.item(), batch)
        loss2_record.update(loss2.item(), batch)
        loss3_record.update(loss3.item(), len(distill_index_tf))
        loss4_record.update(loss4.item(), len(distill_index_ss))
        cls_acc_record.update(cls_batch_acc.item(), batch)
        ssp_acc_record.update(ssp_batch_acc.item(), 3*batch)

    logger.add_scalar('train/ce_loss', loss1_record.avg, epoch+1)
    logger.add_scalar('train/kd_loss', loss2_record.avg, epoch+1)
    logger.add_scalar('train/tf_loss', loss3_record.avg, epoch+1)
    logger.add_scalar('train/ss_loss', loss4_record.avg, epoch+1)
    logger.add_scalar('train/cls_acc', cls_acc_record.avg, epoch+1)
    logger.add_scalar('train/ss_acc', ssp_acc_record.avg, epoch+1)

    run_time = time.time() - start
    info = 'student_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ce_loss:{:.3f}\t kd_loss:{:.3f}\t cls_acc:{:.2f}'.format(
        epoch+1, args.epoch, run_time, loss1_record.avg, loss2_record.avg, cls_acc_record.avg)
    print(info)

    # cls val
    s_model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in val_loader:

        x = x[:,0,:,:,:].cuda()
        target = target.cuda()
        with torch.no_grad():
            output, _, feat = s_model(x)
            loss = F.cross_entropy(output, target)

        batch_acc = accuracy(output, target, topk=(1,))[0]
        acc_record.update(batch_acc.item(), x.size(0))
        loss_record.update(loss.item(), x.size(0))

    run_time = time.time() - start
    logger.add_scalar('val/ce_loss', loss_record.avg, epoch+1)
    logger.add_scalar('val/cls_acc', acc_record.avg, epoch+1)

    info = 'student_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_acc:{:.2f}\n'.format(
            epoch+1, args.epoch, run_time, acc_record.avg)
    print(info)

    if acc_record.avg > best_acc:
        best_acc = acc_record.avg
        state_dict = dict(epoch=epoch+1, state_dict=s_model.state_dict(), best_acc=best_acc)
        name = osp.join(exp_path, 'ckpt/student_best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
    
    scheduler.step()
print('best_acc: {:.2f}'.format(best_acc))
