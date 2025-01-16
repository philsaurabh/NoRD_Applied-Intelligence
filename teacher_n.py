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
from torchvision.datasets import CIFAR100
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy
from models import model_dict
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

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train teacher network.')
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=128)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150,180,210])

parser.add_argument('--save-interval', type=int, default=40)
parser.add_argument('--arch', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

exp_name = 'teacher6vs_{}_seed{}'.format(args.arch, args.seed)
exp_path = './experiments6vs/{}'.format(exp_name)
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


class cifar100Nosiy(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=0):
        super(cifar100Nosiy, self).__init__(root, download=download, transform=transform, target_transform=target_transform)
        self.download = download
        if asym:
            """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
            """
            nb_classes = 100
            P = np.eye(nb_classes)
            n = nosiy_rate
            nb_superclasses = 20
            nb_subclasses = 5

            if n > 0.0:
                for i in np.arange(nb_superclasses):
                    init, end = i * nb_subclasses, (i+1) * nb_subclasses
                    P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                    y_train_noisy = multiclass_noisify(np.array(self.targets), P=P, random_state=seed)
                    actual_noise = (y_train_noisy != np.array(self.targets)).mean()
                assert actual_noise > 0.0
                print('Actual noise %.2f' % actual_noise)
                self.targets = y_train_noisy.tolist()
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
            class_noisy = int(n_noisy / 100)
            noisy_idx = []
            for d in range(100):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=100, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(100):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

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


trainset = cifar100Nosiy('./data', train=True, transform=transform_train, download=True, asym=False, seed =123, nosiy_rate=0.6)
valset = CIFAR100('./data', train=False, transform=transform_test, download=True)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=4, pin_memory=False)

'''
gt_labels = trainset.targets#.clone()
#generate noise
print(gt_labels)
transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise

def generate_noise(nr, gt_labels, noise_mode='sym'):
  noise_label = []
  idx = list(range(len(gt_labels)))
  random.shuffle(idx)
  num_noise = int(nr*len(gt_labels))            
  noise_idx = idx[:num_noise]
  for i in range(len(gt_labels)):
      if i in noise_idx:
        if noise_mode=='sym':
          noiselabel = random.randint(0,9)  
          noise_label.append(noiselabel)
        #elif noise_mode=='asym':   
        #  noiselabel = transition[gt_labels[i].item()]
        #  noise_label.append(noiselabel)                    
      else:    
        noise_label.append(gt_labels[i])  
  return torch.tensor(noise_label)

#generate noisy_benchmark
label_noise={}
label_noise['50_sym']=generate_noise(0.8, gt_labels, 'sym')
#label_noise['40_asym']=generate_noise(0.4, gt_labels, 'asym')
print('real noise rate:{:.2f}'.format((np.array(label_noise['50_sym'])!=np.array(gt_labels)).sum()/len(gt_labels)))

#train with 50% noise rate
train_loader.dataset.targets = label_noise['50_sym']#.clone()  #update current labels
'''
from vit_pytorch import ViT
model = model_dict[args.arch](num_classes=100).cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

logger = SummaryWriter(osp.join(exp_path, 'events'))
best_acc = -1
for epoch in range(args.epoch):

    model.train()
    loss_record = AverageMeter()
    acc_record = AverageMeter()
    
    start = time.time()
    for x, target in train_loader:

        optimizer.zero_grad()
        x = x.cuda()
        target = target.cuda()

        output = model(x)
        
        q=0.7
        num_classes=100
        scale=1.0
        '''
        #loss1 = F.cross_entropy(output[nor_index], target)
        ce = F.cross_entropy(output, target)

        # RCE
        pred = F.softmax(output, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(target, num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), q)
        denominators = num_classes - pred.pow(q).sum(dim=1)
        ngce = numerators / denominators
        normalizor = 1 / 4 * (num_classes - 1)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss = 0.5 * scale * ngce.mean() + 0.5 *normalizor* rce.mean()
        
        #loss = F.cross_entropy(output, target)
        pred = F.log_softmax(output, dim=1)
        label_one_hot = torch.nn.functional.one_hot(target, num_classes).float().cuda()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        
        
        pred1 = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(target, num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred1 * torch.log(label_one_hot), dim=1))
        
        loss= 10 * nce.mean()+ 0.1 * rce.mean()
        '''
        q=0.7
        num_classes=100
        scale=1.0
        #loss1 = F.cross_entropy(output[nor_index], target)
        
        #ce = F.cross_entropy(output, target)
        pred = F.log_softmax(output, dim=1)
        label_one_hot = torch.nn.functional.one_hot(target, num_classes).float().cuda()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
       
        
        
        pred = F.softmax(output, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(target, num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        #return self.scale * rce.mean()

        '''# RCE
        pred = F.softmax(output, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(target, num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), q)
        denominators = num_classes - pred.pow(q).sum(dim=1)
        ngce = numerators / denominators
        normalizor = 1 / 4 * (num_classes - 1)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        '''
        loss = 10.0 * scale * nce.mean() + 0.1 * rce.mean()
        
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(output, target, topk=(1,))[0]
        loss_record.update(loss.item(), x.size(0))
        acc_record.update(batch_acc.item(), x.size(0))

    logger.add_scalar('train/cls_loss', loss_record.avg, epoch+1)
    logger.add_scalar('train/cls_acc', acc_record.avg, epoch+1)

    run_time = time.time() - start

    info = 'train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\t'.format(
        epoch+1, args.epoch, run_time, loss_record.avg, acc_record.avg)
    print(info)

    model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in val_loader:

        x = x.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(x)
            loss = F.cross_entropy(output, target)

        batch_acc = accuracy(output, target, topk=(1,))[0]
        loss_record.update(loss.item(), x.size(0))
        acc_record.update(batch_acc.item(), x.size(0))

    run_time = time.time() - start

    logger.add_scalar('val/cls_loss', loss_record.avg, epoch+1)
    logger.add_scalar('val/cls_acc', acc_record.avg, epoch+1)

    info = 'test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\n'.format(
            epoch+1, args.epoch, run_time, loss_record.avg, acc_record.avg)
    print(info)
    
    scheduler.step()

    # save checkpoint
    if (epoch+1) in args.milestones or epoch+1==args.epoch or (epoch+1)%args.save_interval==0:
        state_dict = dict(epoch=epoch+1, state_dict=model.state_dict(), acc=acc_record.avg)
        name = osp.join(exp_path, 'ckpt/{:03d}.pth'.format(epoch+1))
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)

    # save best
    if acc_record.avg > best_acc:
        state_dict = dict(epoch=epoch+1, state_dict=model.state_dict(), acc=acc_record.avg)
        name = osp.join(exp_path, 'ckpt/best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
        best_acc = acc_record.avg

print('best_acc: {:.2f}'.format(best_acc))
